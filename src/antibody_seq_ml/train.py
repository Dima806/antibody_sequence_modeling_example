"""Training loop with W&B logging, early stopping, and checkpointing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from antibody_seq_ml.dataset import build_dataloaders
from antibody_seq_ml.models.base import SequenceModel, build_model


def compute_grad_norm(model: SequenceModel) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def _run_epoch(
    model: SequenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: DictConfig,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    """Run one pass over loader. Returns dict of averaged metrics."""
    all_cls_preds: list[int] = []
    all_cls_labels: list[int] = []
    all_hydro_preds: list[float] = []
    all_hydro_labels: list[float] = []
    total_loss = total_cls_loss = total_reg_loss = 0.0

    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            length_labels = batch["length_label"].to(device)
            hydro_scores = batch["hydro_score"].to(device)

            if training and optimizer is not None:
                optimizer.zero_grad()

            # ESM-2 expects raw string sequences; all other models take token tensors
            if type(model).__name__ == "ESM2FineTuned":
                output = model(batch["sequence"])
            else:
                output = model(input_ids)
            cls_loss = F.cross_entropy(output["class_logits"], length_labels)
            reg_loss = F.mse_loss(output["hydro_pred"], hydro_scores)
            loss = (
                cfg.training.cls_loss_weight * cls_loss
                + cfg.training.reg_loss_weight * reg_loss
            )

            if training and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()

            preds = output["class_logits"].argmax(dim=1)
            all_cls_preds.extend(preds.detach().cpu().numpy().tolist())
            all_cls_labels.extend(length_labels.cpu().numpy().tolist())
            all_hydro_preds.extend(output["hydro_pred"].detach().cpu().numpy().tolist())
            all_hydro_labels.extend(hydro_scores.cpu().numpy().tolist())

    n_batches = max(len(loader), 1)
    cls_acc = float(np.mean(np.array(all_cls_preds) == np.array(all_cls_labels)))
    try:
        hydro_r2 = float(r2_score(all_hydro_labels, all_hydro_preds))
    except Exception:
        hydro_r2 = 0.0

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "reg_loss": total_reg_loss / n_batches,
        "cls_acc": cls_acc,
        "hydro_r2": hydro_r2,
    }


def train(
    model: SequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    test_loader: DataLoader | None = None,
    wandb_run=None,
) -> dict[str, float]:
    """Full training loop with early stopping and optional W&B logging.

    Args:
        model: Instantiated SequenceModel already on the target device.
        train_loader / val_loader: DataLoaders from build_dataloaders().
        cfg: omegaconf DictConfig (see configs/default.yaml).
        test_loader: Optional; if provided, evaluate on test set after training.
        wandb_run: Active wandb.Run (or None to skip W&B logging).

    Returns:
        dict of final metrics (best_val_loss, plus test metrics if test_loader given).
    """
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.scheduler == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs
            )
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore[assignment]
            optimizer, mode="min", patience=2, factor=0.5
        )

    if wandb_run is not None:
        hardware = (
            f"gpu:{torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else "cpu"
        )
        wandb_run.config.update(
            {
                "hardware": hardware,
                "model_params": model.count_parameters(),
            },
            allow_val_change=True,
        )

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, cfg, device, training=True
        )
        grad_norm = compute_grad_norm(model)
        val_metrics = _run_epoch(model, val_loader, None, cfg, device, training=False)

        if cfg.training.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])  # type: ignore[call-arg]

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d}/{cfg.training.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['cls_acc']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['cls_acc']:.3f} | "
            f"lr={current_lr:.2e}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/cls_loss": train_metrics["cls_loss"],
                    "train/reg_loss": train_metrics["reg_loss"],
                    "train/cls_acc": train_metrics["cls_acc"],
                    "train/hydro_r2": train_metrics["hydro_r2"],
                    "val/loss": val_metrics["loss"],
                    "val/cls_loss": val_metrics["cls_loss"],
                    "val/reg_loss": val_metrics["reg_loss"],
                    "val/cls_acc": val_metrics["cls_acc"],
                    "val/hydro_r2": val_metrics["hydro_r2"],
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                }
            )

        # Checkpointing and early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    result: dict[str, float] = {"best_val_loss": best_val_loss}

    if test_loader is not None:
        # Load best checkpoint for final evaluation
        ckpt_path = checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        from antibody_seq_ml.evaluate import evaluate

        test_metrics = evaluate(model, test_loader, device, cfg)
        result.update({f"test/{k}": v for k, v in test_metrics.items()})

        print("\nTest metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        if wandb_run is not None:
            wandb_run.summary.update({f"test/{k}": v for k, v in test_metrics.items()})

    return result


def _apply_smoke_overrides(cfg: DictConfig) -> None:
    """Reduce hyperparameters to CPU-friendly smoke scale (in-place)."""
    cfg.model.embedding_dim = 64
    cfg.model.d_model = 64
    cfg.model.nhead = 4
    cfg.model.hidden_dim = 128
    cfg.model.num_layers = 2
    cfg.model.dim_feedforward = 256
    cfg.training.epochs = 10
    cfg.data.batch_size = 32
    cfg.wandb.mode = "disabled"


if __name__ == "__main__":
    import wandb

    parser = argparse.ArgumentParser(description="Train AntibodySeqML model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True, help="Path to sequences CSV")
    parser.add_argument(
        "--smoke", action="store_true", help="Use smoke-scale hyperparameters"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.smoke:
        _apply_smoke_overrides(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, args.data)
    model = build_model(cfg, device)
    print(f"Model: {cfg.model.type} | Parameters: {model.count_parameters():,}")

    wandb_run = None
    if cfg.wandb.mode != "disabled":
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["gpu" if torch.cuda.is_available() else "cpu"],
        )

    train(
        model,
        train_loader,
        val_loader,
        cfg,
        test_loader=test_loader,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()
