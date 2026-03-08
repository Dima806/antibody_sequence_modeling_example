"""Evaluation utilities: metrics, confusion matrix, attention visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, f1_score, r2_score
from torch.utils.data import DataLoader

from antibody_seq_ml.models.base import SequenceModel


def evaluate(
    model: SequenceModel,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig | None = None,
) -> dict[str, float]:
    """Run inference on loader and return classification + regression metrics.

    Returns:
        dict with keys: cls_acc, cls_f1, hydro_mse, hydro_r2, loss
    """
    cls_loss_weight = cfg.training.cls_loss_weight if cfg is not None else 1.0
    reg_loss_weight = cfg.training.reg_loss_weight if cfg is not None else 0.1

    model.eval()
    all_cls_preds: list[int] = []
    all_cls_labels: list[int] = []
    all_hydro_preds: list[float] = []
    all_hydro_labels: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            length_labels = batch["length_label"].to(device)
            hydro_scores = batch["hydro_score"].to(device)

            output = model(input_ids)
            cls_loss = F.cross_entropy(output["class_logits"], length_labels)
            reg_loss = F.mse_loss(output["hydro_pred"], hydro_scores)
            loss = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss

            total_loss += loss.item()
            preds = output["class_logits"].argmax(dim=1)
            all_cls_preds.extend(preds.cpu().numpy().tolist())
            all_cls_labels.extend(length_labels.cpu().numpy().tolist())
            all_hydro_preds.extend(output["hydro_pred"].cpu().numpy().tolist())
            all_hydro_labels.extend(hydro_scores.cpu().numpy().tolist())

    arr_preds = np.array(all_cls_preds)
    arr_labels = np.array(all_cls_labels)
    arr_hydro_preds = np.array(all_hydro_preds)
    arr_hydro_labels = np.array(all_hydro_labels)

    cls_acc = float(np.mean(arr_preds == arr_labels))
    cls_f1 = float(f1_score(arr_labels, arr_preds, average="weighted", zero_division=0))
    hydro_mse = float(np.mean((arr_hydro_preds - arr_hydro_labels) ** 2))
    try:
        hydro_r2 = float(r2_score(arr_hydro_labels, arr_hydro_preds))
    except Exception:
        hydro_r2 = 0.0

    return {
        "cls_acc": cls_acc,
        "cls_f1": cls_f1,
        "hydro_mse": hydro_mse,
        "hydro_r2": hydro_r2,
        "loss": total_loss / max(len(loader), 1),
    }


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot and optionally save a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix (normalised)",
    )
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_attention_weights(
    attn_weights: torch.Tensor,
    sequence: str,
    head: int = 0,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualise per-head attention weights as a heatmap over sequence positions.

    Args:
        attn_weights: Tensor of shape [num_heads, seq_len, seq_len].
        sequence: Raw amino acid string (for axis labels).
        head: Which attention head to plot.
        save_path: If given, save figure to this path.
    """
    weights = attn_weights[head].cpu().numpy()  # [seq_len, seq_len]
    labels = ["[CLS]"] + list(sequence)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) // 2), max(6, len(labels) // 2)))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)

    ticks = range(len(labels))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=labels,
        yticklabels=labels,
        title=f"Attention weights — head {head}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def extract_attention_weights(
    model: TransformerModel,  # noqa: F821
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Hook into TransformerEncoder to capture attention weights from the last layer.

    Returns:
        Tensor of shape [num_heads, seq_len+1, seq_len+1] (includes CLS).
    """
    # Access the underlying self_attn of the last encoder layer
    last_layer = model.encoder.layers[-1]
    B = input_ids.size(0)
    embedded = model.embedding(input_ids)
    cls_tokens = model.cls_token.expand(B, -1, -1)
    embedded = torch.cat([cls_tokens, embedded], dim=1)
    embedded = model.pos_encoding(embedded)

    pad_mask_seq = input_ids == 0
    cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
    src_key_padding_mask = torch.cat([cls_mask, pad_mask_seq], dim=1)

    # Run through all layers except last, then call last layer's self_attn manually
    x = embedded
    for layer in model.encoder.layers[:-1]:
        x = layer(x, src_key_padding_mask=src_key_padding_mask)

    # Last layer: get attention weights
    attn_out, attn_weights = last_layer.self_attn(
        x,
        x,
        x,
        key_padding_mask=src_key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    return attn_weights.squeeze(0)  # [num_heads, S+1, S+1]
