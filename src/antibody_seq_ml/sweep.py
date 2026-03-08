"""W&B sweep agent entry point."""

from __future__ import annotations

import argparse
import os

import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from antibody_seq_ml.dataset import build_dataloaders
from antibody_seq_ml.models.base import build_model
from antibody_seq_ml.train import train

# Globals set by CLI before sweep agent starts
_DATA_PATH: str = ""
_BASE_CFG_PATH: str = "configs/default.yaml"


def build_cfg_from_wandb(
    wandb_cfg, base_cfg_path: str = "configs/default.yaml"
) -> DictConfig:
    """Merge wandb.config overrides onto the base YAML config."""
    base = OmegaConf.load(base_cfg_path)
    model_overrides: dict = {}
    training_overrides: dict = {}
    data_overrides: dict = {}

    mapping = {
        "model_type": ("model", "type"),
        "learning_rate": ("training", "learning_rate"),
        "dropout": ("model", "dropout"),
        "batch_size": ("data", "batch_size"),
        "num_layers": ("model", "num_layers"),
        "weight_decay": ("training", "weight_decay"),
    }

    cfg_dict = wandb_cfg if isinstance(wandb_cfg, dict) else dict(wandb_cfg)

    for wkey, (section, ckey) in mapping.items():
        if wkey in cfg_dict:
            if section == "model":
                model_overrides[ckey] = cfg_dict[wkey]
            elif section == "training":
                training_overrides[ckey] = cfg_dict[wkey]
            elif section == "data":
                data_overrides[ckey] = cfg_dict[wkey]

    # embedding_dim syncs to both model.embedding_dim and model.d_model
    if "embedding_dim" in cfg_dict:
        model_overrides["embedding_dim"] = cfg_dict["embedding_dim"]
        model_overrides["d_model"] = cfg_dict["embedding_dim"]

    overrides: dict = {}
    if model_overrides:
        overrides["model"] = model_overrides
    if training_overrides:
        overrides["training"] = training_overrides
    if data_overrides:
        overrides["data"] = data_overrides

    return OmegaConf.merge(base, OmegaConf.create(overrides))


def sweep_train_fn() -> None:
    """Called by wandb.agent for each sweep run."""
    with wandb.init() as run:
        cfg = build_cfg_from_wandb(dict(wandb.config), base_cfg_path=_BASE_CFG_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, _ = build_dataloaders(cfg, _DATA_PATH)
        model = build_model(cfg, device)
        train(model, train_loader, val_loader, cfg, wandb_run=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch W&B sweep")
    parser.add_argument("--config", default="configs/sweep_cpu.yaml")
    parser.add_argument("--data", required=True, help="Path to sequences CSV")
    parser.add_argument("--base-config", default="configs/default.yaml")
    args = parser.parse_args()

    _DATA_PATH = args.data
    _BASE_CFG_PATH = args.base_config

    sweep_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert isinstance(sweep_cfg, dict)
    count = sweep_cfg.pop("count", 5)

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    sweep_id = wandb.sweep(
        sweep=sweep_cfg, project="antibody-seq-ml", entity="dima806-team"
    )
    wandb.agent(sweep_id, function=sweep_train_fn, count=count)
