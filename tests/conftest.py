"""Shared pytest fixtures and test environment setup."""

from __future__ import annotations

import os

import pandas as pd
import pytest
import torch
from antibody_seq_ml.dataset import AminoAcidTokeniser, CDRDataset
from omegaconf import OmegaConf

# Disable W&B for all tests
os.environ["WANDB_MODE"] = "disabled"

SMOKE_DATA_PATH = "data/smoke/sequences_smoke.csv"


@pytest.fixture(scope="session")
def tokeniser() -> AminoAcidTokeniser:
    return AminoAcidTokeniser()


@pytest.fixture(scope="session")
def smoke_df() -> pd.DataFrame:
    return pd.read_csv(SMOKE_DATA_PATH)


@pytest.fixture(scope="session")
def smoke_dataset(smoke_df: pd.DataFrame, tokeniser: AminoAcidTokeniser) -> CDRDataset:
    return CDRDataset(smoke_df, tokeniser, max_len=30)


@pytest.fixture(scope="session")
def smoke_cfg():
    """Smoke-scale omegaconf config (CPU, tiny model)."""
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.embedding_dim = 64
    cfg.model.d_model = 64
    cfg.model.nhead = 4
    cfg.model.hidden_dim = 128
    cfg.model.num_layers = 2
    cfg.model.dim_feedforward = 256
    cfg.training.epochs = 3
    cfg.training.patience = 10
    cfg.data.batch_size = 32
    cfg.data.max_seq_length = 30
    cfg.wandb.mode = "disabled"
    return cfg


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")
