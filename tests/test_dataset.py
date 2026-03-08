"""Tests for dataset.py: tokeniser, CDRDataset, DataLoader."""

from __future__ import annotations

import torch
from antibody_seq_ml.dataset import (
    AminoAcidTokeniser,
    CDRDataset,
    build_dataloaders,
)


def test_tokeniser_known_sequence(tokeniser: AminoAcidTokeniser) -> None:
    """Known AAs should map to non-zero, non-UNK indices."""
    seq = "ACDE"
    ids = tokeniser.encode(seq, max_len=10)
    assert ids.shape == (10,)
    # PAD fills the tail
    assert all(ids[4:] == tokeniser.PAD_IDX)
    # Each AA maps to a distinct, valid index (not UNK, not PAD)
    for i in range(4):
        assert ids[i] != tokeniser.PAD_IDX
        assert ids[i] != tokeniser.UNK_IDX


def test_tokeniser_unknown_aa(tokeniser: AminoAcidTokeniser) -> None:
    """Non-standard AA 'X' should map to UNK_IDX."""
    ids = tokeniser.encode("AXC", max_len=5)
    assert int(ids[1]) == tokeniser.UNK_IDX


def test_padding_to_max_length(tokeniser: AminoAcidTokeniser) -> None:
    """Short sequence should be right-padded to max_len."""
    ids = tokeniser.encode("AC", max_len=10)
    assert ids.shape == (10,)
    assert ids[0] != tokeniser.PAD_IDX
    assert ids[1] != tokeniser.PAD_IDX
    assert all(ids[2:] == tokeniser.PAD_IDX)


def test_truncation_to_max_length(tokeniser: AminoAcidTokeniser) -> None:
    """Long sequence should be silently truncated to max_len."""
    long_seq = "ACDEFGHIKLM"  # 11 chars
    ids = tokeniser.encode(long_seq, max_len=5)
    assert ids.shape == (5,)
    assert all(ids != tokeniser.PAD_IDX)  # no padding needed


def test_dataloader_batch_shapes(smoke_cfg, smoke_df) -> None:
    """DataLoader batches must have expected tensor shapes."""
    train_loader, _, _ = build_dataloaders(smoke_cfg, "data/smoke/sequences_smoke.csv")
    batch = next(iter(train_loader))

    B = smoke_cfg.data.batch_size
    L = smoke_cfg.data.max_seq_length
    assert batch["input_ids"].shape == (B, L)
    assert batch["length_label"].shape == (B,)
    assert batch["hydro_score"].shape == (B,)
    assert batch["length_label"].dtype == torch.long
    assert batch["hydro_score"].dtype == torch.float32


def test_stratified_split_proportions(smoke_cfg, smoke_df) -> None:
    """Train/val/test split should approximate 70/15/15."""
    train_loader, val_loader, test_loader = build_dataloaders(
        smoke_cfg, "data/smoke/sequences_smoke.csv"
    )
    n_total = len(smoke_df)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)  # type: ignore[arg-type]
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]

    assert n_train + n_val + n_test == n_total
    # Allow ±5% tolerance
    assert abs(n_train / n_total - 0.70) < 0.05
    assert abs(n_val / n_total - 0.15) < 0.05
    assert abs(n_test / n_total - 0.15) < 0.05


def test_dataset_length(smoke_dataset: CDRDataset, smoke_df) -> None:
    assert len(smoke_dataset) == len(smoke_df)


def test_dataset_item_types(smoke_dataset: CDRDataset) -> None:
    item = smoke_dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["length_label"], torch.Tensor)
    assert isinstance(item["hydro_score"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long
    assert item["length_label"].dtype == torch.long
    assert item["hydro_score"].dtype == torch.float32
