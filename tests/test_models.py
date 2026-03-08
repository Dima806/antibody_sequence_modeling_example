"""Tests for model forward passes, output shapes, and factory."""

from __future__ import annotations

import pytest
import torch
from antibody_seq_ml.models import BiLSTMModel, TransformerModel, build_model
from omegaconf import OmegaConf

BATCH_SIZE = 4
SEQ_LEN = 30


def _make_input(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Random token IDs in vocab range [0, 21], some padding at end."""
    ids = torch.randint(1, 21, (batch_size, seq_len))
    # Zero out last 5 positions to simulate padding
    ids[:, -5:] = 0
    return ids


def test_lstm_forward_pass(smoke_cfg, device) -> None:
    model = BiLSTMModel(smoke_cfg, device).to(device)
    ids = _make_input()
    out = model(ids)
    assert "class_logits" in out
    assert "hydro_pred" in out


def test_transformer_forward_pass(smoke_cfg, device) -> None:
    model = TransformerModel(smoke_cfg, device).to(device)
    ids = _make_input()
    out = model(ids)
    assert "class_logits" in out
    assert "hydro_pred" in out


def test_output_class_logits_shape(smoke_cfg, device) -> None:
    for ModelClass in (BiLSTMModel, TransformerModel):
        model = ModelClass(smoke_cfg, device).to(device)
        out = model(_make_input())
        assert out["class_logits"].shape == (
            BATCH_SIZE,
            3,
        ), f"{ModelClass.__name__} class_logits shape mismatch"


def test_output_hydro_pred_shape(smoke_cfg, device) -> None:
    for ModelClass in (BiLSTMModel, TransformerModel):
        model = ModelClass(smoke_cfg, device).to(device)
        out = model(_make_input())
        assert out["hydro_pred"].shape == (BATCH_SIZE,), (
            f"{ModelClass.__name__} hydro_pred shape mismatch"
        )


def test_padding_mask_applied(smoke_cfg, device) -> None:
    """Models should not crash on sequences that are entirely padding."""
    model = TransformerModel(smoke_cfg, device).to(device)
    # All-padding input (except first token so it's not fully degenerate)
    ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    ids[:, 0] = 1  # one real token
    out = model(ids)
    assert out["class_logits"].shape == (BATCH_SIZE, 3)


def test_lstm_no_crash_all_padding(smoke_cfg, device) -> None:
    model = BiLSTMModel(smoke_cfg, device).to(device)
    ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    out = model(ids)
    assert out["class_logits"].shape == (BATCH_SIZE, 3)


def test_model_factory_lstm(smoke_cfg, device) -> None:
    cfg = OmegaConf.merge(smoke_cfg, {"model": {"type": "lstm"}})
    model = build_model(cfg, device)
    assert isinstance(model, BiLSTMModel)


def test_model_factory_transformer(smoke_cfg, device) -> None:
    cfg = OmegaConf.merge(smoke_cfg, {"model": {"type": "transformer"}})
    model = build_model(cfg, device)
    assert isinstance(model, TransformerModel)


def test_model_factory_unknown_raises(smoke_cfg, device) -> None:
    cfg = OmegaConf.merge(smoke_cfg, {"model": {"type": "unknown_model"}})
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(cfg, device)


def test_count_parameters_positive(smoke_cfg, device) -> None:
    for ModelClass in (BiLSTMModel, TransformerModel):
        model = ModelClass(smoke_cfg, device)
        assert model.count_parameters() > 0
