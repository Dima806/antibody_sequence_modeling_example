"""Base class and factory for all sequence models."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SequenceModel(nn.Module):
    """Abstract base class for CDR-H3 property prediction models.

    All subclasses must implement forward() returning a dict with:
        - "class_logits": Tensor[B, 3]  — CDR-H3 length class logits
        - "hydro_pred":   Tensor[B]     — hydrophobicity regression prediction
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: DictConfig, device: torch.device) -> SequenceModel:
    """Instantiate the model specified by cfg.model.type and move to device."""
    model_type = cfg.model.type

    if model_type == "lstm":
        from antibody_seq_ml.models.lstm import BiLSTMModel

        model: SequenceModel = BiLSTMModel(cfg, device)
    elif model_type == "transformer":
        from antibody_seq_ml.models.transformer import TransformerModel

        model = TransformerModel(cfg, device)
    elif model_type == "esm2":
        from antibody_seq_ml.models.esm2 import ESM2FineTuned

        model = ESM2FineTuned(cfg, device)
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Choose lstm | transformer | esm2"
        )

    return model.to(device)
