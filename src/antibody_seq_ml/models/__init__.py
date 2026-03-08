"""Model registry and factory."""

from __future__ import annotations

from antibody_seq_ml.models.base import SequenceModel, build_model
from antibody_seq_ml.models.lstm import BiLSTMModel
from antibody_seq_ml.models.transformer import TransformerModel

__all__ = ["SequenceModel", "build_model", "BiLSTMModel", "TransformerModel"]
