"""Bidirectional LSTM sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from antibody_seq_ml.dataset import VOCAB_SIZE
from antibody_seq_ml.models.base import SequenceModel


class BiLSTMModel(SequenceModel):
    """Embedding → BiLSTM → masked mean pooling → dual prediction heads.

    Smoke variant (CPU):  embedding_dim=64,  hidden_dim=128, num_layers=2  (~500K params)
    Full variant (GPU):   embedding_dim=256, hidden_dim=512, num_layers=3  (~8M params)
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

        embedding_dim: int = cfg.model.embedding_dim
        hidden_dim: int = cfg.model.hidden_dim
        num_layers: int = cfg.model.num_layers
        dropout: float = cfg.model.dropout

        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM output dim = hidden_dim * 2 (forward + backward)
        out_dim = hidden_dim * 2
        self.class_head = nn.Linear(out_dim, 3)
        self.hydro_head = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # input_ids: [B, max_len]
        embedded = self.embedding(input_ids)  # [B, max_len, emb_dim]
        lstm_out, _ = self.lstm(embedded)  # [B, max_len, hidden*2]

        # Masked mean pooling — exclude PAD positions (index 0)
        mask = (input_ids != 0).float().unsqueeze(-1)  # [B, max_len, 1]
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = (lstm_out * mask).sum(dim=1) / lengths  # [B, hidden*2]
        pooled = self.dropout(pooled)

        class_logits = self.class_head(pooled)  # [B, 3]
        hydro_pred = self.hydro_head(pooled).squeeze(-1)  # [B]

        return {"class_logits": class_logits, "hydro_pred": hydro_pred}
