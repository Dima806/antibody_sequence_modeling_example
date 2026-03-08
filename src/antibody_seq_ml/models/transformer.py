"""Transformer Encoder sequence model with sinusoidal positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

from antibody_seq_ml.dataset import VOCAB_SIZE
from antibody_seq_ml.models.base import SequenceModel


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learnable) sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


class TransformerModel(SequenceModel):
    """[CLS] + Embedding + sinusoidal PE → TransformerEncoder → CLS pooling → heads.

    Smoke variant (CPU):  d_model=64,  nhead=4, num_layers=3, dim_feedforward=256  (~300K params)
    Full variant (GPU):   d_model=256, nhead=8, num_layers=6, dim_feedforward=1024 (~12M params)
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

        d_model: int = cfg.model.d_model
        nhead: int = cfg.model.nhead
        num_layers: int = cfg.model.num_layers
        dim_feedforward: int = cfg.model.dim_feedforward
        dropout: float = cfg.model.dropout
        max_len: int = cfg.data.max_seq_length + 2  # +1 for CLS, +1 buffer

        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        # Learnable CLS token embedding — prepended before positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=max_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.class_head = nn.Linear(d_model, 3)
        self.hydro_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # input_ids: [B, max_len]
        B = input_ids.size(0)

        embedded = self.embedding(input_ids)  # [B, max_len, d_model]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        embedded = torch.cat([cls_tokens, embedded], dim=1)  # [B, max_len+1, d_model]

        # Add positional encoding
        embedded = self.pos_encoding(embedded)  # [B, max_len+1, d_model]

        # Padding mask: True = position should be IGNORED by attention
        # CLS position is never masked
        pad_mask_seq = input_ids == 0  # [B, max_len]
        cls_mask = torch.zeros(
            B, 1, dtype=torch.bool, device=input_ids.device
        )  # [B, 1]
        src_key_padding_mask = torch.cat(
            [cls_mask, pad_mask_seq], dim=1
        )  # [B, max_len+1]

        encoded = self.encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )  # [B, S+1, d_model]

        # CLS token is at position 0
        cls_output = encoded[:, 0, :]  # [B, d_model]

        class_logits = self.class_head(cls_output)  # [B, 3]
        hydro_pred = self.hydro_head(cls_output).squeeze(-1)  # [B]

        return {"class_logits": class_logits, "hydro_pred": hydro_pred}
