"""ESM-2 fine-tuning wrapper — GPU only (requires fair-esm package)."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from antibody_seq_ml.models.base import SequenceModel


class ESM2FineTuned(SequenceModel):
    """ESM-2 (esm2_t6_8M_UR50D, 8M params) as a partially-frozen feature extractor.

    Only the last `cfg.model.esm_trainable_layers` transformer layers and the
    task heads are trainable. The rest of ESM-2 remains frozen.

    Requires `fair-esm>=2.0.0` (not installed in the CPU Codespaces environment).
    Only instantiate this class when torch.cuda.is_available() is True.

    Input: raw AA sequence strings (handled via ESM batch_converter).
    Output: {"class_logits": Tensor[B, 3], "hydro_pred": Tensor[B]}
    """

    ESM_EMBED_DIM = 320  # esm2_t6_8M_UR50D output dimension

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ESM2FineTuned requires a CUDA GPU. "
                "Use model.type=lstm or model.type=transformer on CPU."
            )

        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "fair-esm is not installed. Run: pip install fair-esm"
            ) from exc

        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze all ESM-2 parameters first
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        n_trainable: int = cfg.model.esm_trainable_layers
        total_layers = len(self.esm_model.layers)
        for layer in self.esm_model.layers[total_layers - n_trainable :]:
            for param in layer.parameters():
                param.requires_grad = True

        # Task-specific MLP heads
        self.class_head = nn.Sequential(
            nn.Linear(self.ESM_EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 3),
        )
        self.hydro_head = nn.Sequential(
            nn.Linear(self.ESM_EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, sequences: list[str]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Accept list of raw AA strings; return prediction dict."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_reps = results["representations"][6]  # [B, L+2, 320]

        # Mean-pool over non-padding residue positions (skip BOS/EOS tokens)
        pooled = token_reps[:, 1:-1, :].mean(dim=1)  # [B, 320]

        class_logits = self.class_head(pooled)  # [B, 3]
        hydro_pred = self.hydro_head(pooled).squeeze(-1)  # [B]

        return {"class_logits": class_logits, "hydro_pred": hydro_pred}
