"""PyTorch Dataset, tokeniser, and DataLoader utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# 20 standard amino acids
_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Vocabulary: PAD=0, then 20 AAs (indices 1-20), UNK=21
VOCAB: list[str] = ["<PAD>"] + _AA_LIST + ["<UNK>"]
VOCAB_SIZE: int = len(VOCAB)  # 22

LENGTH_CLASS_MAP: dict[str, int] = {"short": 0, "medium": 1, "long": 2}
CLASS_NAMES: list[str] = ["short", "medium", "long"]


class AminoAcidTokeniser:
    """Maps amino acid characters to integer indices and back."""

    VOCAB = VOCAB
    VOCAB_SIZE = VOCAB_SIZE
    PAD_IDX: int = 0
    UNK_IDX: int = 21

    def __init__(self) -> None:
        self._token_to_idx: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.VOCAB)
        }
        self._idx_to_token: dict[int, str] = {
            idx: tok for idx, tok in enumerate(self.VOCAB)
        }

    def encode(self, sequence: str, max_len: int) -> torch.Tensor:
        """Tokenise, truncate to max_len, then right-pad with PAD_IDX."""
        indices = [self._token_to_idx.get(aa.upper(), self.UNK_IDX) for aa in sequence]
        indices = indices[:max_len]
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """Convert index tensor back to amino acid string, skipping PAD."""
        return "".join(
            self._idx_to_token.get(int(idx), "<UNK>")
            for idx in indices
            if int(idx) != self.PAD_IDX
        )


class CDRDataset(Dataset):
    """PyTorch Dataset for CDR-H3 sequences with two prediction targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokeniser: AminoAcidTokeniser,
        max_len: int,
    ) -> None:
        self.tokeniser = tokeniser
        self.max_len = max_len
        self.sequences: list[str] = df["cdr_h3"].tolist()
        self.length_labels: list[int] = [
            LENGTH_CLASS_MAP[cls] for cls in df["length_class"].tolist()
        ]
        self.hydro_scores: list[float] = df["hydrophobicity"].tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.tokeniser.encode(self.sequences[idx], self.max_len),
            "length_label": torch.tensor(self.length_labels[idx], dtype=torch.long),
            "hydro_score": torch.tensor(self.hydro_scores[idx], dtype=torch.float32),
        }


def build_dataloaders(
    cfg: DictConfig,
    data_path: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CSV, split into train/val/test, return DataLoaders."""
    df = pd.read_csv(data_path)
    tokeniser = AminoAcidTokeniser()
    max_len: int = cfg.data.max_seq_length

    # Stratified 70 / 15 / 15 split
    train_df, temp_df = train_test_split(
        df,
        test_size=round(1.0 - cfg.data.train_split, 10),
        random_state=42,
        stratify=df["length_class"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["length_class"],
    )

    train_ds = CDRDataset(train_df.reset_index(drop=True), tokeniser, max_len)
    val_ds = CDRDataset(val_df.reset_index(drop=True), tokeniser, max_len)
    test_ds = CDRDataset(test_df.reset_index(drop=True), tokeniser, max_len)

    batch_size: int = cfg.data.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_full_dataset(
    cache_path: str,
    batch_size: int,
    device: str = "cuda",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function for Kaggle notebook: load full OAS dataset."""
    cfg = OmegaConf.create(
        {
            "data": {
                "max_seq_length": 150,
                "train_split": 0.70,
                "val_split": 0.15,
                "batch_size": batch_size,
                "cache_dir": cache_path,
            }
        }
    )
    data_path = str(Path(cache_path) / "sequences_full.csv")
    return build_dataloaders(cfg, data_path)
