# CLAUDE.md — AntibodySeqML Implementation Guide

## Project Overview

**AntibodySeqML** is an end-to-end deep learning pipeline for predicting biophysical properties of antibody CDR-H3 sequences. It is a portfolio project targeting Roche Prescient Design / AI4DD ML Scientist role.

**W&B project:** `antibody-seq-ml` | **W&B entity:** `dima806-team`

**Two prediction tasks (multi-task head or separate runs):**
- CDR-H3 length classification: short / medium / long (3-class)
- Hydrophobicity regression: Kyte-Doolittle score (scalar)

---

## Dual-Environment Strategy

| Environment | Purpose | Dataset | Torch build |
|---|---|---|---|
| GitHub Codespaces (2-core CPU) | Dev, tests, EDA, smoke runs | ~2K sequences (`data/smoke/`) | `torch==2.3.0` CPU-only |
| Kaggle Notebooks (T4/P100 GPU) | Full training, sweeps, ESM-2 | ~500K sequences (OAS) | Pre-installed CUDA torch |

**Detection at runtime:** `torch.cuda.is_available()` — all code must branch on this, never hardcode `"cpu"` or `"cuda"`.

---

## Repository Structure

```
antibody_sequence_modeling_example/
├── .devcontainer/
│   └── devcontainer.json
├── .pre-commit-config.yaml
├── configs/
│   ├── default.yaml
│   ├── schema.yaml                    # Dataset schema — validated on every data write
│   ├── sweep_cpu.yaml
│   └── sweep_gpu.yaml
├── data/
│   ├── smoke/
│   │   └── sequences_smoke.csv
│   ├── download.py                    # Zenodo / OAS API download
│   ├── generate_smoke.py
│   └── validate.py                    # Schema validator (reads configs/schema.yaml)
├── src/
│   └── antibody_seq_ml/
│       ├── __init__.py
│       ├── dataset.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── lstm.py
│       │   ├── transformer.py
│       │   └── esm2.py
│       ├── train.py
│       ├── evaluate.py
│       └── sweep.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_kaggle_training.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_train.py
├── CLAUDE.md
├── Makefile
├── pyproject.toml
└── requirements_kaggle.txt
```

---

## Implementation Phases

### Phase 1 — Scaffold

**Files to create:**

1. **`pyproject.toml`** — uv-managed, CPU torch by default:
   ```toml
   [project]
   name = "antibody-seq-ml"
   version = "0.1.0"
   requires-python = ">=3.11"
   dependencies = [
       "torch==2.3.0",
       "wandb>=0.17",
       "biopython>=1.83",
       "omegaconf>=2.3",
       "pandas>=2.2",
       "numpy>=1.26",
       "scikit-learn>=1.4",
       "matplotlib>=3.8",
       "seaborn>=0.13",
       "jupyter>=1.0",
       "pytest>=8.0",
       "pytest-cov>=5.0",
       "ruff>=0.4",
       "black>=24.0",
       "mypy>=1.9",
       "pre-commit>=3.7",
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/antibody_seq_ml"]

   [tool.ruff]
   line-length = 88
   target-version = "py311"

   [tool.mypy]
   python_version = "3.11"
   ignore_missing_imports = true
   ```

2. **`requirements_kaggle.txt`** — GPU Kaggle environment:
   ```
   wandb>=0.17
   biopython>=1.83
   omegaconf>=2.3
   fair-esm>=2.0.0
   git+https://github.com/dima806/antibody-seq-ml.git
   ```

3. **`.devcontainer/devcontainer.json`**:
   ```json
   {
     "name": "AntibodySeqML",
     "image": "mcr.microsoft.com/devcontainers/python:3.11",
     "postCreateCommand": "pip install uv && uv pip install -e '.[dev]' --system",
     "features": {
       "ghcr.io/devcontainers/features/github-cli:1": {}
     }
   }
   ```

4. **`Makefile`** (grouped by section):
   ```makefile
   # Environment
   setup:
       pip install uv && uv pip install -e . --system
       uv tool install prek && prek install

   # Data
   data-smoke:
       python data/generate_smoke.py --output data/smoke/sequences_smoke.csv
   data-full:
       python data/download.py --output data/full/ --source zenodo

   # Training
   train-smoke:
       WANDB_MODE=disabled python -m antibody_seq_ml.train \
           --config configs/default.yaml \
           --data data/smoke/sequences_smoke.csv \
           --smoke
   train-full:
       python -m antibody_seq_ml.train \
           --config configs/default.yaml \
           --data data/full/sequences_full.csv
   sweep-cpu:
       python -m antibody_seq_ml.sweep \
           --config configs/sweep_cpu.yaml \
           --data data/smoke/sequences_smoke.csv
   sweep-gpu:
       python -m antibody_seq_ml.sweep \
           --config configs/sweep_gpu.yaml \
           --data data/full/sequences_full.csv

   # Quality
   test:
       WANDB_MODE=disabled pytest tests/ -v --cov=src/antibody_seq_ml --cov-report=term-missing
   lint:
       pre-commit run --all-files

   # Notebooks
   notebook:
       jupyter notebook notebooks/01_eda.ipynb
   ```

5. **`.pre-commit-config.yaml`**:
   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.6.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-toml
         - id: check-json
         - id: check-merge-conflict
         - id: check-added-large-files
           args: [--maxkb=500]
         - id: debug-statements
         - id: mixed-line-ending
           args: [--fix=lf]
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.4.4
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
   ```

### Phase 2 — Data Layer

#### `data/smoke/sequences_smoke.csv`

Generate a synthetic stratified smoke dataset with ~2,000 rows. Columns:
- `sequence_id`: string identifier
- `heavy_chain_sequence`: full heavy chain AA sequence (string, 20 AA alphabet)
- `cdr_h3`: extracted CDR-H3 subsequence (5–25 AA)
- `cdr_h3_length`: integer length of CDR-H3
- `length_class`: `short` (≤9), `medium` (10–14), `long` (≥15)
- `hydrophobicity`: float, Kyte-Doolittle score computed via BioPython

Stratify 70/15/15 train/val/test across `length_class`. Commit this file — it is the only data committed to the repo.

**Script to generate smoke data: `data/generate_smoke.py`** — uses BioPython `ProtParam` and random AA sequences seeded for reproducibility.

#### `data/download.py`

Downloads full OAS paired dataset. Supports two sources via `--source`:
- `zenodo` (default) — p-IgGen pre-processed snapshot from Zenodo record 13880874, no auth
- `oas-api` — OAS REST API at opig.stats.ox.ac.uk, no auth, `--max-units` controls volume

When the source data has no CDR-H3 column, it is extracted from the heavy chain sequence using the conserved `C...WGxG` flanking motif regex. Calls `data/validate.py` before writing the CSV.

#### `data/validate.py`

Reads `configs/schema.yaml` and validates a DataFrame against it. Checks column presence, nullability, dtype ranges, enum values, and derived invariants (`cdr_h3_length == len(cdr_h3)`, length class boundaries). Raises `ValueError` on any violation. Called by both `generate_smoke.py` and `download.py`.

#### `configs/schema.yaml`

Single source of truth for the dataset schema. Defines dtype, nullable, min/max (numeric), min/max_length (string), and allowed values for each of the 6 columns.

#### `src/antibody_seq_ml/dataset.py`

Key design decisions:
- **Vocabulary:** 22 tokens — 20 standard AAs + `<PAD>` (index 0) + `<UNK>` (index 21)
- **Max sequence length:** configurable via config, default 30 for smoke, 150 for full
- **Tokeniser:** `AminoAcidTokeniser` class — maps each AA char to integer index
- **Dataset class:** `CDRDataset(torch.utils.data.Dataset)` — reads CSV, tokenises CDR-H3, pads/truncates to `max_len`, returns `(token_tensor, length_class_label, hydrophobicity_score)`
- **DataLoaders:** `build_dataloaders(cfg, data_path, device)` — returns `(train_loader, val_loader, test_loader)` with stratified split
- **Caching:** after first tokenisation, serialize tensors to `data/cache/*.pt` and load from cache on subsequent runs
- **Device agnostic:** tensors created on CPU in Dataset; move to device in training loop, not in Dataset

```python
# Key interfaces to implement:

class AminoAcidTokeniser:
    VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["<PAD>", "<UNK>"]
    PAD_IDX = 0

    def encode(self, sequence: str, max_len: int) -> torch.Tensor: ...
    def decode(self, indices: torch.Tensor) -> str: ...

class CDRDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokeniser: AminoAcidTokeniser, max_len: int): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...
    # Returns: {"input_ids": Tensor[max_len], "length_label": Tensor[], "hydro_score": Tensor[]}

def build_dataloaders(cfg: DictConfig, data_path: str) -> tuple[DataLoader, DataLoader, DataLoader]: ...
def build_full_dataset(cache_path: str, batch_size: int, device: str) -> tuple[DataLoader, DataLoader, DataLoader]: ...
```

#### `notebooks/01_eda.ipynb`

EDA notebook to run on Codespaces CPU with smoke data. Cells should cover:
1. Load smoke CSV, display shape, dtypes, null checks
2. Distribution of `length_class` (bar chart)
3. Distribution of CDR-H3 lengths (histogram)
4. Distribution of hydrophobicity scores (histogram + KDE)
5. Example sequences per class
6. Amino acid frequency heatmap (position × AA)
7. Correlation between CDR-H3 length and hydrophobicity

### Phase 3 — Models

#### `src/antibody_seq_ml/models/base.py`

```python
class SequenceModel(nn.Module):
    """Base class for all sequence models."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # Must return {"class_logits": Tensor[B, 3], "hydro_pred": Tensor[B]}
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

```python
# Factory function:
def build_model(cfg: DictConfig, device: torch.device) -> SequenceModel:
    model_type = cfg.model.type  # "lstm" | "transformer" | "esm2"
    # Select class, instantiate, move to device, return
```

#### `src/antibody_seq_ml/models/lstm.py`

```python
class BiLSTMModel(SequenceModel):
    """
    Embedding → BiLSTM → mean pooling → two linear heads
    Smoke variant: embedding_dim=64, hidden=128, layers=2
    Full variant:  embedding_dim=256, hidden=512, layers=3
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        # cfg.model.embedding_dim, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout
        self.embedding = nn.Embedding(22, cfg.model.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=cfg.model.embedding_dim,
            hidden_size=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.class_head = nn.Linear(cfg.model.hidden_dim * 2, 3)
        self.hydro_head = nn.Linear(cfg.model.hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # input_ids: [B, max_len]
        # embed → lstm → mean pool over time dim → heads
        # Return {"class_logits": [B, 3], "hydro_pred": [B]}
```

#### `src/antibody_seq_ml/models/transformer.py`

```python
class TransformerModel(SequenceModel):
    """
    [CLS] + Embedding + sinusoidal PE → TransformerEncoder → CLS pooling → heads
    Smoke: d_model=64, nhead=4, layers=3, dim_ff=256
    Full:  d_model=256, nhead=8, layers=6, dim_ff=1024
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        # Prepend CLS token (index 1, reserved — shift vocab by 1 or use separate param)
        # Use nn.TransformerEncoder with nn.TransformerEncoderLayer
        # Sinusoidal positional encoding: implement as fixed (not learnable) buffer
        self.embedding = nn.Embedding(22, cfg.model.d_model, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.model.d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(cfg.model.d_model, max_len=152)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_layers)
        self.class_head = nn.Linear(cfg.model.d_model, 3)
        self.hydro_head = nn.Linear(cfg.model.d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # Embed, prepend CLS, add positional encoding
        # Create padding mask from input_ids (where == 0)
        # Encode → extract CLS output → heads
```

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal PE. Register as buffer (not parameter)."""
```

#### `src/antibody_seq_ml/models/esm2.py`

GPU-only model. Guard with `if not torch.cuda.is_available(): raise RuntimeError(...)`.

```python
class ESM2FineTuned(SequenceModel):
    """
    ESM-2 (esm2_t6_8M_UR50D, 8M params) as frozen feature extractor.
    Only last 2 transformer layers + task head are trainable.
    Input: raw AA string sequences (not pre-tokenised)
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        import esm
        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        # Freeze all layers except last cfg.model.esm_trainable_layers
        # MLP head: Linear(320, 128) → ReLU → Linear(128, 3) for class
        #           Linear(320, 128) → ReLU → Linear(128, 1) for hydro

    def forward(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        # Use batch_converter, run ESM, mean-pool token representations, apply heads
```

### Phase 4 — Training Loop & W&B

#### `src/antibody_seq_ml/train.py`

Key design decisions:
- Single `train()` function accepts `(model, train_loader, val_loader, cfg, wandb_run=None)`
- Supports both classification and regression tasks simultaneously (multi-task)
- **Loss:** `CrossEntropyLoss` for classification + `MSELoss` for regression, summed with configurable weights `cfg.training.cls_loss_weight` and `cfg.training.reg_loss_weight`
- **Optimizer:** `AdamW` with `weight_decay` from config
- **Scheduler:** `CosineAnnealingLR` or `ReduceLROnPlateau` (configurable)
- **Early stopping:** track `val/loss`, stop if no improvement for `cfg.training.patience` epochs, save best checkpoint
- **Checkpointing:** save `best_model.pt` to `cfg.training.checkpoint_dir`
- **W&B logging per epoch:**
  ```python
  wandb.log({
      "train/loss": ..., "train/cls_loss": ..., "train/reg_loss": ...,
      "train/cls_acc": ..., "train/hydro_r2": ...,
      "val/loss": ..., "val/cls_loss": ..., "val/reg_loss": ...,
      "val/cls_acc": ..., "val/hydro_r2": ...,
      "lr": scheduler.get_last_lr()[0],
      "grad_norm": compute_grad_norm(model),
      "epoch": epoch,
  })
  ```
- **Hardware tag:** log `wandb.config.update({"hardware": "gpu:T4" if cuda else "cpu"})`
- **W&B authentication:** reads `WANDB_API_KEY` from environment and calls `wandb.login()` before `wandb.init()`. Injected automatically from Codespaces / Kaggle secrets.
- **CLI entry point:** `python -m antibody_seq_ml.train --config configs/default.yaml --data path/to/data.csv [--smoke]`
  - `--smoke` flag overrides config to use smoke-scale hyperparameters (small model, 10 epochs)

```python
def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def train(
    model: SequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    wandb_run=None,
) -> dict[str, float]:
    """Returns final test metrics dict."""
```

#### `src/antibody_seq_ml/evaluate.py`

```python
def evaluate(
    model: SequenceModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Returns {"cls_acc", "cls_f1", "hydro_mse", "hydro_r2", "loss"}."""

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None): ...
def plot_attention_weights(attn_weights, sequence, save_path=None): ...
    # For TransformerModel: hook into encoder layers to capture attention
    # Visualise as heatmap: position × position, one plot per head
```

#### `src/antibody_seq_ml/sweep.py`

```python
def sweep_train_fn():
    """W&B sweep agent entry point. Reads hyperparams from wandb.config."""
    with wandb.init() as run:
        cfg = build_cfg_from_wandb(wandb.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, test_loader = build_dataloaders(cfg, data_path)
        model = build_model(cfg, device)
        train(model, train_loader, val_loader, cfg, wandb_run=run)

# CLI: python -m antibody_seq_ml.sweep --config configs/sweep_cpu.yaml --data path/to/data.csv
```

#### `configs/default.yaml`

```yaml
data:
  max_seq_length: 30
  train_split: 0.70
  val_split: 0.15
  batch_size: 64
  cache_dir: data/cache

model:
  type: transformer        # lstm | transformer | esm2
  embedding_dim: 64
  d_model: 64
  nhead: 4
  hidden_dim: 128
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  esm_trainable_layers: 2

training:
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 5
  checkpoint_dir: checkpoints
  cls_loss_weight: 1.0
  reg_loss_weight: 0.1
  scheduler: cosine        # cosine | plateau

wandb:
  project: antibody-seq-ml
  entity: dima806-team
  log_gradients: true
  mode: online             # online | offline | disabled
```

#### `configs/sweep_cpu.yaml`

```yaml
method: random
metric:
  name: val/loss
  goal: minimize
parameters:
  model_type:    {values: [lstm, transformer]}
  learning_rate: {values: [0.001, 0.01]}
  dropout:       {values: [0.1, 0.3]}
  batch_size:    {values: [32, 64]}
count: 5
```

#### `configs/sweep_gpu.yaml`

```yaml
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  model_type:       {values: [lstm, transformer, esm2]}
  learning_rate:    {min: 0.00001, max: 0.01, distribution: log_uniform_values}
  embedding_dim:    {values: [64, 128, 256]}
  dropout:          {min: 0.0, max: 0.4}
  batch_size:       {values: [128, 256, 512]}
  num_layers:       {values: [2, 3, 6]}
  weight_decay:     {min: 0.00001, max: 0.001, distribution: log_uniform_values}
count: 50
```

### Phase 5 — Kaggle Notebook

#### `notebooks/02_kaggle_training.ipynb`

Self-contained Kaggle notebook. Six cell groups:

1. **Setup:** install deps from `requirements_kaggle.txt` via pip; login to W&B via Kaggle Secrets (`WANDB_API_KEY`).
2. **Data:** download full OAS dataset, preprocess, cache as Kaggle Dataset artifact.
3. **Single GPU training run:** load full-scale config, build transformer/LSTM, run `train()`.
4. **W&B Bayesian sweep:** `wandb.sweep()` → `wandb.agent()` with 50 runs.
5. **ESM-2 fine-tuning:** load `ESM2FineTuned`, fine-tune on CDR sequences, log to W&B.
6. **Final evaluation:** load best checkpoint, evaluate on test set, log metrics to W&B Model Registry, save checkpoint to Kaggle output.

### Phase 6 — Results & Polish

- Fill in README with dual-environment diagram, quick-start badge, results table
- Run attention visualisation on best Transformer model
- Log final model artifacts to W&B Model Registry with metadata

---

## Testing Requirements

All tests must pass on Codespaces CPU using only smoke data. No GPU required.

#### `tests/conftest.py`

```python
import pytest
import pandas as pd
import torch
from antibody_seq_ml.dataset import AminoAcidTokeniser, CDRDataset

SMOKE_DATA_PATH = "data/smoke/sequences_smoke.csv"

@pytest.fixture
def tokeniser():
    return AminoAcidTokeniser()

@pytest.fixture
def smoke_df():
    return pd.read_csv(SMOKE_DATA_PATH)

@pytest.fixture
def smoke_dataset(smoke_df, tokeniser):
    return CDRDataset(smoke_df, tokeniser, max_len=30)
```

#### `tests/test_dataset.py`

- `test_tokeniser_known_sequence()` — encode `"ACDE"`, verify indices match vocab
- `test_tokeniser_unknown_aa()` — encode sequence with `"X"`, verify maps to UNK index
- `test_padding_to_max_length()` — encode short sequence, verify output length == max_len
- `test_truncation_to_max_length()` — encode long sequence, verify output length == max_len
- `test_dataloader_batch_shapes()` — iterate one batch, verify shapes: `input_ids [B, 30]`, `length_label [B]`, `hydro_score [B]`
- `test_stratified_split_proportions()` — verify train/val/test sizes are approximately 70/15/15

#### `tests/test_models.py`

- `test_lstm_forward_pass()` — create smoke BiLSTM, pass random `[4, 30]` tensor, verify output dict has correct keys and shapes
- `test_transformer_forward_pass()` — same for TransformerModel
- `test_output_class_logits_shape()` — verify `class_logits` is `[B, 3]`
- `test_output_hydro_pred_shape()` — verify `hydro_pred` is `[B]`
- `test_padding_mask_applied()` — verify model doesn't crash with all-padding sequences
- `test_model_factory()` — `build_model(cfg)` returns correct class for each `model.type`

#### `tests/test_train.py`

- `test_single_epoch_runs_without_error()` — smoke data + smoke model, run 1 epoch, no exception
- `test_loss_decreases()` — run 3 epochs, verify final loss < initial loss
- `test_early_stopping_triggers()` — mock val loss not improving, verify training stops at `patience`
- `test_checkpoint_saved()` — run training, verify `best_model.pt` exists after completion
- `test_wandb_disabled_mode()` — set `cfg.wandb.mode = "disabled"`, verify training runs cleanly

---

## Code Conventions

### Type Hints
All functions must have full type hints. Use `from __future__ import annotations` at top of each file.

### Imports
```python
# Standard order:
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
```

### Device Handling
Never use string literals `"cpu"` or `"cuda"` except in the single device-selection line:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Then pass `device` as a parameter everywhere.

### Config Pattern
Use `omegaconf.DictConfig` for all config objects. Load with `OmegaConf.load("configs/default.yaml")`. Override individual keys when `--smoke` flag is set:
```python
if smoke:
    cfg.model.embedding_dim = 64
    cfg.model.num_layers = 2
    cfg.training.epochs = 10
    cfg.data.batch_size = 32
```

### W&B Integration
Always check `if wandb_run is not None:` before calling `wandb.log()`. This allows tests to run without W&B. In `sweep.py`, W&B is always active. In `train.py`, it is optional.

### Error Handling
Only validate at boundaries (CLI args, CSV loading). Trust internal function contracts. Do not add try/except around PyTorch operations.

---

## Smoke vs Full Scale Hyperparameter Reference

| Hyperparameter | Smoke (CPU) | Full (GPU) |
|---|---|---|
| `embedding_dim` / `d_model` | 64 | 256 |
| `hidden_dim` (LSTM) | 128 | 512 |
| `num_layers` | 2 | 3–6 |
| `dim_feedforward` | 256 | 1024 |
| `nhead` | 4 | 8 |
| `batch_size` | 32–64 | 128–512 |
| `epochs` | 10 | 30–50 |
| `max_seq_length` | 30 | 150 |
| Dataset size | ~2K | ~500K |
| Expected train time | <3 min | ~6 min/run |

---

## CDR-H3 Extraction Logic

CDR-H3 per IMGT numbering rules: residues 105–117 in the heavy chain. In OAS paired data, CDR-H3 is provided as an annotated column. For raw sequences, use BioPython with IMGT scheme:

```python
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def compute_hydrophobicity(sequence: str) -> float:
    """Kyte-Doolittle scale, window=None (whole sequence mean)."""
    analysis = ProteinAnalysis(sequence.upper())
    return analysis.gravy()  # GRAVY = Grand Average of hYdropathicity

def classify_length(length: int) -> str:
    if length <= 9: return "short"
    if length <= 14: return "medium"
    return "long"
```

---

## W&B Artifact Schema

| Artifact | Type | Logged from | Description |
|---|---|---|---|
| `oas-smoke-dataset` | `dataset` | Codespaces | 2K smoke CSV |
| `oas-full-dataset` | `dataset` | Kaggle | 500K Parquet |
| `tokenised-smoke` | `dataset` | Codespaces | Pre-tokenised `.pt` tensors |
| `tokenised-full` | `dataset` | Kaggle | Pre-tokenised tensors |
| `bilstm-best` | `model` | Kaggle | Best BiLSTM checkpoint |
| `transformer-best` | `model` | Kaggle | Best Transformer checkpoint |
| `esm2-finetuned` | `model` | Kaggle | ESM-2 fine-tuned weights |

---

## Success Criteria (from PRD)

| Metric | Target |
|---|---|
| CDR-H3 length classification accuracy | >85% (full GPU data) |
| Hydrophobicity regression R² | >0.75 (full GPU data) |
| ESM-2 fine-tuned accuracy | >90% (stretch) |
| CPU smoke train time | <3 min |
| Test coverage | >80% |
| CI passing on every push | Required |
| W&B dashboard public | Required |
| Kaggle notebook public | Required |

---

## Implementation Notes & Gotchas

1. **OAS data format:** OAS paired data CSVs have a metadata header row — skip it with `pd.read_csv(..., header=1)`. Column names vary by OAS study; normalise to `sequence_id`, `heavy_sequence`, `cdr_h3` during download preprocessing.

2. **BiLSTM mean pooling:** Mask out padding positions before pooling — do not include PAD token representations in the mean. Use the actual sequence lengths from `input_ids != 0`.

3. **Transformer padding mask:** `nn.TransformerEncoder` expects `src_key_padding_mask` of shape `[B, S]` with `True` for positions to **ignore**. Generate as `input_ids == 0` (padding index).

4. **CLS token index:** Reserve index 1 in the vocabulary for `<CLS>`. Do not use 0 (that is `<PAD>`). Prepend to embedded sequence before positional encoding.

5. **Multi-task loss balancing:** Regression loss (MSE on hydrophobicity) will be orders of magnitude smaller than classification loss (CrossEntropy). Use `cfg.training.reg_loss_weight` to scale appropriately. Start with `cls=1.0, reg=0.1`.

6. **ESM-2 import guard:** Wrap `import esm` inside the `__init__` method, not at module level. This prevents `ImportError` on Codespaces where `fair-esm` is not installed.

7. **Gradient clipping:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` in training loop to stabilise LSTM training.

8. **Smoke data generation:** Use `random.seed(42)` and `numpy.random.seed(42)` for reproducibility. The generated CSV must be committed so tests are deterministic.

9. **`wandb.init()` in tests:** Always set `WANDB_MODE=disabled` in test environment or pass `mode="disabled"` to `wandb.init()`. Add this to `conftest.py` via `monkeypatch.setenv`.

10. **Kaggle Secrets:** `WANDB_API_KEY` is accessed via `kaggle_secrets.UserSecretsClient().get_secret("WANDB_API_KEY")`. This is Kaggle-specific; document this clearly in the notebook.
