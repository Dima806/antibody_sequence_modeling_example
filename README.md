# AntibodySeqML

End-to-end deep learning pipeline for predicting biophysical properties of antibody CDR-H3 sequences.

**Two prediction tasks (multi-task):**
- CDR-H3 length classification: `short` / `medium` / `long` (3-class)
- Hydrophobicity regression: Kyte-Doolittle GRAVY score (scalar)

**W&B project:** [dima806/antibody-seq-ml](https://wandb.ai/dima806/antibody-seq-ml)

---

## Dual-Environment Strategy

| Environment | Purpose | Dataset | Hardware |
|---|---|---|---|
| GitHub Codespaces | Dev, tests, EDA, smoke runs | ~2K sequences (`data/smoke/`) | 2-core CPU |
| Kaggle Notebook | Full training, sweeps, ESM-2 | ~500K OAS sequences | T4 / P100 GPU |

---

## Repository Structure

```
antibody_sequence_modeling_example/
├── .devcontainer/
│   └── devcontainer.json              # Codespaces: Python 3.11, uv
├── .pre-commit-config.yaml            # ruff, trailing-whitespace, yaml/toml checks, etc.
├── configs/
│   ├── default.yaml                   # Smoke-scale defaults
│   ├── schema.yaml                    # Dataset schema (validated on every generate/download)
│   ├── sweep_cpu.yaml                 # Random sweep, CPU
│   └── sweep_gpu.yaml                 # Bayesian sweep, GPU
├── data/
│   ├── smoke/
│   │   └── sequences_smoke.csv        # ~2K synthetic sequences (committed)
│   ├── download.py                    # Download OAS data from Zenodo / OAS API
│   ├── generate_smoke.py              # Generate synthetic smoke dataset
│   └── validate.py                    # Schema validation (reads configs/schema.yaml)
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA on smoke data (Codespaces)
│   └── 02_kaggle_training.ipynb       # Full GPU training (Kaggle)
├── src/
│   └── antibody_seq_ml/
│       ├── dataset.py                 # Tokeniser, CDRDataset, DataLoaders
│       ├── train.py                   # Training loop + W&B logging
│       ├── evaluate.py                # Metrics, confusion matrix, attention viz
│       ├── sweep.py                   # W&B sweep agent entry point
│       └── models/
│           ├── base.py                # SequenceModel base class + factory
│           ├── lstm.py                # BiLSTM with mean pooling
│           ├── transformer.py         # TransformerEncoder + sinusoidal PE
│           └── esm2.py                # ESM-2 fine-tuning (GPU only)
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_train.py
├── Makefile
├── pyproject.toml
└── requirements_kaggle.txt
```

---

## Quick Start (Codespaces / CPU)

### 1. Setup

```bash
make setup
```

Installs `uv`, the project package in editable mode, and pre-commit hooks.

### 2. Generate smoke dataset

```bash
make data-smoke
```

Regenerates `data/smoke/sequences_smoke.csv` (~2K synthetic sequences, seed=42) and validates the output against `configs/schema.yaml`. The file is already committed — only run this to regenerate.

### 3. Smoke training run

```bash
make train-smoke
```

Trains a small Transformer on the smoke dataset for 10 epochs with W&B disabled. Checkpoint saved to `checkpoints/best_model.pt`.

### 4. Run tests

```bash
make test
```

Runs pytest with coverage across `tests/`. All tests use smoke data and run on CPU.

### 5. Lint

```bash
make lint
```

Runs `pre-commit run --all-files` — ruff (lint + format), trailing whitespace, YAML/TOML checks, merge conflict detection, large file guard, and more.

### 6. CPU sweep

```bash
make sweep-cpu
```

Launches a W&B random sweep (5 runs) over `lstm` and `transformer` using smoke data.

---

## Downloading the Full OAS Dataset (Kaggle / GPU)

Two sources are supported, both require no authentication:

| Source | Command | Notes |
|---|---|---|
| Zenodo | `make data-full` | p-IgGen pre-processed OAS snapshot (default) |
| OAS REST API | `python data/download.py --source oas-api` | Direct from opig.stats.ox.ac.uk, up to `--max-units` units |

```bash
# Zenodo (default) — single file, no auth
make data-full

# OAS REST API — downloads up to 200 data units
python data/download.py --output data/full/ --source oas-api
```

Both sources produce `data/full/sequences_full.csv` with identical schema (validated automatically):
`sequence_id`, `heavy_chain_sequence`, `cdr_h3`, `cdr_h3_length`, `length_class`, `hydrophobicity`.

When no CDR-H3 column is present in the source data, it is extracted from the heavy chain sequence using the conserved `C...WGxG` flanking motif.

For Codespaces smoke testing, the committed dataset at `data/smoke/sequences_smoke.csv` is used instead.

---

## Dataset Schema Validation

The schema is defined in `configs/schema.yaml` and enforced by `data/validate.py` after every `make data-smoke` and `make data-full` run. It checks:

- All 6 required columns are present and non-null
- String length bounds (`cdr_h3`: 5–30 AA, `heavy_chain_sequence`: ≥30 AA)
- `length_class` ∈ {`short`, `medium`, `long`}
- `hydrophobicity` ∈ [-5.0, 5.0]
- `cdr_h3_length` matches `len(cdr_h3)` exactly
- Length class boundaries are consistent with `cdr_h3_length`

---

## Models

| Model | Architecture | Smoke params | Full params |
|---|---|---|---|
| `lstm` | Embedding → BiLSTM → mean pool → heads | ~200K | ~3M |
| `transformer` | [CLS] + Embedding + sinusoidal PE → TransformerEncoder → heads | ~150K | ~4M |
| `esm2` | ESM-2 (8M) frozen + 2 trainable layers + MLP heads | ~1M trainable | GPU only |

Select model via `configs/default.yaml`:
```yaml
model:
  type: transformer   # lstm | transformer | esm2
```

---

## Training

### Losses

Multi-task: `loss = cls_loss_weight × CrossEntropy + reg_loss_weight × MSE`

Default weights: `cls=1.0`, `reg=0.1` (MSE is orders of magnitude smaller than CE).

### Scheduler & early stopping

`CosineAnnealingLR` or `ReduceLROnPlateau` (configurable). Early stopping on `val/loss` with configurable `patience`.

### W&B authentication

`WANDB_API_KEY` is read from the environment (injected automatically from Codespaces / Kaggle secrets). `train.py` and `sweep.py` call `wandb.login()` at startup when the key is present.

### W&B logging

Every epoch logs:
```
train/loss  train/cls_loss  train/reg_loss  train/cls_acc  train/hydro_r2
val/loss    val/cls_loss    val/reg_loss    val/cls_acc    val/hydro_r2
lr  grad_norm  epoch
```

---

## Makefile Reference

| Target | Description |
|---|---|
| `make setup` | Install uv, package, pre-commit hooks |
| `make data-smoke` | Generate ~2K synthetic smoke sequences |
| `make data-full` | Download full OAS dataset from Zenodo |
| `make train-smoke` | Smoke training run, W&B disabled |
| `make train-full` | Full training run on `data/full/sequences_full.csv` |
| `make sweep-cpu` | Random sweep (5 runs) on smoke data |
| `make sweep-gpu` | Bayesian sweep (50 runs) on full data |
| `make test` | pytest + coverage |
| `make lint` | pre-commit (ruff, format, hooks) |
| `make notebook` | Open EDA notebook |

---

## Configuration

All hyperparameters live in `configs/default.yaml`. The `--smoke` flag overrides to small-scale settings:

| Hyperparameter | Smoke (CPU) | Full (GPU) |
|---|---|---|
| `d_model` / `embedding_dim` | 64 | 256 |
| `hidden_dim` (LSTM) | 128 | 512 |
| `num_layers` | 2 | 3–6 |
| `dim_feedforward` | 256 | 1024 |
| `nhead` | 4 | 8 |
| `batch_size` | 32–64 | 128–512 |
| `epochs` | 10 | 30–50 |
| `max_seq_length` | 30 | 150 |

---

## Kaggle Full Training

Open `notebooks/02_kaggle_training.ipynb` on Kaggle with GPU enabled.

The notebook self-bootstraps:
1. Clones this repo from GitHub
2. Installs all dependencies (`pip install -e .` + `fair-esm`)
3. Logs in to W&B and HuggingFace via Kaggle Secrets (`WANDB_API_KEY`, `HF_API_KEY`)
4. Downloads full OAS data from Zenodo (fallback: OAS API)
5. Runs full-scale Transformer training
6. Launches 50-run Bayesian W&B sweep
7. Fine-tunes ESM-2 (stretch goal)
8. Evaluates best model and logs to W&B Model Registry

**Prerequisites:**
- Kaggle Settings → Accelerator → GPU T4 x2 or P100
- Kaggle Settings → Internet → On
- Kaggle Secrets → `WANDB_API_KEY` and `HF_API_KEY` set

---

## Success Targets

| Metric | Target |
|---|---|
| CDR-H3 length classification accuracy | >85% (full GPU data) |
| Hydrophobicity R² | >0.75 (full GPU data) |
| ESM-2 fine-tuned accuracy | >90% (stretch) |
| CPU smoke train time | <3 min |
| Test coverage | >80% |

---

## Development Conventions

- **Device:** never hardcode `"cpu"` / `"cuda"` — always branch on `torch.cuda.is_available()`
- **Config:** `omegaconf.DictConfig` throughout; load with `OmegaConf.load("configs/default.yaml")`
- **W&B in tests:** `WANDB_MODE=disabled` set by `conftest.py` and `make test`
- **Schema:** both `data/generate_smoke.py` and `data/download.py` call `data/validate.py` before writing CSV
- **Type hints:** all functions use full type hints with `from __future__ import annotations`
- **Imports:** stdlib → third-party → local, enforced by ruff
