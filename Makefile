.PHONY: setup \
        data-smoke data-full \
        train-smoke train-full sweep-cpu sweep-gpu \
        test lint \
        notebook

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

setup:
	pip install uv && uv pip install -e . --system
	uv tool install prek && prek install

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

data-smoke:
	@echo "Generating smoke dataset..."
	python data/generate_smoke.py --output data/smoke/sequences_smoke.csv
	@echo "Smoke dataset ready at data/smoke/sequences_smoke.csv"

data-full:
	python data/download.py --output data/full/ --source zenodo

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Quality
# ------------------------------------------------------------

test:
	WANDB_MODE=disabled pytest tests/ -v --cov=src/antibody_seq_ml --cov-report=term-missing

lint:
	pre-commit run --all-files

# ------------------------------------------------------------
# Notebooks
# ------------------------------------------------------------

notebook:
	jupyter notebook notebooks/01_eda.ipynb
