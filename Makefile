.PHONY: setup data-smoke data-full train-smoke sweep-cpu test prek notebook

setup:
	pip install uv && uv pip install -e . --system
	uv tool install prek && prek install

data-smoke:
	@echo "Generating smoke dataset..."
	python data/generate_smoke.py --output data/smoke/sequences_smoke.csv
	@echo "Smoke dataset ready at data/smoke/sequences_smoke.csv"

data-full:
	python data/download.py --output data/full/

train-smoke:
	WANDB_MODE=disabled python -m antibody_seq_ml.train \
		--config configs/default.yaml \
		--data data/smoke/sequences_smoke.csv \
		--smoke

sweep-cpu:
	python -m antibody_seq_ml.sweep \
		--config configs/sweep_cpu.yaml \
		--data data/smoke/sequences_smoke.csv

test:
	WANDB_MODE=disabled pytest tests/ -v --cov=src/antibody_seq_ml --cov-report=term-missing

prek:
	pre-commit run --all-files

notebook:
	jupyter notebook notebooks/01_eda.ipynb
