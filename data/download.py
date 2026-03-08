"""Download full OAS paired antibody dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/bloyal/oas-paired-sequence-data

Usage (on Kaggle or locally):
    python data/download.py --output data/full/
    python data/download.py --output data/full/ --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

HF_DATASET = "bloyal/oas-paired-sequence-data"


def gravy(sequence: str) -> float:
    sequence = sequence.upper()
    valid = [aa for aa in sequence if aa in KYTE_DOOLITTLE]
    if not valid:
        return 0.0
    return sum(KYTE_DOOLITTLE[aa] for aa in valid) / len(valid)


def classify_length(length: int) -> str:
    if length <= 9:
        return "short"
    elif length <= 14:
        return "medium"
    else:
        return "long"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Extract CDR-H3, compute labels, drop rows with missing data."""
    cdr_col = next(
        (c for c in df.columns if "cdr3" in c.lower() and "h" in c.lower()),
        None,
    )
    seq_col = next(
        (c for c in df.columns if "sequence" in c.lower() and "heavy" in c.lower()),
        None,
    )
    if cdr_col is None or seq_col is None:
        print(f"  Could not find required columns. Available: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["sequence_id"] = df.index.astype(str)
    out["heavy_chain_sequence"] = df[seq_col].fillna("")
    out["cdr_h3"] = df[cdr_col].fillna("")
    out = out[out["cdr_h3"].str.len() >= 5]
    out["cdr_h3"] = out["cdr_h3"].str.upper()
    out["cdr_h3_length"] = out["cdr_h3"].str.len()
    out["length_class"] = out["cdr_h3_length"].apply(classify_length)
    out["hydrophobicity"] = out["cdr_h3"].apply(gravy).round(4)
    return out.reset_index(drop=True)


def download(output_dir: str, split: str = "train") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading HuggingFace dataset: {HF_DATASET} (split={split}) ...")
    hf_ds = load_dataset(HF_DATASET, split=split)
    df = hf_ds.to_pandas()
    print(f"Downloaded {len(df):,} rows with columns: {list(df.columns)}")

    print("Preprocessing ...")
    processed = preprocess(df)

    if processed.empty:
        print("No valid sequences found after preprocessing.")
        return

    processed = processed.drop_duplicates(subset=["cdr_h3"]).reset_index(drop=True)
    processed["sequence_id"] = [f"seq_{i:06d}" for i in range(len(processed))]

    out_csv = output_path / "sequences_full.csv"
    processed.to_csv(out_csv, index=False)
    print(f"\nSaved {len(processed):,} sequences → {out_csv}")
    print(processed["length_class"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download OAS paired dataset from HuggingFace"
    )
    parser.add_argument("--output", default="data/full/", help="Output directory")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (train / validation / test)",
    )
    args = parser.parse_args()
    download(args.output, args.split)
