"""Download full OAS paired antibody dataset (~500K sequences).

Usage (on Kaggle or locally with bandwidth):
    python data/download.py --output data/full/
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

# OAS bulk download index for paired human sequences
OAS_PAIRED_INDEX = (
    "http://opig.stats.ox.ac.uk/webapps/oas/api/paired/?species=human&limit=1000"
)

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


def download_oas_unit(url: str, retries: int = 3) -> pd.DataFrame | None:
    """Download a single OAS data unit CSV and return parsed DataFrame."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            # OAS CSVs have a metadata header row — skip it
            df = pd.read_csv(io.StringIO(resp.text), header=1)
            return df
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"  Failed to download {url}: {exc}")
    return None


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


def download(output_dir: str, max_units: int = 200) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Fetching OAS paired index...")
    try:
        resp = requests.get(OAS_PAIRED_INDEX, timeout=30)
        resp.raise_for_status()
        index_data = resp.json()
    except Exception as exc:
        print(f"Failed to fetch OAS index: {exc}")
        print(
            "Please download OAS data manually from http://opig.stats.ox.ac.uk/webapps/oas/"
        )
        return

    units = index_data.get("results", [])[:max_units]
    print(f"Found {len(units)} data units to download")

    all_dfs = []
    for i, unit in enumerate(units):
        url = unit.get("data_url", "")
        if not url:
            continue
        print(f"  [{i+1}/{len(units)}] {url}")
        raw = download_oas_unit(url)
        if raw is not None and len(raw) > 0:
            processed = preprocess(raw)
            if len(processed) > 0:
                all_dfs.append(processed)

    if not all_dfs:
        print("No data downloaded.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["cdr_h3"]).reset_index(drop=True)
    full_df["sequence_id"] = [f"seq_{i:06d}" for i in range(len(full_df))]

    out_csv = output_path / "sequences_full.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(full_df)} sequences → {out_csv}")
    print(full_df["length_class"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full OAS dataset")
    parser.add_argument("--output", default="data/full/", help="Output directory")
    parser.add_argument(
        "--max-units", type=int, default=200, help="Max OAS units to download"
    )
    args = parser.parse_args()
    download(args.output, args.max_units)
