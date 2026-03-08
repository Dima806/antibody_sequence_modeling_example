"""Download full OAS paired antibody dataset.

Supported sources:
  zenodo   — p-IgGen pre-processed OAS snapshot on Zenodo (default, no auth)
  oas-api  — OAS REST API at opig.stats.ox.ac.uk (no auth required)

Usage:
    python data/download.py --output data/full/
    python data/download.py --output data/full/ --source oas-api
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.validate import validate

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

OAS_API_URL = (
    "http://opig.stats.ox.ac.uk/webapps/oas/api/paired/?species=human&limit=1000"
)
# p-IgGen: pre-processed OAS paired sequences, no auth required
ZENODO_URL = "https://zenodo.org/api/records/13880874/files/paired_oas_human_train.csv.gz/content"


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


# CDR-H3 is flanked by a conserved Cys and the WGxG motif of the J gene
_CDR_H3_RE = re.compile(r"C([A-Z]{3,30})WG[A-Z]G", re.IGNORECASE)


def extract_cdr_h3(sequence: str) -> str:
    """Extract CDR-H3 from a heavy chain sequence using the C...WGxG motif."""
    m = _CDR_H3_RE.search(sequence)
    return m.group(1).upper() if m else ""


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
    # p-IgGen / AbBFN2: no named header — first column is heavy sequence
    if seq_col is None and df.shape[1] >= 1:
        seq_col = df.columns[0]

    if seq_col is None:
        print(f"  Could not find heavy sequence column. Available: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["sequence_id"] = df.index.astype(str)
    out["heavy_chain_sequence"] = df[seq_col].fillna("").str.upper()

    if cdr_col is not None:
        out["cdr_h3"] = df[cdr_col].fillna("").str.upper()
    else:
        print("  No CDR-H3 column found — extracting via C...WGxG motif.")
        out["cdr_h3"] = out["heavy_chain_sequence"].apply(extract_cdr_h3)

    out = out[out["cdr_h3"].str.len() >= 5]
    out["cdr_h3_length"] = out["cdr_h3"].str.len()
    out["length_class"] = out["cdr_h3_length"].apply(classify_length)
    out["hydrophobicity"] = out["cdr_h3"].apply(gravy).round(4)
    return out.reset_index(drop=True)


def _save(df: pd.DataFrame, output_path: Path) -> None:
    df = df.drop_duplicates(subset=["cdr_h3"]).reset_index(drop=True)
    df["sequence_id"] = [f"seq_{i:06d}" for i in range(len(df))]
    validate(df)
    out_csv = output_path / "sequences_full.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df):,} sequences → {out_csv}")
    print(df["length_class"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Source: OAS REST API
# ---------------------------------------------------------------------------


def _fetch_oas_unit(url: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            # OAS CSVs have a metadata header row — skip it
            return pd.read_csv(io.StringIO(resp.text), header=1)
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"  Failed: {url} — {exc}")
    return None


def download_oas_api(output_path: Path, max_units: int = 200) -> None:
    print(f"Fetching OAS paired index ({OAS_API_URL}) ...")
    try:
        resp = requests.get(OAS_API_URL, timeout=30)
        resp.raise_for_status()
        index_data = resp.json()
    except Exception as exc:
        print(f"Failed to fetch OAS index: {exc}")
        return

    units = index_data.get("results", [])[:max_units]
    print(f"Found {len(units)} data units to download.")

    all_dfs: list[pd.DataFrame] = []
    for i, unit in enumerate(units):
        url = unit.get("data_url", "")
        if not url:
            continue
        print(f"  [{i + 1}/{len(units)}] {url}")
        raw = _fetch_oas_unit(url)
        if raw is not None and len(raw) > 0:
            processed = preprocess(raw)
            if len(processed) > 0:
                all_dfs.append(processed)

    if not all_dfs:
        print("No data downloaded.")
        return

    _save(pd.concat(all_dfs, ignore_index=True), output_path)


# ---------------------------------------------------------------------------
# Source: Zenodo (p-IgGen pre-processed OAS snapshot)
# ---------------------------------------------------------------------------


def download_zenodo(output_path: Path) -> None:
    print(f"Downloading p-IgGen OAS snapshot from Zenodo:\n  {ZENODO_URL}")
    resp = requests.get(ZENODO_URL, stream=True, timeout=120)
    resp.raise_for_status()

    gz_path = output_path / "paired_oas_human_train.csv.gz"
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with gz_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="\r")
    print(f"\nDownloaded → {gz_path}")

    print("Parsing CSV ...")
    df = pd.read_csv(gz_path, compression="gzip")
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")
    processed = preprocess(df)
    if processed.empty:
        print("No valid sequences found after preprocessing.")
        return
    gz_path.unlink()  # remove raw gz once parsed
    _save(processed, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def download(
    output_dir: str,
    source: str = "zenodo",
    max_units: int = 200,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source == "zenodo":
        download_zenodo(output_path)
    elif source == "oas-api":
        download_oas_api(output_path, max_units=max_units)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose: zenodo, oas-api")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download OAS paired dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="data/full/", help="Output directory")
    parser.add_argument(
        "--source",
        default="zenodo",
        choices=["zenodo", "oas-api"],
        help="Download source (default: zenodo)",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=200,
        help="Max OAS data units to download — oas-api only",
    )
    args = parser.parse_args()
    download(args.output, source=args.source, max_units=args.max_units)
