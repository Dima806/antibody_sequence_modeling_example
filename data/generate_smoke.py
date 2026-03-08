"""Generate a synthetic stratified smoke dataset (~2,000 CDR-H3 sequences)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
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


def random_sequence(length: int) -> str:
    return "".join(random.choices(AA, k=length))


def gravy(sequence: str) -> float:
    """Grand Average of hYdropathicity (Kyte-Doolittle)."""
    return sum(KYTE_DOOLITTLE[aa] for aa in sequence.upper()) / len(sequence)


def classify_length(length: int) -> str:
    if length <= 9:
        return "short"
    elif length <= 14:
        return "medium"
    else:
        return "long"


def generate(output_path: str, n_per_class: dict[str, int] | None = None) -> None:
    if n_per_class is None:
        n_per_class = {"short": 700, "medium": 700, "long": 600}

    rows = []
    seq_id = 0

    for cls, count in n_per_class.items():
        if cls == "short":
            lengths = np.random.randint(5, 10, size=count)
        elif cls == "medium":
            lengths = np.random.randint(10, 15, size=count)
        else:
            lengths = np.random.randint(15, 26, size=count)

        for length in lengths:
            cdr_h3 = random_sequence(int(length))
            # Build a fake heavy chain: ~70 AA framework prefix, CDR-H3, ~30 AA suffix
            prefix = random_sequence(70)
            suffix = random_sequence(30)
            heavy_chain = prefix + cdr_h3 + suffix

            rows.append(
                {
                    "sequence_id": f"seq_{seq_id:05d}",
                    "heavy_chain_sequence": heavy_chain,
                    "cdr_h3": cdr_h3,
                    "cdr_h3_length": len(cdr_h3),
                    "length_class": cls,
                    "hydrophobicity": round(gravy(cdr_h3), 4),
                }
            )
            seq_id += 1

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sequences → {output_path}")
    print(df["length_class"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate smoke dataset")
    parser.add_argument(
        "--output",
        default="data/smoke/sequences_smoke.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    generate(args.output)
