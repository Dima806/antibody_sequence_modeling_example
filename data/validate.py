"""Validate a sequences DataFrame against configs/schema.yaml."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "schema.yaml"


def validate(df: pd.DataFrame, schema_path: Path = SCHEMA_PATH) -> None:
    """Raise ValueError if df does not conform to the schema.

    Checks:
    - All required columns are present
    - No nulls in non-nullable columns
    - dtype compatibility (str / int / float)
    - Value ranges and allowed values
    - Derived invariants (cdr_h3_length, length_class boundaries)
    """
    schema = yaml.safe_load(schema_path.read_text())
    columns = schema["columns"]
    errors: list[str] = []

    # 1. Required columns present
    missing = [col for col in columns if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        raise ValueError("\n".join(errors))

    for col, rules in columns.items():
        series = df[col]

        # 2. Nullability
        if not rules.get("nullable", True) and series.isna().any():
            errors.append(f"[{col}] contains nulls")

        dtype = rules.get("dtype")

        # 3. String rules
        if dtype == "str":
            lengths = series.dropna().str.len()
            if "min_length" in rules and (lengths < rules["min_length"]).any():
                n = (lengths < rules["min_length"]).sum()
                errors.append(
                    f"[{col}] {n} values shorter than min_length={rules['min_length']}"
                )
            if "max_length" in rules and (lengths > rules["max_length"]).any():
                n = (lengths > rules["max_length"]).sum()
                errors.append(
                    f"[{col}] {n} values longer than max_length={rules['max_length']}"
                )
            if "values" in rules:
                bad = ~series.dropna().isin(rules["values"])
                if bad.any():
                    errors.append(
                        f"[{col}] unexpected values: {series[bad].unique().tolist()}"
                    )

        # 4. Numeric rules (int / float)
        if dtype in ("int", "float"):
            numeric = pd.to_numeric(series, errors="coerce")
            if "min" in rules and (numeric < rules["min"]).any():
                n = (numeric < rules["min"]).sum()
                errors.append(f"[{col}] {n} values below min={rules['min']}")
            if "max" in rules and (numeric > rules["max"]).any():
                n = (numeric > rules["max"]).sum()
                errors.append(f"[{col}] {n} values above max={rules['max']}")

    # 5. Derived invariants
    if "cdr_h3" in df.columns and "cdr_h3_length" in df.columns:
        mismatch = df["cdr_h3_length"] != df["cdr_h3"].str.len()
        if mismatch.any():
            errors.append(f"cdr_h3_length != len(cdr_h3) for {mismatch.sum()} rows")

    if "cdr_h3_length" in df.columns and "length_class" in df.columns:
        length = df["cdr_h3_length"]
        cls = df["length_class"]
        bad_short = ((length <= 9) & (cls != "short")).sum()
        bad_medium = ((length >= 10) & (length <= 14) & (cls != "medium")).sum()
        bad_long = ((length >= 15) & (cls != "long")).sum()
        if bad_short:
            errors.append(f"{bad_short} rows with length<=9 not labelled 'short'")
        if bad_medium:
            errors.append(
                f"{bad_medium} rows with 10<=length<=14 not labelled 'medium'"
            )
        if bad_long:
            errors.append(f"{bad_long} rows with length>=15 not labelled 'long'")

    if errors:
        raise ValueError(
            f"Schema validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    print(f"Schema validation passed — {len(df):,} rows, {len(df.columns)} columns.")
