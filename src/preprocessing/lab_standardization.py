"""Laboratory Unit Standardization and Outlier Handling for T-TFGN.

Ensures all laboratory observations for the five target analytes (Glucose,
Creatinine, Potassium, Sodium, Hemoglobin) are converted to canonical units,
cleans impossible physiological extremes, and validates unit tags.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical units and plausible physiological cutoffs for adult ICU patients
CANONICAL_LAB_SPECS: Dict[str, Dict] = {
    "glucose": {
        "canonical_unit": "mg/dL",
        "plausible_min": 10.0,
        "plausible_max": 2000.0,
        "conversions": {
            "mmol/L": lambda v: v * 18.0182,
            "mg/dL": lambda v: v,
        },
    },
    "creatinine": {
        "canonical_unit": "mg/dL",
        "plausible_min": 0.1,
        "plausible_max": 35.0,
        "conversions": {
            "umol/L": lambda v: v * 0.011312,
            "µmol/L": lambda v: v * 0.011312,
            "mg/dL": lambda v: v,
        },
    },
    "potassium": {
        "canonical_unit": "mEq/L",
        "plausible_min": 1.0,
        "plausible_max": 12.0,
        "conversions": {
            "mmol/L": lambda v: v,
            "mEq/L": lambda v: v,
        },
    },
    "sodium": {
        "canonical_unit": "mEq/L",
        "plausible_min": 85.0,
        "plausible_max": 200.0,
        "conversions": {
            "mmol/L": lambda v: v,
            "mEq/L": lambda v: v,
        },
    },
    "hemoglobin": {
        "canonical_unit": "g/dL",
        "plausible_min": 2.0,
        "plausible_max": 28.0,
        "conversions": {
            "g/L": lambda v: v * 0.1,
            "mmol/L": lambda v: v * 1.611,
            "g/dL": lambda v: v,
        },
    },
}

# Standard MIMIC-IV item IDs for the 5 target analytes
ITEMID_TO_ANALYTE: Dict[int, str] = {
    50931: "glucose",
    50809: "glucose",
    52569: "glucose",
    50912: "creatinine",
    52546: "creatinine",
    50971: "potassium",
    50822: "potassium",
    52610: "potassium",
    50983: "sodium",
    50824: "sodium",
    52623: "sodium",
    51222: "hemoglobin",
    50811: "hemoglobin",
}


def standardize_lab_value(
    analyte: str,
    raw_val: float,
    unit: Optional[str]
) -> Tuple[Optional[float], str]:
    """Standardizes a single lab value to its canonical unit and flags anomalies.

    Returns:
        (standardized_value, status_code)
    """
    if np.isnan(raw_val) or raw_val is None:
        return None, "MISSING"

    spec = CANONICAL_LAB_SPECS.get(analyte.lower())
    if not spec:
        return raw_val, "UNMAPPED_ANALYTE"

    clean_unit = str(unit).strip() if unit is not None else spec["canonical_unit"]
    conversion_func = spec["conversions"].get(clean_unit)

    if conversion_func:
        val_conv = conversion_func(float(raw_val))
    else:
        # Fallback to assuming canonical if unit is missing or unlisted
        val_conv = float(raw_val)

    if val_conv < spec["plausible_min"] or val_conv > spec["plausible_max"]:
        return None, f"OUT_OF_PHYSIOLOGICAL_RANGE ({val_conv:.2f} {spec['canonical_unit']})"

    return round(val_conv, 3), "VALID"


def standardize_labevents_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes an entire labevents DataFrame with itemid and valueuom columns.

    Enforces analyte recognition, standard units, and physiological bounds.
    """
    df = df.copy()
    if "analyte" not in df.columns and "itemid" in df.columns:
        df["analyte"] = df["itemid"].map(ITEMID_TO_ANALYTE)

    df = df[df["analyte"].notna()].copy()

    std_values = []
    status_flags = []
    for _, row in df.iterrows():
        val, status = standardize_lab_value(
            str(row["analyte"]),
            row["valuenum"] if "valuenum" in row else row.get("value"),
            row.get("valueuom")
        )
        std_values.append(val)
        status_flags.append(status)

    df["std_valuenum"] = std_values
    df["validation_status"] = status_flags
    return df[df["validation_status"] == "VALID"].copy()
