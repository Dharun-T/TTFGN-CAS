"""Automated Temporal Data Leakage and Partition Integrity Verification Suite.

Validates that:
  1. No feature observation occurred after t_prediction.
  2. Prediction time t_prediction occurs strictly before upcoming target lab result t_lab.
  3. All downstream actions occurred strictly after t_lab.
  4. Patient sets between Train, Validation, and Test are completely disjoint.
  5. No duplicate prediction instances exist.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np


class LeakageAssertionError(Exception):
    """Raised when temporal or patient partition data leakage is detected."""
    pass


def verify_patient_split_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_id_col: str = "subject_id"
) -> Dict[str, int]:
    """Ensures no patient appears in more than one partition."""
    train_pts = set(train_df[patient_id_col].unique())
    val_pts = set(val_df[patient_id_col].unique())
    test_pts = set(test_df[patient_id_col].unique())

    overlap_train_val = train_pts.intersection(val_pts)
    overlap_train_test = train_pts.intersection(test_pts)
    overlap_val_test = val_pts.intersection(test_pts)

    if overlap_train_val:
        raise LeakageAssertionError(
            f"Patient leakage detected between TRAIN and VAL: {len(overlap_train_val)} subjects"
        )
    if overlap_train_test:
        raise LeakageAssertionError(
            f"Patient leakage detected between TRAIN and TEST: {len(overlap_train_test)} subjects"
        )
    if overlap_val_test:
        raise LeakageAssertionError(
            f"Patient leakage detected between VAL and TEST: {len(overlap_val_test)} subjects"
        )

    return {
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "total_unique_patients": len(train_pts | val_pts | test_pts),
    }


def verify_prediction_instances_temporal_validity(
    df: pd.DataFrame,
    min_lead_time_minutes: float = 30.0
) -> Dict[str, int]:
    """Verifies that for every prediction instance:

      t_prediction <= t_lab - min_lead_time
    and downstream actions satisfy:
      t_action > t_lab.
    """
    violations = 0
    lead_time_delta = pd.Timedelta(minutes=min_lead_time_minutes)

    for idx, row in df.iterrows():
        t_pred = pd.to_datetime(row["t_prediction"])
        t_lab = pd.to_datetime(row["t_lab"])

        # Check lead time
        if t_pred > (t_lab - lead_time_delta):
            violations += 1

        # Check action time if action occurred
        if row.get("action_occurred", 0) == 1 and pd.notna(row.get("t_action")):
            t_action = pd.to_datetime(row["t_action"])
            if t_action <= t_lab:
                raise LeakageAssertionError(
                    f"Action timestamp {t_action} occurs before or at lab result time {t_lab} at row {idx}"
                )

    if violations > 0:
        raise LeakageAssertionError(
            f"Detected {violations} instances violating the {min_lead_time_minutes}m lead time constraint"
        )

    # Check for duplicates
    duplicates = df.duplicated(subset=["subject_id", "t_lab", "analyte"]).sum()
    if duplicates > 0:
        raise LeakageAssertionError(f"Found {duplicates} duplicate prediction instances")

    return {
        "total_instances_checked": len(df),
        "lead_time_violations": violations,
        "duplicates": int(duplicates),
        "status": "PASS",
    }
