#!/usr/bin/env python3
"""Phase 3 Actionability Label Generation Script.

Constructs prediction instances for eligible upcoming laboratory reports
for Glucose, Creatinine, Potassium, Sodium, and Hemoglobin.
Evaluates downstream clinical actionability strictly within (t_lab, t_lab + 12h].
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cohort_builder import construct_prediction_instances
from src.utils.leakage import verify_prediction_instances_temporal_validity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("03_create_labels")


def main():
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    extracted_dir = Path("data/extracted")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cohort, standardized labevents, and medication records...")
    cohort_df = pd.read_csv(interim_dir / "cohort.csv")
    labevents_df = pd.read_csv(extracted_dir / "labevents.csv")
    prescriptions_df = pd.read_csv(extracted_dir / "prescriptions.csv")
    prescriptions_df["starttime"] = pd.to_datetime(prescriptions_df["starttime"])

    logger.info("Constructing temporal prediction instances (48h lookback, 30m lead time, 12h action window)...")
    instances_df = construct_prediction_instances(
        cohort_df=cohort_df,
        labevents_df=labevents_df,
        prescriptions_df=prescriptions_df,
        target_analytes=["glucose", "creatinine", "potassium", "sodium", "hemoglobin"],
        lookback_hours=48.0,
        lead_time_minutes=30.0,
        action_window_hours=12.0,
        min_prior_labs=2,
    )

    logger.info(f"Generated {len(instances_df)} eligible prediction instances.")

    # Audit Leakage and Temporal Constraints
    logger.info("Executing automated temporal leakage verification...")
    audit = verify_prediction_instances_temporal_validity(instances_df, min_lead_time_minutes=30.0)
    logger.info(f"Leakage Audit Status: {audit['status']} (Checked {audit['total_instances_checked']} instances)")

    # Label statistics
    pos_count = int(instances_df["action_occurred"].sum())
    pos_rate = round(pos_count / max(1, len(instances_df)), 4)
    logger.info(f"Actionability Label Distribution: {pos_count}/{len(instances_df)} positive ({pos_rate * 100:.1f}%)")
    logger.info("Action Type Breakdown:")
    for act_type, count in instances_df["action_type"].value_counts().items():
        logger.info(f"  {act_type:<26}: {count}")

    output_path = processed_dir / "prediction_instances.csv"
    instances_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction instances to {output_path}")


if __name__ == "__main__":
    main()
