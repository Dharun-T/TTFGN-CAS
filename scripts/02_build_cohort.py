#!/usr/bin/env python3
"""Phase 2 Cohort Selection Script.

Filters adult ICU admissions, verifies minimum length of stay (>= 24 hours),
links demographic and administrative records, and produces the base cohort.
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cohort_builder import build_icu_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("02_build_cohort")


def main():
    extracted_dir = Path("data/extracted")
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading extracted patient, admission, and ICU stay records...")
    patients_df = pd.read_csv(extracted_dir / "patients.csv")
    admissions_df = pd.read_csv(extracted_dir / "admissions.csv")
    icustays_df = pd.read_csv(extracted_dir / "icustays.csv")

    logger.info(f"Total raw patients: {len(patients_df)}, ICU stays: {len(icustays_df)}")
    cohort_df = build_icu_cohort(
        patients_df=patients_df,
        admissions_df=admissions_df,
        icustays_df=icustays_df,
        min_age=18,
        min_los_hours=24.0,
    )

    output_path = interim_dir / "cohort.csv"
    cohort_df.to_csv(output_path, index=False)
    logger.info(f"Eligible adult ICU cohort constructed: {len(cohort_df)} stays across {cohort_df['subject_id'].nunique()} unique patients.")
    logger.info(f"Saved cohort to {output_path}")


if __name__ == "__main__":
    main()
