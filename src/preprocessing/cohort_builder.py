"""Cohort Construction and Temporal Prediction Instance Assembler.

Constructs the adult ICU patient cohort (age >= 18, ICU LOS >= 24h),
identifies eligible upcoming laboratory reports for the 5 target analytes,
and constructs temporal prediction instances with strict lead times.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing.lab_standardization import standardize_labevents_dataframe, ITEMID_TO_ANALYTE
from src.labels.actionability import evaluate_actionability_for_lab

logger = logging.getLogger(__name__)


def build_icu_cohort(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    icustays_df: pd.DataFrame,
    min_age: int = 18,
    min_los_hours: float = 24.0
) -> pd.DataFrame:
    """Builds the eligible adult ICU cohort."""
    # Merge demographics and ICU stays
    merged = icustays_df.merge(patients_df[["subject_id", "gender", "anchor_age"]], on="subject_id", how="inner")
    merged = merged.merge(admissions_df[["hadm_id", "admittime", "admission_type"]], on="hadm_id", how="inner")

    # Filter age >= min_age
    merged = merged[merged["anchor_age"] >= min_age]

    # Filter ICU length of stay >= min_los_hours
    # los in icustays is in days
    if "los" in merged.columns:
        merged = merged[merged["los"] * 24.0 >= min_los_hours]

    return merged.reset_index(drop=True)


def construct_prediction_instances(
    cohort_df: pd.DataFrame,
    labevents_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    target_analytes: List[str] = ["glucose", "creatinine", "potassium", "sodium", "hemoglobin"],
    lookback_hours: float = 48.0,
    lead_time_minutes: float = 30.0,
    action_window_hours: float = 12.0,
    min_prior_labs: int = 2
) -> pd.DataFrame:
    """Constructs temporal prediction instances for eligible upcoming lab reports.

    Temporal Assertion:
        t_prediction = t_lab - lead_time_minutes
        Prior history: [t_prediction - lookback_hours, t_prediction]
        Downstream window: (t_lab, t_lab + action_window_hours]
    """
    # Standardize labevents
    labs_std = standardize_labevents_dataframe(labevents_df)
    labs_std["charttime"] = pd.to_datetime(labs_std["charttime"])

    lead_delta = pd.Timedelta(minutes=lead_time_minutes)
    lookback_delta = pd.Timedelta(hours=lookback_hours)

    instances = []
    eligible_subjects = set(cohort_df["subject_id"].unique())

    # Filter labs to eligible cohort
    labs_cohort = labs_std[labs_std["subject_id"].isin(eligible_subjects)].sort_values("charttime")

    for (subject_id, analyte), group in labs_cohort.groupby(["subject_id", "analyte"]):
        if analyte not in target_analytes:
            continue

        obs_list = group.to_dict("records")
        for i in range(len(obs_list)):
            target_lab = obs_list[i]
            t_lab = target_lab["charttime"]
            t_prediction = t_lab - lead_delta
            window_start = t_prediction - lookback_delta

            # Find prior labs for this patient strictly within lookback
            prior_labs = [
                row for row in obs_list[:i]
                if window_start <= row["charttime"] <= t_prediction
            ]

            if len(prior_labs) < min_prior_labs:
                continue

            # Evaluate downstream actionability in (t_lab, t_lab + action_window_hours]
            hadm_id = target_lab.get("hadm_id", 0)
            stay_id = target_lab.get("stay_id")

            action_res = evaluate_actionability_for_lab(
                subject_id=subject_id,
                hadm_id=hadm_id,
                stay_id=stay_id,
                analyte=analyte,
                t_lab=t_lab,
                prescriptions_df=prescriptions_df,
                subsequent_labs_df=labs_std,
                action_window_hours=action_window_hours,
            )

            instances.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "analyte": analyte,
                "t_prediction": t_prediction,
                "t_lab": t_lab,
                "target_lab_value": target_lab["std_valuenum"],
                "target_lab_unit": target_lab.get("valueuom", ""),
                "action_occurred": action_res.action_occurred,
                "action_type": action_res.action_type,
                "time_to_action_hours": action_res.time_to_action_hours,
                "specific_intervention": action_res.specific_intervention,
                "prior_lab_count": len(prior_labs),
            })

    instances_df = pd.DataFrame(instances)
    return instances_df
