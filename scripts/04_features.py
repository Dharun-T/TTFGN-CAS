#!/usr/bin/env python3
"""Phase 4 Feature Engineering Script.

Computes pre-prediction longitudinal laboratory trajectory features
and patient clinical context matrices strictly prior to t_prediction.
Saves aligned matrices and metadata for model training.
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.temporal_labs import compute_analyte_trajectory_features
from src.features.clinical_context import (
    extract_patient_demographics,
    extract_active_medication_context,
    extract_diagnosis_context,
    extract_vitals_severity_context,
)
from src.preprocessing.lab_standardization import standardize_labevents_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("04_features")

ANALYTE_TO_IDX = {
    "glucose": 0,
    "creatinine": 1,
    "potassium": 2,
    "sodium": 3,
    "hemoglobin": 4,
}


def main():
    features_dir = Path("data/features")
    processed_dir = Path("data/processed")
    extracted_dir = Path("data/extracted")
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading prediction instances and clinical relational tables...")
    instances_df = pd.read_csv(processed_dir / "prediction_instances.csv")
    instances_df["t_prediction"] = pd.to_datetime(instances_df["t_prediction"])
    instances_df["t_lab"] = pd.to_datetime(instances_df["t_lab"])

    patients_df = pd.read_csv(extracted_dir / "patients.csv").set_index("subject_id")
    icustays_df = pd.read_csv(extracted_dir / "icustays.csv")
    icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
    diagnoses_df = pd.read_csv(extracted_dir / "diagnoses_icd.csv")
    prescriptions_df = pd.read_csv(extracted_dir / "prescriptions.csv")
    prescriptions_df["starttime"] = pd.to_datetime(prescriptions_df["starttime"])
    prescriptions_df["stoptime"] = pd.to_datetime(prescriptions_df["stoptime"])
    chartevents_df = pd.read_csv(extracted_dir / "chartevents.csv")
    chartevents_df["charttime"] = pd.to_datetime(chartevents_df["charttime"])

    labevents_std = standardize_labevents_dataframe(pd.read_csv(extracted_dir / "labevents.csv"))
    labevents_std["charttime"] = pd.to_datetime(labevents_std["charttime"])

    lab_feature_rows = []
    elapsed_hour_rows = []
    analyte_idx_rows = []
    context_feature_rows = []
    labels = []

    lab_feat_names = [
        "current_prev_val", "prev_val", "delta", "time_since_last_meas_hrs",
        "rate_of_change", "recent_mean", "recent_std", "recent_min",
        "recent_max", "meas_count", "direction_of_change", "trend_slope"
    ]
    context_feat_names = []

    logger.info(f"Extracting temporal and clinical features for {len(instances_df)} instances...")

    for idx, row in instances_df.iterrows():
        subj_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        t_pred = row["t_prediction"]
        analyte = str(row["analyte"]).lower()

        # 1. Longitudinal Lab Trajectory features (charttime <= t_pred and within 48h)
        lookback_start = t_pred - pd.Timedelta(hours=48.0)
        pt_labs = labevents_std[
            (labevents_std["subject_id"] == subj_id) &
            (labevents_std["analyte"] == analyte) &
            (labevents_std["charttime"] >= lookback_start) &
            (labevents_std["charttime"] <= t_pred)
        ]
        lab_feats = compute_analyte_trajectory_features(pt_labs, t_pred, lookback_hours=48.0)
        lab_vec = [lab_feats[k] for k in lab_feat_names]
        elapsed_val = lab_feats["time_since_last_meas_hrs"]

        # 2. Patient Demographics
        pt_row = patients_df.loc[subj_id] if subj_id in patients_df.index else None
        demo_dict = extract_patient_demographics(pt_row)

        # 3. Active Meds prior to t_pred
        med_dict = extract_active_medication_context(subj_id, t_pred, prescriptions_df)

        # 4. Chronic Diagnoses
        diag_dict = extract_diagnosis_context(subj_id, hadm_id, diagnoses_df)

        # 5. Vitals and ICU stay severity
        icu_sub = icustays_df[icustays_df["subject_id"] == subj_id]
        icu_intime = icu_sub["intime"].iloc[0] if not icu_sub.empty else None
        vitals_sub = chartevents_df[chartevents_df["subject_id"] == subj_id]
        vitals_dict = extract_vitals_severity_context(vitals_sub, t_pred, icu_intime)

        # Assemble full context vector
        full_context_dict = {**demo_dict, **med_dict, **diag_dict, **vitals_dict}
        if not context_feat_names:
            context_feat_names = list(full_context_dict.keys())
        context_vec = [full_context_dict[k] for k in context_feat_names]

        lab_feature_rows.append(lab_vec)
        elapsed_hour_rows.append(elapsed_val)
        analyte_idx_rows.append(ANALYTE_TO_IDX[analyte])
        context_feature_rows.append(context_vec)
        labels.append(int(row["action_occurred"]))

    X_lab = np.array(lab_feature_rows, dtype=np.float32)
    elapsed_hrs = np.array(elapsed_hour_rows, dtype=np.float32)
    analyte_indices = np.array(analyte_idx_rows, dtype=np.int64)
    X_context = np.array(context_feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # Save feature matrices
    out_npz = features_dir / "features_dataset.npz"
    np.savez_compressed(
        out_npz,
        X_lab=X_lab,
        elapsed_hrs=elapsed_hrs,
        analyte_indices=analyte_indices,
        X_context=X_context,
        y=y,
        lab_feat_names=np.array(lab_feat_names),
        context_feat_names=np.array(context_feat_names),
    )

    feature_metadata = {
        "num_instances": len(y),
        "lab_feature_dim": X_lab.shape[1],
        "context_feature_dim": X_context.shape[1],
        "lab_feature_names": lab_feat_names,
        "context_feature_names": context_feat_names,
        "positive_rate": float(np.mean(y)),
    }
    with open(features_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=2)

    logger.info(f"Feature dataset saved to {out_npz} (Lab dim: {X_lab.shape[1]}, Context dim: {X_context.shape[1]})")


if __name__ == "__main__":
    main()
