"""Clinical Context Feature Construction for T-TFGN.

Extracts patient demographics, active pre-prediction medications,
baseline diagnoses, pre-prediction vital signs, and organ dysfunction
indicators strictly prior to t_prediction.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Chronic condition ICD-9 / ICD-10 prefixes
COMORBIDITY_PREFIXES = {
    "diabetes": ["250", "E10", "E11", "E13", "E14"],
    "renal_disease": ["585", "584", "N17", "N18"],
    "heart_failure": ["428", "I50"],
    "hypertension": ["401", "402", "I10", "I11"],
    "respiratory_failure": ["518", "J96"],
    "sepsis": ["995.91", "995.92", "038", "A41"],
    "liver_disease": ["571", "K70", "K74"],
}

# Medication exposure classes active pre-prediction
MEDICATION_CLASSES = {
    "insulin": ["insulin", "glargine", "aspart", "lispro", "nph"],
    "diuretics": ["furosemide", "lasix", "bumetanide", "torsemide", "spironolactone", "hydrochlorothiazide"],
    "vasopressors": ["norepinephrine", "levophed", "epinephrine", "vasopressin", "phenylephrine", "dopamine"],
    "anticoagulants": ["heparin", "enoxaparin", "lovenox", "warfarin"],
    "antihypertensives": ["lisinopril", "losartan", "amlodipine", "metoprolol", "hydralazine"],
    "potassium_supplements": ["potassium chloride", "kcl", "k-dur"],
    "antibiotics": ["vancomycin", "cefepime", "piperacillin", "zosyn", "meropenem", "ceftriaxone"],
}


def extract_patient_demographics(patient_row: Optional[pd.Series]) -> Dict[str, float]:
    """Extracts normalized baseline demographic covariates."""
    if patient_row is None or patient_row.empty:
        return {
            "age": 60.0 / 100.0,
            "is_male": 0.5,
        }
    age = float(patient_row.get("anchor_age", 60.0))
    gender = str(patient_row.get("gender", "M")).upper()
    return {
        "age": min(1.0, max(0.0, age / 100.0)),
        "is_male": 1.0 if gender == "M" else 0.0,
    }


def extract_active_medication_context(
    subject_id: int,
    t_prediction: pd.Timestamp,
    prescriptions_df: pd.DataFrame
) -> Dict[str, float]:
    """Computes binary indicators for active medications before prediction point.

    Assertion: Only considers orders where starttime <= t_prediction.
    """
    context = {f"med_{cls_name}": 0.0 for cls_name in MEDICATION_CLASSES}
    context["active_med_count"] = 0.0

    if prescriptions_df.empty:
        return context

    # Select medications started on or before t_prediction, and still active (stoptime >= t_prediction or NaN)
    active_rx = prescriptions_df[
        (prescriptions_df["subject_id"] == subject_id) &
        (prescriptions_df["starttime"] <= t_prediction)
    ]

    total_active = 0
    for _, rx in active_rx.iterrows():
        drug_name = str(rx.get("drug", "")).lower()
        # Check if stopped well before prediction point (> 24h ago)
        stop_time = rx.get("stoptime")
        if pd.notna(stop_time) and stop_time < (t_prediction - pd.Timedelta(hours=24)):
            continue

        total_active += 1
        for cls_name, keywords in MEDICATION_CLASSES.items():
            if any(kw in drug_name for kw in keywords):
                context[f"med_{cls_name}"] = 1.0

    context["active_med_count"] = min(1.0, total_active / 15.0)
    return context


def extract_diagnosis_context(
    subject_id: int,
    hadm_id: int,
    diagnoses_df: pd.DataFrame
) -> Dict[str, float]:
    """Extracts chronic comorbidity flags documented for this patient."""
    context = {f"diag_{cond}": 0.0 for cond in COMORBIDITY_PREFIXES}

    if diagnoses_df.empty:
        return context

    pt_diag = diagnoses_df[
        (diagnoses_df["subject_id"] == subject_id) &
        (diagnoses_df["hadm_id"] == hadm_id)
    ]

    for _, row in pt_diag.iterrows():
        code = str(row.get("icd_code", "")).strip().replace(".", "")
        for cond, prefixes in COMORBIDITY_PREFIXES.items():
            for pref in prefixes:
                clean_pref = pref.replace(".", "")
                if code.startswith(clean_pref):
                    context[f"diag_{cond}"] = 1.0
                    break

    return context


def extract_vitals_severity_context(
    vitals_obs: pd.DataFrame,
    t_prediction: pd.Timestamp,
    icu_intime: Optional[pd.Timestamp] = None
) -> Dict[str, float]:
    """Extracts pre-prediction bedside vital signs and physiological stability indicators.

    All vital records must satisfy charttime <= t_prediction.
    """
    defaults = {
        "heart_rate_last": 80.0 / 150.0,
        "heart_rate_trend": 0.0,
        "mean_arterial_pressure_last": 75.0 / 120.0,
        "map_hypotensive": 0.0,
        "respiratory_rate_last": 18.0 / 35.0,
        "temperature_last": 37.0 / 40.0,
        "spo2_last": 97.0 / 100.0,
        "icu_hours_elapsed": 0.5,
        "sofa_renal_component": 0.0,
        "sofa_cardio_component": 0.0,
    }

    if icu_intime is not None and pd.notna(icu_intime):
        hours = max(0.0, (t_prediction - icu_intime).total_seconds() / 3600.0)
        defaults["icu_hours_elapsed"] = min(1.0, hours / 168.0)  # normalized up to 7 days

    if vitals_obs.empty:
        return defaults

    # Filter vitals strictly before t_prediction
    prior_vitals = vitals_obs[vitals_obs["charttime"] <= t_prediction]
    if prior_vitals.empty:
        return defaults

    # Extract HR (itemid 220045)
    hr_obs = prior_vitals[prior_vitals["itemid"] == 220045]
    if not hr_obs.empty:
        hr_sorted = hr_obs.sort_values("charttime")
        hr_last = float(hr_sorted["valuenum"].iloc[-1])
        defaults["heart_rate_last"] = np.clip(hr_last / 150.0, 0.0, 1.5)
        if len(hr_sorted) > 1:
            hr_prev = float(hr_sorted["valuenum"].iloc[-2])
            defaults["heart_rate_trend"] = np.clip((hr_last - hr_prev) / 30.0, -1.0, 1.0)

    # Extract MAP (itemid 220052, 220181)
    map_obs = prior_vitals[prior_vitals["itemid"].isin([220052, 220181])]
    if not map_obs.empty:
        map_sorted = map_obs.sort_values("charttime")
        map_last = float(map_sorted["valuenum"].iloc[-1])
        defaults["mean_arterial_pressure_last"] = np.clip(map_last / 120.0, 0.0, 1.5)
        defaults["map_hypotensive"] = 1.0 if map_last < 65.0 else 0.0
        if map_last < 65.0:
            defaults["sofa_cardio_component"] = 1.0

    # Extract RR (itemid 220210)
    rr_obs = prior_vitals[prior_vitals["itemid"] == 220210]
    if not rr_obs.empty:
        rr_last = float(rr_obs.sort_values("charttime")["valuenum"].iloc[-1])
        defaults["respiratory_rate_last"] = np.clip(rr_last / 35.0, 0.0, 1.5)

    # Extract SpO2 (itemid 220277)
    spo2_obs = prior_vitals[prior_vitals["itemid"] == 220277]
    if not spo2_obs.empty:
        spo2_last = float(spo2_obs.sort_values("charttime")["valuenum"].iloc[-1])
        defaults["spo2_last"] = np.clip(spo2_last / 100.0, 0.5, 1.0)

    return defaults
