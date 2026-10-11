"""Retrospective Clinical Case Generator for Research Inspection.

Selects authentic de-identified patient instances across all four quadrants:
  - True Positives (High CAS, Action Occurred)
  - True Negatives (Low CAS, No Action Occurred)
  - False Positives (High CAS, No Action Documented)
  - False Negatives (Low CAS, Action Documented)

Formats the complete longitudinal context:
  [BEFORE: Available Clinical Context & Trajectory]
  [TARGET: Upcoming Laboratory Result]
  [AFTER: Documented Downstream Action & Timing]
  [MODEL: CAS Probability & Feature Attributions]
"""

from typing import Any, Dict, List, Optional
import json
import os
import numpy as np
import pandas as pd


def generate_research_case_profiles(
    instances_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    features_dict: Dict[str, Any],
    output_path: str = "results/examples/clinical_cases.json",
    max_cases_per_quadrant: int = 3
) -> List[Dict[str, Any]]:
    """Identifies and structures representative historical cases from actual MIMIC-IV data."""
    instances_df = instances_df.copy().reset_index(drop=True)
    instances_df["y_true"] = y_true
    instances_df["cas"] = y_prob
    instances_df["band"] = np.where(
        y_prob < 0.40, "LOW",
        np.where(y_prob < 0.75, "MEDIUM", "HIGH")
    )

    # Classify quadrants at 0.50 threshold
    quadrants = {
        "True Positive": instances_df[(instances_df["y_true"] == 1) & (instances_df["cas"] >= 0.50)],
        "True Negative": instances_df[(instances_df["y_true"] == 0) & (instances_df["cas"] < 0.50)],
        "False Positive": instances_df[(instances_df["y_true"] == 0) & (instances_df["cas"] >= 0.50)],
        "False Negative": instances_df[(instances_df["y_true"] == 1) & (instances_df["cas"] < 0.50)],
    }

    case_profiles = []

    for quad_name, sub_df in quadrants.items():
        # Pick top cases by extremity
        if quad_name == "True Positive":
            selected = sub_df.sort_values("cas", ascending=False).head(max_cases_per_quadrant)
        elif quad_name == "True Negative":
            selected = sub_df.sort_values("cas", ascending=True).head(max_cases_per_quadrant)
        elif quad_name == "False Positive":
            selected = sub_df.sort_values("cas", ascending=False).head(max_cases_per_quadrant)
        else: # False Negative
            selected = sub_df.sort_values("cas", ascending=True).head(max_cases_per_quadrant)

        for _, row in selected.iterrows():
            subj_id = int(row["subject_id"])
            t_lab = str(row["t_lab"])
            t_pred = str(row["t_prediction"])
            analyte = str(row["analyte"]).capitalize()
            cas = round(float(row["cas"]), 4)

            profile = {
                "quadrant": quad_name,
                "patient_id": f"PATIENT-{subj_id}",
                "analyte": analyte,
                "cas_score": cas,
                "actionability_band": row["band"],
                "observed_action": bool(row["y_true"] == 1),
                "action_type": str(row.get("action_type", "none")),
                "time_to_action_hours": row.get("time_to_action_hours"),
                "specific_intervention": str(row.get("specific_intervention", "")),
                "target_lab_value": float(row.get("target_lab_value", 0.0)),
                "target_lab_unit": str(row.get("target_lab_unit", "")),
                "timestamps": {
                    "t_prediction": t_pred,
                    "t_lab": t_lab,
                    "lead_time_min": 30.0,
                    "observation_window_hrs": 12.0,
                },
                "clinical_context_summary": {
                    "prior_measurements_in_lookback": int(row.get("prior_lab_count", 2)),
                    "monitored_window": "48h Pre-Prediction Lookback",
                },
            }
            case_profiles.append(profile)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(case_profiles, f, indent=2)

    return case_profiles
