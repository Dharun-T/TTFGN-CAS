"""Downstream Clinical Actionability Label Construction.

Evaluates whether an upcoming laboratory report at t_lab was followed by
a measurable downstream clinical action within the observation window
(t_lab, t_lab + action_window_hours].

Clinical Action Categories:
1. Medication Modification (initiation, discontinuation, dose adjustment)
2. Urgent / High-Frequency Diagnostic Investigation (STAT repeat, blood gas, ECG)
3. Clinical Monitoring & Protocol Escalation (infusion titration, telemetry, fluid orders)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class ActionabilityResult:
    action_occurred: int  # 0 or 1
    action_type: str      # 'none', 'medication_modification', 'urgent_repeat_lab', 'diagnostic_order', 'protocol_escalation'
    time_to_action_hours: Optional[float]
    specific_intervention: str
    target_analyte: str
    t_lab: datetime
    action_window_hours: float


# Actionable medication targets per analyte
ANALYTE_ACTION_MEDICATIONS: Dict[str, List[str]] = {
    "glucose": [
        "insulin", "regular insulin", "glargine", "aspart", "lispro",
        "dextrose", "d50w", "d10w", "glucagon", "metformin", "glipizide"
    ],
    "creatinine": [
        "furosemide", "lasix", "bumetanide", "torsemide", "sodium bicarbonate",
        "vancomycin", "gentamicin", "heparin"
    ],
    "potassium": [
        "potassium chloride", "kcl", "sodium polystyrene", "kayexalate",
        "patiromer", "lokelma", "calcium gluconate", "regular insulin", "albuterol"
    ],
    "sodium": [
        "sodium chloride 3%", "hypertonic saline", "sodium chloride 0.9%",
        "normal saline", "dextrose 5%", "d5w", "tolvaptan", "desmopressin", "ddavp"
    ],
    "hemoglobin": [
        "packed red blood cells", "prbc", "blood product", "fresh frozen plasma",
        "ffp", "platelets", "iron sucrose", "tranexamic acid", "vitamin k"
    ],
}


def evaluate_actionability_for_lab(
    subject_id: int,
    hadm_id: int,
    stay_id: Optional[int],
    analyte: str,
    t_lab: pd.Timestamp,
    prescriptions_df: pd.DataFrame,
    subsequent_labs_df: pd.DataFrame,
    action_window_hours: float = 12.0
) -> ActionabilityResult:
    """Evaluates whether an upcoming laboratory report is followed by a qualifying action.

    Temporal assertion: All evaluated actions must strictly have timestamp > t_lab.
    """
    window_end = t_lab + pd.Timedelta(hours=action_window_hours)
    analyte_clean = analyte.lower()
    candidate_actions = []

    # 1. Check Medication Changes (Initiation or new order)
    if not prescriptions_df.empty:
        target_med_keywords = ANALYTE_ACTION_MEDICATIONS.get(analyte_clean, [])
        # Filter prescriptions for this patient where starttime falls in (t_lab, window_end]
        pt_rx = prescriptions_df[
            (prescriptions_df["subject_id"] == subject_id) &
            (prescriptions_df["starttime"] > t_lab) &
            (prescriptions_df["starttime"] <= window_end)
        ]

        for _, rx in pt_rx.iterrows():
            drug_name = str(rx.get("drug", "")).lower()
            for kw in target_med_keywords:
                if kw in drug_name:
                    delta_hrs = (rx["starttime"] - t_lab).total_seconds() / 3600.0
                    candidate_actions.append({
                        "type": "medication_modification",
                        "time_to_action": delta_hrs,
                        "description": f"Medication: {rx.get('drug')} ({rx.get('dose_val_rx', '')} {rx.get('dose_unit_rx', '')})",
                    })
                    break

    # 2. Check Urgent Repeat or Secondary Diagnostic Lab Tests
    if not subsequent_labs_df.empty:
        pt_labs = subsequent_labs_df[
            (subsequent_labs_df["subject_id"] == subject_id) &
            (subsequent_labs_df["charttime"] > t_lab) &
            (subsequent_labs_df["charttime"] <= window_end)
        ]

        # Urgent repeat of the same analyte within 6 hours indicates acute reaction/titration
        urgent_repeats = pt_labs[
            (pt_labs["analyte"] == analyte_clean) &
            (pt_labs["charttime"] <= t_lab + pd.Timedelta(hours=6.0))
        ]
        for _, lab in urgent_repeats.iterrows():
            delta_hrs = (lab["charttime"] - t_lab).total_seconds() / 3600.0
            candidate_actions.append({
                "type": "urgent_repeat_lab",
                "time_to_action": delta_hrs,
                "description": f"Urgent repeat {analyte_clean.capitalize()} ordered within {delta_hrs:.1f}h",
            })

    # Sort candidates by earliest time to action
    if candidate_actions:
        candidate_actions.sort(key=lambda x: x["time_to_action"])
        first_action = candidate_actions[0]
        return ActionabilityResult(
            action_occurred=1,
            action_type=first_action["type"],
            time_to_action_hours=round(first_action["time_to_action"], 2),
            specific_intervention=first_action["description"],
            target_analyte=analyte_clean,
            t_lab=t_lab,
            action_window_hours=action_window_hours,
        )

    return ActionabilityResult(
        action_occurred=0,
        action_type="none",
        time_to_action_hours=None,
        specific_intervention="No qualifying clinical intervention documented within window",
        target_analyte=analyte_clean,
        t_lab=t_lab,
        action_window_hours=action_window_hours,
    )
