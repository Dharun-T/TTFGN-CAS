"""Longitudinal Laboratory Trajectory Feature Extraction.

Extracts time-decayed dynamic trajectory features for Glucose, Creatinine,
Potassium, Sodium, and Hemoglobin strictly within the pre-prediction
lookback window [t_prediction - lookback_hours, t_prediction].
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_analyte_trajectory_features(
    analyte_obs: pd.DataFrame,
    t_prediction: pd.Timestamp,
    lookback_hours: float = 48.0
) -> Dict[str, float]:
    """Computes dynamic trajectory metrics for a single analyte prior to t_prediction.

    Input DataFrame MUST be strictly filtered by:
        charttime <= t_prediction AND charttime >= t_prediction - lookback_hours
    """
    if analyte_obs.empty:
        return {
            "current_prev_val": 0.0,
            "prev_val": 0.0,
            "delta": 0.0,
            "time_since_last_meas_hrs": lookback_hours,
            "rate_of_change": 0.0,
            "recent_mean": 0.0,
            "recent_std": 0.0,
            "recent_min": 0.0,
            "recent_max": 0.0,
            "meas_count": 0.0,
            "direction_of_change": 0.0,
            "trend_slope": 0.0,
            "has_measurement": 0.0,
        }

    # Sort chronologically
    sorted_obs = analyte_obs.sort_values("charttime")
    vals = sorted_obs["std_valuenum"].to_numpy(dtype=float)
    times = sorted_obs["charttime"]

    # Elapsed hours from each observation to t_prediction (strictly >= 0)
    elapsed_to_pred = np.array([
        (t_prediction - t).total_seconds() / 3600.0
        for t in times
    ])

    meas_count = len(vals)
    last_val = vals[-1]
    time_since_last = max(0.0, elapsed_to_pred[-1])

    if meas_count >= 2:
        penultimate_val = vals[-2]
        delta = last_val - penultimate_val
        time_diff = max(0.1, (times.iloc[-1] - times.iloc[-2]).total_seconds() / 3600.0)
        rate_of_change = delta / time_diff
        direction = 1.0 if delta > 0.05 else (-1.0 if delta < -0.05 else 0.0)

        # Trend slope via linear fit against time in hours
        t_hours = np.array([(t - times.iloc[0]).total_seconds() / 3600.0 for t in times])
        if np.std(t_hours) > 1e-4:
            slope = float(np.polyfit(t_hours, vals, 1)[0])
        else:
            slope = 0.0
    else:
        penultimate_val = last_val
        delta = 0.0
        rate_of_change = 0.0
        direction = 0.0
        slope = 0.0

    return {
        "current_prev_val": float(last_val),
        "prev_val": float(penultimate_val),
        "delta": float(delta),
        "time_since_last_meas_hrs": float(time_since_last),
        "rate_of_change": float(rate_of_change),
        "recent_mean": float(np.mean(vals)),
        "recent_std": float(np.std(vals)) if meas_count > 1 else 0.0,
        "recent_min": float(np.min(vals)),
        "recent_max": float(np.max(vals)),
        "meas_count": float(meas_count),
        "direction_of_change": float(direction),
        "trend_slope": float(slope),
        "has_measurement": 1.0,
    }
