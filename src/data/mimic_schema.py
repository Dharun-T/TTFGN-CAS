"""MIMIC-IV Data Schema and Leakage Safety Specification.

Defines the relational structures, datatypes, timestamp definitions,
and temporal leakage risk profiles for all MIMIC-IV tables utilized in T-TFGN.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    description: str
    is_timestamp: bool = False
    is_patient_id: bool = False
    is_stay_id: bool = False
    leakage_risk: str = "none"  # "none", "high_future_leakage", "target_outcome"
    handling_rule: str = "Permitted for feature computation if timestamp <= t_prediction"


@dataclass(frozen=True)
class TableSchema:
    name: str
    module: str  # "hosp" or "icu"
    primary_key: List[str]
    time_column: Optional[str]
    patient_id_col: str
    admission_id_col: Optional[str]
    stay_id_col: Optional[str]
    description: str
    columns: List[ColumnSchema] = field(default_factory=list)


MIMIC_IV_SCHEMA: Dict[str, TableSchema] = {
    "patients": TableSchema(
        name="patients",
        module="hosp",
        primary_key=["subject_id"],
        time_column=None,
        patient_id_col="subject_id",
        admission_id_col=None,
        stay_id_col=None,
        description="Patient demographic metadata, de-identified anchor age, and mortality",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("gender", "string", "Biological sex of patient (M/F)"),
            ColumnSchema("anchor_age", "int64", "Age during the anchor year"),
            ColumnSchema("anchor_year", "int64", "Shifted anchor year for de-identification"),
            ColumnSchema("anchor_year_group", "string", "Date range for anchor year"),
            ColumnSchema("dod", "string", "Date of death (if applicable)", is_timestamp=True, leakage_risk="high_future_leakage", handling_rule="Strictly excluded from input features"),
        ],
    ),
    "admissions": TableSchema(
        name="admissions",
        module="hosp",
        primary_key=["hadm_id"],
        time_column="admittime",
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col=None,
        description="Hospital admissions, admission/discharge timestamps, admission type",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Unique hospital admission identifier"),
            ColumnSchema("admittime", "datetime64[ns]", "Hospital admission timestamp", is_timestamp=True),
            ColumnSchema("dischtime", "datetime64[ns]", "Hospital discharge timestamp", is_timestamp=True, leakage_risk="high_future_leakage", handling_rule="Never use future discharge time"),
            ColumnSchema("deathtime", "datetime64[ns]", "In-hospital death timestamp", is_timestamp=True, leakage_risk="high_future_leakage", handling_rule="Never use future deathtime"),
            ColumnSchema("admission_type", "string", "Type of admission (Emergency, Urgent, Elective)"),
            ColumnSchema("admission_location", "string", "Physical location prior to admission"),
            ColumnSchema("discharge_location", "string", "Destination post-discharge", leakage_risk="high_future_leakage", handling_rule="Excluded (post-admission artifact)"),
            ColumnSchema("insurance", "string", "Payer category"),
            ColumnSchema("language", "string", "Preferred spoken language"),
            ColumnSchema("marital_status", "string", "Marital status"),
            ColumnSchema("race", "string", "Reported race/ethnicity"),
        ],
    ),
    "d_labitems": TableSchema(
        name="d_labitems",
        module="hosp",
        primary_key=["itemid"],
        time_column=None,
        patient_id_col="",
        admission_id_col=None,
        stay_id_col=None,
        description="Laboratory dictionary mapping item IDs to analyte labels and categories",
        columns=[
            ColumnSchema("itemid", "int64", "Unique laboratory test concept identifier"),
            ColumnSchema("label", "string", "Clinical description / analyte name"),
            ColumnSchema("fluid", "string", "Specimen body fluid (Blood, Urine, etc.)"),
            ColumnSchema("category", "string", "Lab department (Chemistry, Hematology, Blood Gas)"),
            ColumnSchema("loinc_code", "string", "Standard LOINC code"),
        ],
    ),
    "labevents": TableSchema(
        name="labevents",
        module="hosp",
        primary_key=["labevent_id"],
        time_column="charttime",
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col=None,
        description="Laboratory test observations with charttime, raw value, numeric value, and unit",
        columns=[
            ColumnSchema("labevent_id", "int64", "Unique laboratory event identifier"),
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Associated hospital admission identifier"),
            ColumnSchema("specimen_id", "int64", "Specimen identifier"),
            ColumnSchema("itemid", "int64", "Laboratory item code (linked to d_labitems)"),
            ColumnSchema("charttime", "datetime64[ns]", "Specimen collection / acquisition timestamp", is_timestamp=True),
            ColumnSchema("storetime", "datetime64[ns]", "Result availability in EHR timestamp", is_timestamp=True),
            ColumnSchema("value", "string", "Text or numeric result string"),
            ColumnSchema("valuenum", "float64", "Parsed numeric result value"),
            ColumnSchema("valueuom", "string", "Unit of measurement"),
            ColumnSchema("ref_range_lower", "float64", "Lower bound of institutional normal range"),
            ColumnSchema("ref_range_upper", "float64", "Upper bound of institutional normal range"),
            ColumnSchema("flag", "string", "Abnormality indicator (abnormal / normal)"),
            ColumnSchema("priority", "string", "Order priority (STAT / ROUTINE)"),
            ColumnSchema("comments", "string", "Technician and specimen comments"),
        ],
    ),
    "prescriptions": TableSchema(
        name="prescriptions",
        module="hosp",
        primary_key=["pharmacy_id"],
        time_column="starttime",
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col=None,
        description="Medication orders, administration schedules, doses, routes, and start/stop timestamps",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Associated hospital admission identifier"),
            ColumnSchema("pharmacy_id", "int64", "Unique prescription identifier"),
            ColumnSchema("starttime", "datetime64[ns]", "Medication order start timestamp", is_timestamp=True),
            ColumnSchema("stoptime", "datetime64[ns]", "Medication order stop timestamp", is_timestamp=True),
            ColumnSchema("drug_type", "string", "Classification of drug order (Main, Base, Additive)"),
            ColumnSchema("drug", "string", "Generic or brand medication name"),
            ColumnSchema("gsn", "string", "Generic Sequence Number"),
            ColumnSchema("ndc", "string", "National Drug Code"),
            ColumnSchema("prod_strength", "string", "Medication strength/concentration"),
            ColumnSchema("dose_val_rx", "string", "Ordered dose value"),
            ColumnSchema("dose_unit_rx", "string", "Ordered dose unit"),
            ColumnSchema("form_val_disp", "string", "Dispensed form value"),
            ColumnSchema("form_unit_disp", "string", "Dispensed form unit"),
            ColumnSchema("route", "string", "Route of administration (IV, PO, SC, INHALED)"),
        ],
    ),
    "diagnoses_icd": TableSchema(
        name="diagnoses_icd",
        module="hosp",
        primary_key=["subject_id", "hadm_id", "seq_num"],
        time_column=None,
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col=None,
        description="Billed ICD-9 / ICD-10 hospital discharge diagnoses",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Associated hospital admission identifier"),
            ColumnSchema("seq_num", "int64", "Priority order of diagnosis (1 = principal)"),
            ColumnSchema("icd_code", "string", "ICD-9 or ICD-10 diagnosis code"),
            ColumnSchema("icd_version", "int64", "ICD revision (9 or 10)"),
        ],
    ),
    "icustays": TableSchema(
        name="icustays",
        module="icu",
        primary_key=["stay_id"],
        time_column="intime",
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col="stay_id",
        description="Intensive care unit stays, admission/discharge times, care unit types, LOS",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Associated hospital admission identifier"),
            ColumnSchema("stay_id", "int64", "Unique ICU stay identifier", is_stay_id=True),
            ColumnSchema("first_careunit", "string", "Initial ICU careunit (MICU, SICU, CCU, TSICU, CVICU)"),
            ColumnSchema("last_careunit", "string", "Discharging ICU careunit"),
            ColumnSchema("intime", "datetime64[ns]", "ICU admission timestamp", is_timestamp=True),
            ColumnSchema("outtime", "datetime64[ns]", "ICU discharge timestamp", is_timestamp=True, leakage_risk="high_future_leakage", handling_rule="Strictly enforce t <= outtime during ICU stay tracking"),
            ColumnSchema("los", "float64", "Length of stay in ICU (fractional days)", leakage_risk="high_future_leakage", handling_rule="Derived from outtime; excluded from real-time features"),
        ],
    ),
    "chartevents": TableSchema(
        name="chartevents",
        module="icu",
        primary_key=["stay_id", "charttime", "itemid"],
        time_column="charttime",
        patient_id_col="subject_id",
        admission_id_col="hadm_id",
        stay_id_col="stay_id",
        description="High-frequency ICU bedside chart observations (vitals, ventilator, GCS)",
        columns=[
            ColumnSchema("subject_id", "int64", "Unique patient identifier", is_patient_id=True),
            ColumnSchema("hadm_id", "int64", "Associated hospital admission identifier"),
            ColumnSchema("stay_id", "int64", "Unique ICU stay identifier", is_stay_id=True),
            ColumnSchema("charttime", "datetime64[ns]", "Observation recording timestamp", is_timestamp=True),
            ColumnSchema("storetime", "datetime64[ns]", "Verification timestamp", is_timestamp=True),
            ColumnSchema("itemid", "int64", "Bedside observation concept ID"),
            ColumnSchema("value", "string", "Raw recorded bedside value"),
            ColumnSchema("valuenum", "float64", "Bedside numeric value"),
            ColumnSchema("valueuom", "string", "Measurement unit"),
            ColumnSchema("warning", "int64", "Nurse validation warning flag"),
        ],
    ),
}


def get_leakage_audit_summary() -> Dict[str, List[str]]:
    """Returns a breakdown of high-risk columns and handling policies."""
    leakage_risks = {}
    for table_name, schema in MIMIC_IV_SCHEMA.items():
        risky_cols = [
            f"{c.name} ({c.handling_rule})"
            for c in schema.columns
            if c.leakage_risk != "none"
        ]
        if risky_cols:
            leakage_risks[table_name] = risky_cols
    return leakage_risks
