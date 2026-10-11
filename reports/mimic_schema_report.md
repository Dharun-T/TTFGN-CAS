# MIMIC-IV Clinical Database Relational Schema & Temporal Leakage Audit

**Dataset Version:** MIMIC-IV v2.2 (Hospital & ICU Modules)  
**Research Target:** T-TFGN (Time-Decay Tensor Fusion Gated Network for Clinical Actionability)  
**Prediction Cutoff:** $t_{\text{prediction}} \le t_{\text{lab}} - 30\text{ min}$ (Strict pre-result availability)  

---

## 1. Relational Table Inventory

The following core MIMIC-IV tables form the foundation of our analytical pipeline:

| Module | Table | Primary Key | Time Field | Patient ID | Stay ID | Primary Analytical Utility |
|---|---|---|---|---|---|---|
| **hosp** | `patients` | `subject_id` | N/A | `subject_id` | N/A | Anchor age, gender, baseline demographics |
| **hosp** | `admissions` | `hadm_id` | `admittime` | `subject_id` | N/A | Admission timing, type, hospital trajectory |
| **hosp** | `d_labitems` | `itemid` | N/A | N/A | N/A | Analyte vocabulary, fluid, category, LOINC mapping |
| **hosp** | `labevents` | `labevent_id` | `charttime` | `subject_id` | N/A | Historical lab trajectory ($t \le t_{pred}$) and target reports ($t_{lab}$) |
| **hosp** | `prescriptions`| `pharmacy_id` | `starttime` | `subject_id` | N/A | Pre-test medication exposure & post-test therapeutic changes |
| **hosp** | `diagnoses_icd`| `(hadm_id, seq_num)` | N/A | `subject_id` | N/A | Documented clinical conditions and organ dysfunction |
| **icu** | `icustays` | `stay_id` | `intime` | `subject_id` | `stay_id` | ICU admission window, care unit type, ICU stay alignment |
| **icu** | `chartevents` | `(stay_id, charttime, itemid)` | `charttime` | `subject_id` | `stay_id` | Bedside physiological vitals (HR, MAP, RR, SpO2, Temp) |

---

## 2. Table-by-Table Leakage Risk Assessment

### A. `hosp.patients`
- **Columns:** `subject_id`, `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, `dod`
- **Leakage Risk Analysis:**
  - `anchor_age` and `gender` are safe static baseline covariates.
  - `dod` (Date of Death) represents future outcome data. **Strict Policy:** `dod` is excluded from all feature generation matrices to prevent fatal survivorship leakage.

### B. `hosp.admissions`
- **Columns:** `subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime`, `admission_type`, `admission_location`, `discharge_location`, `insurance`, `race`
- **Leakage Risk Analysis:**
  - `admittime` is safe if $t_{\text{admit}} \le t_{\text{prediction}}$.
  - `dischtime`, `deathtime`, and `discharge_location` are recorded at the termination of hospital stay. **Strict Policy:** Excluded from real-time feature extraction.

### C. `hosp.labevents`
- **Columns:** `labevent_id`, `subject_id`, `hadm_id`, `specimen_id`, `itemid`, `charttime`, `storetime`, `value`, `valuenum`, `valueuom`, `flag`, `priority`
- **Leakage Risk Analysis:**
  - Any observation with $\text{charttime} > t_{\text{prediction}}$ is strictly forbidden in input features.
  - The upcoming target laboratory report ($t_{\text{lab}}$) and its numerical value are held out. Only its identity (analyte type) is passed as the target query to T-TFGN.
  - Lookback history features are computed exclusively over the interval $[t_{\text{prediction}} - 48\text{h}, t_{\text{prediction}}]$.

### D. `hosp.prescriptions`
- **Columns:** `subject_id`, `hadm_id`, `pharmacy_id`, `starttime`, `stoptime`, `drug`, `dose_val_rx`, `route`
- **Leakage Risk Analysis:**
  - Prescriptions active before $t_{\text{prediction}}$ ($\text{starttime} \le t_{\text{prediction}}$ and $\text{stoptime} \ge t_{\text{prediction}}$) serve as active medication context.
  - Prescriptions initiated or modified strictly in the post-result window $(t_{\text{lab}}, t_{\text{lab}} + \text{action\_window}]$ represent **downstream action outcomes (labels)**. They must never enter feature vectors.

### E. `icu.icustays`
- **Columns:** `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `intime`, `outtime`, `los`
- **Leakage Risk Analysis:**
  - `los` (Total ICU length of stay) is derived from `outtime` and encodes hospital duration. **Strict Policy:** Excluded from features; elapsed ICU hours prior to $t_{\text{prediction}}$ is computed dynamically ($t_{\text{prediction}} - \text{intime}$).

### F. `icu.chartevents`
- **Columns:** `stay_id`, `charttime`, `itemid`, `valuenum`, `valueuom`
- **Leakage Risk Analysis:**
  - Bedside vital signs are filtered strictly by $\text{charttime} \le t_{\text{prediction}}$. Future vitals are ignored.

---

## 3. Automated Temporal Leakage Invariant Guarantees

Our pipeline enforces four mathematical assertions prior to model ingestion:
$$\forall f_i \in \text{Features}, \quad \text{timestamp}(f_i) \le t_{\text{prediction}}$$
$$t_{\text{prediction}} \le t_{\text{lab}} - \Delta t_{\text{lead}} \quad (\Delta t_{\text{lead}} \ge 30\text{ min})$$
$$\forall a_j \in \text{Downstream Actions}, \quad \text{timestamp}(a_j) > t_{\text{lab}}$$
$$\text{Patients}(\mathcal{D}_{\text{train}}) \cap \text{Patients}(\mathcal{D}_{\text{val}}) \cap \text{Patients}(\mathcal{D}_{\text{test}}) = \emptyset$$
