# T-TFGN: Time-Decay Tensor Fusion Gated Network for Predicting Clinical Actionability of the Next Laboratory Report

**Authors:** Computational Intensive Care & Clinical Decision Science Group  
**Dataset:** MIMIC-IV Clinical Database v2.2 (Beth Israel Deaconess Medical Center / PhysioNet)  
**Target Prediction:** Downstream Clinical Actionability Score (CAS) $\in [0, 1]$  
**Codebase Repository:** Reproducible Research Platform  

---

## 1. Abstract
Serial laboratory investigations in the intensive care unit (ICU) are frequently performed on predetermined schedules. While serial monitoring tracks vital organ function, a significant proportion of test results produce no clinical consequence, whereas subtle laboratory fluctuations in a precarious clinical state provoke urgent therapeutic intervention. Traditional clinical prediction models forecast continuous future laboratory concentrations or classify binary reference-range abnormalities; however, neither paradigm establishes whether an upcoming result will alter clinical management. We propose **T-TFGN (Time-Decay Tensor Fusion Gated Network)**, a multimodal deep architecture designed to predict the **Clinical Actionability Score (CAS)** of an upcoming laboratory report prior to its availability. Formulated over 3,294 retrospective ICU prediction instances across five core analytes (Glucose, Creatinine, Potassium, Sodium, Hemoglobin), T-TFGN integrates: (1) analyte-specific continuous time decay ($\lambda_k$), (2) categorical analyte embeddings, (3) low-rank bilinear tensor fusion capturing cross-modal laboratory-context interactions, and (4) dynamic adaptive gating. On held-out patient-level test sets, multimodal clinical context substantially improves discrimination over laboratory-only trajectories, validating the hypothesis that laboratory dynamics alone are insufficient to estimate clinical actionability.

---

## 2. Introduction
Laboratory tests represent one of the highest-volume diagnostic modalities in modern intensive care medicine. Critical care teams rely on serial chemistry and hematology panels to detect acute physiological changes, titrate vasoactive infusions, manage glycemic control, and administer electrolyte repletion. Despite their diagnostic indispensability, excessive and non-actionable laboratory testing contributes to iatrogenic anemia, diagnostic fatigue, workflow interruptions, and unnecessary healthcare expenditure.

A fundamental barrier to intelligent laboratory stewardship is the absence of models predicting **clinical actionability**. When an upcoming laboratory order is pending, clinicians currently have no automated method to quantify how likely that upcoming result is to yield an immediate, measurable clinical intervention.

---

## 3. Problem Statement
The clinical actionability prediction problem is formulated as follows:

$$\text{Given everything known about a patient BEFORE an upcoming laboratory report, how likely is that report to be followed by a measurable downstream clinical action?}$$

Crucially, the system does **not** predict:
- Whether the laboratory value will simply increase or decrease.
- Whether the laboratory value will fall outside statistical normal ranges.
- Whether an arbitrary disease entity is present.
- Whether a patient will deteriorate over multi-day horizons.
- Whether an automated test cancellation should be enforced.

The primary prediction target is the **downstream clinical actionability** of the upcoming report.

---

## 4. Research Gap
Traditional machine learning approaches in hospital laboratory analytics fall into three primary historical paradigms:
1. **Laboratory Value Forecasting:** Autoregressive time series models (ARIMA, LSTM, GRU) forecasting numerical value $y_{t+1}$.
2. **Abnormality Detection:** Classifiers predicting whether $y_{t+1} < \text{LLN}$ or $y_{t+1} > \text{ULN}$.
3. **Global Deterioration Predictors:** Sepsis, shock, or mortality models operating on non-specific clinical endpoints.

The scientific gap addressed by T-TFGN is that:
- **Abnormal $\ne$ Actionable:** Persistently abnormal lab values in patients with chronic baseline disease (e.g. chronic kidney disease with creatinine $2.8\text{ mg/dL}$) often generate zero acute management changes.
- **Normal $\ne$ Non-Actionable:** A value falling within reference bounds can prompt immediate cessation of an active infusion (e.g. normalizing potassium prompting discontinuation of continuous IV potassium).
- **Context Dictates Action:** The clinical significance of a laboratory trend is mediated by active medications, organ failure severity, and hemodynamic instability.

---

## 5. Research Question
Does the integration of continuous time decay, cross-modal bilinear tensor fusion, and adaptive clinical context gating improve the prediction of laboratory actionability compared to classical machine learning models and laboratory-only trajectories?

---

## 6. Dataset
Experiments were conducted using the **MIMIC-IV Clinical Database v2.2**, comprising de-identified electronic health records for patients admitted to the ICU at Beth Israel Deaconess Medical Center between 2008 and 2019. Tables utilized include:
- `hosp.patients`: Baseline demographics and anchor age.
- `hosp.admissions`: Hospitalization trajectory and admission types.
- `hosp.d_labitems`: Laboratory analyte dictionary and LOINC codes.
- `hosp.labevents`: Serial laboratory observations and standard units.
- `hosp.prescriptions`: Longitudinal medication orders and administration timing.
- `hosp.diagnoses_icd`: Billed ICD-9 and ICD-10 diagnostic classifications.
- `icu.icustays`: ICU admission timestamps and lengths of stay.
- `icu.chartevents`: Bedside physiological vitals (HR, MAP, RR, SpO2).

---

## 7. Cohort Construction
The cohort was established under strict clinical eligibility criteria:
1. **Adult ICU Patients:** Age $\ge 18$ years at anchor year.
2. **ICU Length of Stay:** Verified ICU duration $\ge 24$ hours.
3. **Serial Monitoring Density:** At least 2 prior observations for the target analyte within the 48-hour pre-prediction lookback window.

---

## 8. Data Preprocessing & Unit Standardization
All laboratory observations for the five target analytes were normalized to canonical international clinical units:
- **Glucose:** $\text{mg/dL}$ (conversion: $\text{mmol/L} \times 18.0182$).
- **Creatinine:** $\text{mg/dL}$ (conversion: $\mu\text{mol/L} \times 0.011312$).
- **Potassium:** $\text{mEq/L}$ (canonical $\equiv \text{mmol/L}$).
- **Sodium:** $\text{mEq/L}$ (canonical $\equiv \text{mmol/L}$).
- **Hemoglobin:** $\text{g/dL}$ (conversion: $\text{g/L} \times 0.1$, $\text{mmol/L} \times 1.611$).

Physiological boundary checks were enforced; values outside plausible adult ICU limits (e.g. potassium $< 1.0$ or $> 12.0\text{ mEq/L}$) were flagged and excluded.

---

## 9. Actionability Label Construction
For each eligible upcoming laboratory result at timestamp $t_{\text{lab}}$, the model prediction timestamp was fixed strictly at $t_{\text{prediction}} = t_{\text{lab}} - 30\text{ minutes}$.

Downstream clinical actionability ($Y \in \{0, 1\}$) was evaluated over the subsequent observation window $(t_{\text{lab}}, t_{\text{lab}} + 12\text{ hours}]$:
- **Medication Modifications:** Initiation or dose adjustment of pharmacological agents biologically targeted to the analyte.
- **Urgent Re-evaluations:** Unscheduled repeat testing of the analyte within 6 hours.
- **Clinical Monitoring & Orders:** Bedside intervention orders documented in EHR tables.

---

## 10. Feature Engineering
Feature extraction was restricted exclusively to events with $\text{timestamp} \le t_{\text{prediction}}$:
1. **Longitudinal Lab Features (12 dims):** Most recent prior value, penultimate value, delta, time elapsed since last measurement, rate of change, lookback mean, standard deviation, minimum, maximum, observation count, change direction, trend slope.
2. **Patient Demographics (2 dims):** Normalized anchor age, biological sex.
3. **Active Medication Context (8 dims):** Active pre-prediction exposure to insulins, loop diuretics, vasopressors, anticoagulants, antihypertensives, potassium supplements, antibiotics, and active medication count.
4. **Chronic Comorbidity Context (7 dims):** Documented history of diabetes, renal disease, heart failure, hypertension, respiratory failure, sepsis, and liver disease.
5. **Bedside Vital Signs & Severity (10 dims):** Most recent heart rate, heart rate trend, mean arterial pressure, hypotension flag (MAP $< 65\text{ mmHg}$), respiratory rate, SpO2, elapsed ICU hours, cardiovascular and renal SOFA organ dysfunction components.

---

## 11. T-TFGN Architecture
T-TFGN consists of five interconnected neural modules:
1. **Analyte-Specific Time-Decay:** $\lambda_k = \sigma(\gamma_k \cdot \Delta t_k + 1.0)$, parameterizing the rate at which historical measurements lose predictive relevance.
2. **Analyte Embedding:** Dense 16-dimensional embedding projecting analyte categorical identity into latent space.
3. **Laboratory & Context Encoders:** Multi-layer perceptrons with Layer Normalization and Dropout ($p = 0.2$).
4. **Low-Rank Bilinear Tensor Fusion:** Rank-16 factorized bilinear interaction matrix:
   $$\mathbf{h}_{\text{tensor}} = \sum_{r=1}^{16} (\mathbf{U}_r \mathbf{h}_{\text{lab}}) \odot (\mathbf{V}_r \mathbf{h}_{\text{context}}) + \mathbf{b}$$
5. **Adaptive Multimodal Gate:** Dynamic gate balancing acute lab volatility against chronic context:
   $$\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{lab}}, \mathbf{h}_{\text{context}}, \mathbf{h}_{\text{tensor}}] + \mathbf{b}_g\right)$$
   $$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{W}_l \mathbf{h}_{\text{lab}} + (1 - \mathbf{g}) \odot \mathbf{W}_c \mathbf{h}_{\text{context}} + \mathbf{W}_t \mathbf{h}_{\text{tensor}}$$
6. **Prediction Head:** Computes the Clinical Actionability Score $\text{CAS} = \sigma(z) \in [0, 1]$.

---

## 12. Baseline Models
The proposed model was benchmarked against four competitive baselines:
1. **Laboratory-Only Baseline (Random Forest):** Evaluated strictly on the 12 laboratory trajectory features.
2. **Logistic Regression (L2 regularized):** Classical linear baseline with isotonic calibration.
3. **Random Forest Classifier:** Ensembled decision trees (150 estimators, max depth 8).
4. **HistGradientBoosting (XGBoost comparator):** Boosted decision trees with histogram binning.

---

## 13. Training Procedure
- **Disjoint Partitioning:** 70% Train, 15% Validation, 15% Test partitioned strictly at the patient level (`subject_id`).
- **Optimization:** AdamW optimizer ($\text{lr} = 0.001$, $\text{weight decay} = 10^{-4}$), ReduceLROnPlateau scheduling.
- **Early Stopping:** Monitored validation PR-AUC with patience of 8 epochs.
- **Reproducibility:** Seed 42 fixed across NumPy, Python, PyTorch; hardware and environment provenance recorded in `models/t_tfgn_provenance.json`.

---

## 14. Experimental Results & Discussion
Primary experimental findings across the held-out test cohort:
- **Baseline vs. Multimodal Utility:** The Laboratory-Only comparator achieved lower performance than multimodal models, confirming that laboratory values in isolation fail to capture clinical decision-making.
- **Ablation Validation:** Systematically ablating time decay, tensor fusion, or adaptive gating produced measurable drops in discriminatory power, establishing the necessity of each architectural component.
- **Calibration Analysis:** CAS scores showed close alignment between predicted probability and observed action frequencies across uniform decile bins.

---

## 15. Conclusion
T-TFGN demonstrates that downstream clinical actionability of upcoming laboratory reports is predictable prior to test availability using longitudinal trajectories coupled with multimodal clinical context. By shifting the objective from value forecasting to actionability prediction, the platform establishes an empirical foundation for context-aware laboratory stewardship.
