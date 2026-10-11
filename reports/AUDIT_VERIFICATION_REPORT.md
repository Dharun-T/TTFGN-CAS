# Independent Scientific & Empirical Audit Verification Report: T-TFGN Evaluation

**Date of Audit**: October 11, 2026  
**Auditor**: Independent Scientific Review  
**Subject**: Time-Decay Tensor Fusion Gated Network (T-TFGN) Clinical Actionability System  
**Primary Dataset**: MIMIC-IV Clinical Extract (Adult ICU Cohort)  
**Execution Environment**: Linux / Node.js 22 Runtime / Python 3.10.12 (Zero-dependency Standard Library Verification Engine)

---

## Executive Summary

An independent, end-to-end verification of the T-TFGN codebase, raw evaluation predictions (`results/predictions/test_predictions.npz`), dataset extraction files (`data/extracted/*.csv`), feature matrices (`data/features/features_dataset.npz`), model architectures, and frontend presentations was conducted.

### Critical Top-Line Findings:
1. **Mathematical Consistency of Real Test Predictions**: All reported metrics from the model evaluation step (`results/tables/final_model_comparison.csv`) agree mathematically with raw test predictions on held-out test data ($N = 504$ instances, $73$ positives, $431$ negatives).
2. **Prior Frontend Discrepancy & Overclaiming**: The prior React frontend (`src/App.tsx` and `src/components/ROCThresholdScrubber.tsx`) displayed hardcoded, fabricated metrics for T-TFGN (claiming ROC-AUC 0.5842 and PR-AUC 0.2418). Independent reproduction confirms the true evaluated model achieved **ROC-AUC of 0.5034** and **PR-AUC of 0.1616**.
3. **Research Baseline Underperformance**: The proposed T-TFGN neural network **does not outperform** classical baselines on this dataset. A simple unimodal **Random Forest (Lab-Only)** achieved higher discriminatory ability (**ROC-AUC 0.5648**, **PR-AUC 0.1960**).
4. **Verified Flaw in Ablation Code**: In `src/training/ablations.py`, Experiment B ("Clinical-Context-Only") and Experiment C ("Standard Lab + Context MLP") produced identical metrics because feature modalities were never zeroed/masked out during data preparation.
5. **Timeline and Attribution Ambiguities**: 79.0% of all labeled downstream actions (362 of 458) were categorized as `urgent_repeat_lab` within 6 hours. In intensive care practice, scheduled repeat labs (e.g., Q4H/Q6H glucose checks or morning routine panels) frequently co-occur without representing reactive acute physician interventions to the previous specific report.

---

## 1. Verified Facts

The following facts are verified by direct file inspection, schema parsing, and programmatic execution:

### 1.1 Dataset Inventory & Storage
All 8 relational clinical tables reside in `data/extracted/` with the following confirmed row counts, column lists, and schemas:

| Table File | Row Count | Column Count | Columns Present |
| :--- | :---: | :---: | :--- |
| `patients.csv` | 60 | 6 | `subject_id`, `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, `dod` |
| `admissions.csv` | 60 | 12 | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime`, `admission_type`, `admission_location`, `discharge_location`, `insurance`, `language`, `marital_status`, `race` |
| `icustays.csv` | 60 | 8 | `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `last_careunit`, `intime`, `outtime`, `los` |
| `d_labitems.csv` | 13 | 5 | `itemid`, `label`, `fluid`, `category`, `loinc_code` |
| `labevents.csv` | 3,916 | 9 | `labevent_id`, `subject_id`, `hadm_id`, `itemid`, `charttime`, `value`, `valuenum`, `valueuom`, `flag` |
| `prescriptions.csv` | 169 | 9 | `subject_id`, `hadm_id`, `pharmacy_id`, `starttime`, `stoptime`, `drug`, `dose_val_rx`, `dose_unit_rx`, `route` |
| `diagnoses_icd.csv` | 100 | 5 | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` |
| `chartevents.csv` | 7,476 | 7 | `subject_id`, `hadm_id`, `stay_id`, `charttime`, `itemid`, `valuenum`, `valueuom` |

### 1.2 Cohort Filtering Stages
Re-executing the cohort selection pipeline on raw source tables confirms:
- **Stage 0 (Raw Intake)**: 60 patients, 60 admissions, 60 ICU stays.
- **Stage 1 (Inner Relational Merge)**: All 60 ICU stays link to valid `patients` and `admissions`.
- **Stage 2 (Adult Age Filter $\ge 18$)**: All 60 patients satisfy `anchor_age >= 18` (range: 24 to 89 years). 0 excluded.
- **Stage 3 (ICU Length of Stay $\ge 24.0\text{h}$)**: All 60 stays satisfy `los * 24.0 >= 24.0` (minimum LOS is 1.12 days). 0 excluded.
- **Final Adult ICU Cohort (`data/interim/cohort.csv`)**: Exactly 60 stays across 60 unique patients.

### 1.3 Target Prediction Instances
Constructing prediction instances with 48h lookback, 30m lead time, minimum 2 prior measurements, and 12h action window yields:
- **Total Prediction Instances**: 3,294 instances across 60 unique patients and 60 hospital admissions.
- **Analyte Breakdown**:
  - Creatinine: 676 instances
  - Potassium: 664 instances
  - Hemoglobin: 664 instances
  - Sodium: 659 instances
  - Glucose: 631 instances
- **Label Breakdown**:
  - Negative ($y = 0$, no qualifying action): 2,836 (86.10%)
  - Positive ($y = 1$, qualifying action): 458 (13.90%)
- **Positive Action Types**:
  - `urgent_repeat_lab`: 362 (79.04% of positive actions)
  - `medication_modification`: 96 (20.96% of positive actions)

### 1.4 Patient-Level Partitioning & Leakage Safeguard
- The dataset was partitioned by patient (`subject_id`) using a 70/15/15 target split:
  - **Train**: 42 patients (70.0% of cohort)
  - **Validation**: 9 patients (15.0% of cohort)
  - **Test**: 9 patients (15.0% of cohort)
- **Leakage Verification**:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping patients)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping patients)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping patients)
  - In `test_predictions.npz`, `test_indices` contains exactly 504 instances belonging exclusively to the 9 test patients: `[10000001, 10000004, 10000006, 10000009, 10000013, 10000018, 10000046, 10000051, 10000055]`.

---

## 2. Reproduced Experimental Results

Direct recomputation from the binary arrays `y_test.npy`, `cas_test.npy`, and `test_indices.npy` in `results/predictions/test_predictions.npz` confirms mathematical reproducibility down to the 4th decimal place.

### 2.1 Recomputed Test Confusion Matrix ($\tau = 0.50$)

$$\begin{array}{c|cc|c}
& \hat{y} = 1 & \hat{y} = 0 & \text{Total} \\
\hline
y = 1 & \text{TP} = 21 & \text{FN} = 52 & 73 \\
y = 0 & \text{FP} = 123 & \text{TN} = 308 & 431 \\
\hline
\text{Total} & 144 & 360 & N = 504
\end{array}$$

### 2.2 Recomputed Mathematical Metric Proofs

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{21 + 308}{504} = \frac{329}{504} = 0.652777\dots \implies \mathbf{0.6528}$$

$$\text{Precision (PPV)} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{21}{21 + 123} = \frac{21}{144} = 0.145833\dots \implies \mathbf{0.1458}$$

$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{21}{21 + 52} = \frac{21}{73} = 0.287671\dots \implies \mathbf{0.2877}$$

$$\text{Specificity (TNR)} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{308}{308 + 123} = \frac{308}{431} = 0.714617\dots \implies \mathbf{0.7146}$$

$$\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot 21}{144 + 73} = \frac{42}{217} = 0.193548\dots \implies \mathbf{0.1935}$$

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2 = \mathbf{0.2376}$$

$$\text{Expected Calibration Error (ECE, 10 Bins)} = \sum_{b=1}^{10} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)| = \mathbf{0.3303}$$

$$\text{ROC-AUC (Mann-Whitney U Rank Sum)} = \frac{U}{N_{\text{pos}} \cdot N_{\text{neg}}} = \frac{15814.0}{73 \cdot 431} = \mathbf{0.5034}$$

$$\text{PR-AUC (Average Precision, Integral of Step Curve)} = \sum_k (R_k - R_{k-1}) P_k = \mathbf{0.1616}$$

All recomputed metrics match `results/tables/final_model_comparison.csv` **identically**.

### 2.3 Master Benchmark Comparison Table

| Model | Feature Set | ROC-AUC [95% CI] | PR-AUC [95% CI] | F1-Score | Brier | ECE | Accuracy | Recall | Precision | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Lab-Only)** | Trajectory Only | **0.5648** [0.494-0.636] | **0.1960** [0.141-0.272] | 0.1404 | 0.1783 | 0.2156 | 0.8056 | 0.1096 | 0.1951 | **REPRODUCED** |
| **Gradient Boosting** | Lab + Context | 0.5424 [0.465-0.617] | 0.1833 [0.136-0.263] | 0.0000 | 0.1241 | 0.0192 | 0.8492 | 0.0000 | 0.0000 | **REPRODUCED** |
| **Random Forest** | Lab + Context | 0.5288 [0.454-0.598] | 0.1753 [0.133-0.251] | 0.0488 | 0.1265 | 0.0248 | 0.8452 | 0.0274 | 0.2222 | **REPRODUCED** |
| **Logistic Regression** | Lab + Context | 0.5280 [0.448-0.604] | 0.1534 [0.120-0.207] | 0.0000 | 0.1276 | 0.0237 | 0.8532 | 0.0000 | 0.0000 | **REPRODUCED** |
| **T-TFGN (Proposed)** | Multimodal Tensor | **0.5034** [0.435-0.581] | **0.1616** [0.124-0.230] | 0.1935 | 0.2376 | 0.3303 | 0.6528 | 0.2877 | 0.1458 | **REPRODUCED** |

---

## 3. Corrected Implementation Details

During our audit, three major implementation and reporting bugs were uncovered and corrected:

### 3.1 Correction of Frontend Metric Fabrication
- **Issue**: `src/App.tsx` and `src/components/ROCThresholdScrubber.tsx` previously hardcoded overclaimed performance values for T-TFGN:
  - Hardcoded ROC-AUC: `0.5842` (True: `0.5034`, overstated by $+0.0808$).
  - Hardcoded PR-AUC: `0.2418` (True: `0.1616`, overstated by $+0.0802$).
  - Hardcoded Accuracy: `0.7812` (True: `0.6528`, overstated by $+0.1284$).
  - Hardcoded ECE: `0.1142` (True: `0.3303`, understated by $-0.2161$).
  - Hardcoded False Positives: `15` (True: `123`, understated by $-108$).
- **Correction Applied**:
  - Exported canonical metrics JSON (`results/metrics/canonical_test_metrics.json` and `src/data/canonical_test_metrics.json`).
  - Modified `src/App.tsx` and `ROCThresholdScrubber.tsx` to read directly from the verified canonical data.
  - Added an Audited Scientific Integrity banner in the application dashboard alerting users to verified metrics.

### 3.2 Correction of Ablation Modality Masking Bug
- **Issue**: In `src/training/ablations.py`, Experiment B ("Clinical-Context-Only") and Experiment C ("Standard Lab + Context MLP") yielded identical metrics (ROC-AUC `0.5392`, PR-AUC `0.1831`, F1 `0.1772`). Investigation revealed that Exp_B claimed to isolate clinical context, but passed all laboratory features without zeroing them.
- **Correction Applied**: Updated `src/training/ablations.py` so that:
  - `Exp_A` explicitly zeroes `context_features`.
  - `Exp_B` explicitly zeroes `lab_features` and `elapsed_hours`.
  - `Exp_C` evaluates standard concatenation without zeroing.

### 3.3 Probability Calibration Asymmetry
- **Issue**: Classical baseline classifiers (`BaselineClassifierSuite` in `src/models/baselines/classical.py`) used `CalibratedClassifierCV(method="isotonic", cv=3)`, resulting in low ECE ($0.019 - 0.025$). T-TFGN in `src/models/tfgn/t_tfgn.py` only applied raw `torch.sigmoid(logits)` with no temperature scaling, Platt scaling, or isotonic regression, resulting in an uncalibrated ECE of $0.3303$.
- **Finding**: The apparent calibration deficit of T-TFGN is partially an artifact of post-hoc calibration applied to baselines but omitted from the neural network.

---

## 4. Unverified Claims

The following claims from previous reports could **not** be verified or were proven incorrect:

1. **Claim: "T-TFGN significantly outperforms all classical baselines"**:  
   **INCORRECT**. The unimodal Random Forest baseline (Lab-Only) achieved ROC-AUC 0.5648 and PR-AUC 0.1960, whereas T-TFGN achieved ROC-AUC 0.5034 and PR-AUC 0.1616. T-TFGN essentially performs near random guessing on discrimination in this dataset.
2. **Claim: "Exp_B and Exp_C demonstrate separate architectural effects"**:  
   **INCORRECT**. In the previous run, Exp_B and Exp_C produced identical weights and predictions due to identical unmasked inputs.
3. **Claim: "The model achieves 78.1% accuracy and 15 false positives at standard cutoff"**:  
   **INCORRECT**. The actual confusion matrix at $\tau = 0.50$ has 123 false positives and 65.3% accuracy.

---

## 5. Timeline, Attribution & Dataset Limitations

### 5.1 Temporal Ambiguity in Specimen Timestamps
- MIMIC-IV `labevents.csv` contains `charttime`, which marks specimen collection/charting.
- Laboratory turnaround time (`storetime` / result verification time) is **absent** from the extracted data.
- The pipeline defines $t_{\text{prediction}} = t_{\text{lab}} - 30\text{ minutes}$. This implies that the prediction cutoff occurs before the blood draw. In clinical practice, a model predicting whether an upcoming lab will prompt an action must differentiate between pre-draw prediction and post-result decision support.

### 5.2 Clinical Action Attribution Ambiguity
- Of 458 positive labels, 362 (79.0%) are attributed to an `urgent_repeat_lab` within 6 hours.
- In ICU clinical workflows, patients on intravenous insulin, vasopressors, or standard critical care protocols have standing orders for repeat blood draws every 4 to 6 hours regardless of the prior lab's status.
- Attributing scheduled monitoring blood draws to "actionability" introduces significant label noise.

### 5.3 Hospital-Wide Labs Lacking `stay_id`
- In MIMIC-IV, laboratory events are recorded hospital-wide and do not index `stay_id`.
- Consequently, all `stay_id` values in `prediction_instances.csv` are blank (`""`). ICU-specific context must be joined strictly via admission (`hadm_id`) and timestamp intervals.

---

## 6. Execution Commands & Verification Test Log

The entire verification pipeline was executed without third-party scientific library dependencies using standard Python:

### Command 1: Mathematical Recomputation & Canonical Export
```bash
python3 scripts/verify_audit.py
```
- **Outcome**: Exited with code 0.
- **Results**: Parsed `test_predictions.npz`, recomputed all 10 metrics, verified mathematical consistency, validated zero patient leakage, and generated `results/metrics/canonical_test_metrics.json`.

### Command 2: Automated Unit Test Suite
```bash
python3 tests/test_audit_verification.py
```
- **Outcome**: `Ran 6 tests in 0.119s - OK`.
- **Tests Passed**:
  - `test_canonical_file_exists`: PASS
  - `test_metric_mathematical_consistency`: PASS
  - `test_patient_leakage_absence`: PASS
  - `test_temporal_lead_time_integrity`: PASS
  - `test_downstream_action_timing`: PASS
  - `test_label_generation_distribution`: PASS

### Command 3: Frontend TypeScript Lint & Compilation
```bash
npm run lint && npm run build
```
- **Outcome**: Exited with code 0. Build succeeded, zero TypeScript errors.

---

## Conclusion & Scientific Recommendation

The project code and evaluation data are structurally sound, well-partitioned, and free of patient data leakage. However, the proposed T-TFGN model does not support superior clinical actionability claims over simpler baselines on this dataset. 

All hardcoded inflations in the user interface have been corrected. The system is now 100% synchronized with the actual empirical results. It should be presented as an investigational multimodal modeling architecture requiring larger cohorts, calibrated post-hoc inference, and more granular medication order timestamps before any clinical claims can be made.
