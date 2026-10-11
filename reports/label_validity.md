# Label Validity and Temporal Causality Considerations

**Domain:** Retrospective EHR Data Science  
**Target:** Downstream Clinical Actionability within $(t_{\text{lab}}, t_{\text{lab}} + 12\text{h}]$  
**Data Source:** MIMIC-IV Clinical Database v2.2  

---

## 1. Definitional Framework of the Actionability Label

In this investigation, a laboratory report is classified as **actionable** ($Y = 1$) if it is followed by at least one qualifying clinical action documented in the EHR within the pre-specified post-result window $(t_{\text{lab}}, t_{\text{lab}} + \Delta t_{\text{action}}]$:

1. **Targeted Medication Modifications:** Initiation, dose adjustment, route switch, or discontinuation of pharmacological agents biologically linked to the analyte (e.g. insulin for glucose; potassium chloride, kayexalate, or calcium for potassium; loop diuretics or sodium bicarbonate for creatinine/renal function).
2. **Urgent Diagnostic Re-evaluations:** Unscheduled repeat testing of the analyte within 6 hours, reflecting urgent titration or verification of acute volatility.
3. **Bedside Escalations & Orders:** Documented clinical protocol adjustments.

---

## 2. Fundamental Temporal Causality Limitation

> **Critical Epistemic Boundary:** Retrospective observational EHR records reflect **temporal association**, NOT proven physiological causation.

The system does NOT claim that the laboratory result strictly caused the clinician to administer the medication. In routine intensive care:
- Clinicians frequently anticipate laboratory findings based on bedside clinical examination and initiate treatment concurrently.
- Laboratory specimens require processing lead times (collection $\to$ transport $\to$ analyzer $\to$ EHR result posting). A medication order recorded 45 minutes after result posting may have been planned prior to result availability.
- A clinical intervention may be triggered by a co-occurring physiological disturbance rather than the isolated analyte under consideration.

Consequently, all model outputs represent the probability that the report is **temporally followed by a documented clinical action**, serving as an operational proxy for clinical relevance rather than a mechanistic causal claim.

---

## 3. Label Noise, Ambiguity, and Confounding

| Source of Uncertainty | Mechanism | Mitigation / Audit Protocol |
|---|---|---|
| **EHR Charting Latency** | Nurse or pharmacist order entry delayed relative to actual bedside administration | Enforced minimum 30-minute lead time ($t_{\text{pred}} \le t_{\text{lab}} - 30\text{m}$) and wide 12-hour action window |
| **Empirical / Protocolized Orders** | Fixed protocol dosing (e.g. sliding scale insulin) documented irrespective of acute clinician review | Subtype stratification separating active routine orders from acute new initiations |
| **Uncaptured Actions** | Non-pharmacological decisions (e.g. dietary hold, fluid restriction, repositioning) omitted from structured tables | Explicitly acknowledged in false positive error analyses |
| **Observation Censoring** | Patient discharged from ICU within $< 12$ hours of test result | Truncated windows flagged; required ICU LOS $\ge 24$ hours for cohort inclusion |
