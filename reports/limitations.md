# Study Limitations and Ethical Considerations

---

## 1. Study Limitations

1. **Single-Center Retrospective Bias:** MIMIC-IV originates from a single tertiary academic medical center (Beth Israel Deaconess Medical Center, Boston, MA). Institutional practice patterns, local order sets, and staffing models may not generalize to community or international ICU settings.
2. **Observational Confounding by Indication:** Patients with higher monitoring frequency are intrinsically more critically ill. The model may capture clinician ordering intent as a latent confounder.
3. **Absence of Prospective Validation:** All evaluations are strictly retrospective. Prospective clinical trials are required before considering any real-time decision-support deployment.
4. **Subtype Imbalance:** Certain analytes (e.g. Sodium) exhibit lower acute medication intervention rates compared to Glucose or Potassium, impacting analyte-specific positive predictive value.
5. **Structured Data Exclusions:** Unstructured clinical progress notes, nursing narratives, and imaging reports were excluded to ensure real-time computational tractability, omitting potential qualitative rationale.

---

## 2. Ethical and Data Governance Statement

- **MIMIC-IV De-identification:** Data conforms to HIPAA Safe Harbor de-identification standards under institutional review board (IRB) approval (Beth Israel Deaconess Medical Center and MIT).
- **Date Shifting:** Real calendar dates are shifted per patient while preserving exact relative temporal distances between sequential events.
- **Fairness & Subgroup Disparities:** Clinical action thresholds may vary across demographic cohorts due to structural healthcare disparities. Models must be audited for subgroup calibration equity prior to deployment.
- **Decision-Support Guardrails:** CAS is strictly an assistive research index and MUST NOT be used to autonomously order, delay, or cancel diagnostic tests or medical treatments.
