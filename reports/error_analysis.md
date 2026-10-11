# Error Analysis and Retrospective Quadrant Audit

Detailed qualitative and quantitative examination of model predictions across the confusion matrix quadrants.

---

## 1. True Positives (High CAS, Action Occurred)
The model correctly anticipated that acute laboratory trajectories within high-risk clinical context would trigger prompt clinical management:
### Patient PATIENT-10000048 - Hemoglobin
- **Predicted CAS:** 0.9062 (HIGH)
- **Target Value:** 4.52 g/dL
- **Observed Action:** True (medication_modification)
- **Intervention:** Medication: Packed Red Blood Cells (PRBC) (1 Unit)
- **Time to Action:** 2.89 hours post-result
### Patient PATIENT-10000048 - Hemoglobin
- **Predicted CAS:** 0.9029 (HIGH)
- **Target Value:** 3.88 g/dL
- **Observed Action:** True (medication_modification)
- **Intervention:** Medication: Packed Red Blood Cells (PRBC) (1 Unit)
- **Time to Action:** 1.44 hours post-result


## 2. False Positives (High CAS, No Action Documented)
Instances where the model predicted high actionability, but no qualifying intervention was documented in the EHR within 12 hours:
### Patient PATIENT-10000041 - Hemoglobin
- **Predicted CAS:** 0.8998 (HIGH)
- **Target Value:** 5.66 g/dL
- **Observed Action:** False (none)
- **Intervention:** No qualifying clinical intervention documented within window
- **Time to Action:** nan hours post-result
### Patient PATIENT-10000055 - Hemoglobin
- **Predicted CAS:** 0.8915 (HIGH)
- **Target Value:** 7.88 g/dL
- **Observed Action:** False (none)
- **Intervention:** No qualifying clinical intervention documented within window
- **Time to Action:** nan hours post-result

*Root Cause Analysis:* Clinical 'watchful waiting' where the care team elected to observe without pharmacological adjustment, or institutional protocol variations.

## 3. False Negatives (Low CAS, Action Documented)
Instances where the model predicted low actionability, but a qualifying intervention occurred:
### Patient PATIENT-10000033 - Glucose
- **Predicted CAS:** 0.2327 (LOW)
- **Target Value:** 131.99 mg/dL
- **Observed Action:** True (urgent_repeat_lab)
- **Intervention:** Urgent repeat Glucose ordered within 6.0h
- **Time to Action:** 6.0 hours post-result
### Patient PATIENT-10000035 - Glucose
- **Predicted CAS:** 0.2401 (LOW)
- **Target Value:** 120.63 mg/dL
- **Observed Action:** True (urgent_repeat_lab)
- **Intervention:** Urgent repeat Glucose ordered within 6.0h
- **Time to Action:** 6.0 hours post-result

*Root Cause Analysis:* Clinical decisions triggered by unmodeled bedside events (e.g. abrupt clinical deterioration not captured in routine vitals).

## 4. True Negatives (Low CAS, No Action Occurred)
The model correctly identified stable patients whose upcoming test produced no subsequent change:
### Patient PATIENT-10000033 - Hemoglobin
- **Predicted CAS:** 0.23 (LOW)
- **Target Value:** 10.72 g/dL
- **Observed Action:** False (none)
- **Intervention:** No qualifying clinical intervention documented within window
- **Time to Action:** nan hours post-result
### Patient PATIENT-10000014 - Hemoglobin
- **Predicted CAS:** 0.2304 (LOW)
- **Target Value:** 13.21 g/dL
- **Observed Action:** False (none)
- **Intervention:** No qualifying clinical intervention documented within window
- **Time to Action:** nan hours post-result

