import React, { useState } from "react";
import { X, Printer, Copy, Check, FileText, Activity, Pill, ShieldAlert } from "lucide-react";
import { PatientCase } from "../data/patientDatabase";

interface RoundsSummaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  patient: PatientCase;
}

export const RoundsSummaryModal: React.FC<RoundsSummaryModalProps> = ({
  isOpen,
  onClose,
  patient,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleCopy = () => {
    const text = `=====================================================
ICU ROUNDS SUMMARY & CAS PROTOCOL SHEET
=====================================================
BED: ${patient.demographics.bed} (${patient.nameAlias}) | MRN: ${patient.id}
UNIT: ${patient.demographics.careUnit} | AGE/GENDER: ${patient.demographics.age}yo ${patient.demographics.gender}
ADMIT DIAGNOSIS: ${patient.demographics.admitDiagnosis}
CODE STATUS: ${patient.demographics.codeStatus} | LOS: ${patient.demographics.losHours} hours
SOFA SCORE: Cardio ${patient.demographics.sofaCardio} + Renal ${patient.demographics.sofaRenal} = Total ${patient.demographics.sofaCardio + patient.demographics.sofaRenal}

BEDSIDE VITALS:
- HR: ${patient.vitals.heartRate} bpm (${patient.vitals.rhythm})
- BP: ${patient.vitals.bloodPressure} (MAP ${patient.vitals.map} mmHg)
- SpO2: ${patient.vitals.spo2}% | RR: ${patient.vitals.respiratoryRate}/min | Temp: ${patient.vitals.temperature}°C | GCS: ${patient.vitals.gcs}/15

T-TFGN CLINICAL ACTIONABILITY EVALUATION:
- PENDING LAB: ${patient.analyte} (${patient.canonicalUnit})
- CAS PREDICTION: ${patient.cas.toFixed(3)} [PRIORITY: ${patient.band}]
- RECOMMENDED POSTURE: ${patient.clinicalPostureRecommendation}

ACTIVE MEDICATIONS / INFUSIONS:
${patient.activeMedications.map((m) => `- ${m.drug} (${m.route}, ${m.dose}) [${m.category}]`).join("\n")}

AUTOMATED CPOE RECOMMENDATIONS:
${patient.orderSet.map((o) => `[${o.urgent ? "STAT" : "ROUTINE"}] ${o.title}: ${o.details}`).join("\n")}

SBAR SHIFT HANDOVER:
S: ${patient.sbar.situation}
B: ${patient.sbar.background}
A: ${patient.sbar.assessment}
R: ${patient.sbar.recommendation}
=====================================================`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl shadow-2xl overflow-hidden my-6">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/95">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-lg bg-indigo-950 border border-indigo-800 flex items-center justify-center text-indigo-400">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                ICU Rounds & Handover Summary: {patient.demographics.bed}
              </h2>
              <p className="text-xs text-slate-400">
                Printable clinical dossier with automated decision support for morning physician rounds.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-200 border border-slate-700 flex items-center gap-1.5 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-emerald-300">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Copy Text</span>
                </>
              )}
            </button>
            <button
              onClick={handlePrint}
              className="px-3 py-1.5 rounded-lg bg-indigo-950 hover:bg-indigo-900 text-xs font-mono text-indigo-300 border border-indigo-700 flex items-center gap-1.5 transition-colors"
            >
              <Printer className="h-3.5 w-3.5" />
              <span>Print Sheet</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors ml-2"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Printable Content */}
        <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto font-sans bg-slate-950/60">
          {/* Patient Banner */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">Patient & Bed</span>
              <span className="text-sm font-bold text-slate-100">{patient.demographics.bed} · {patient.nameAlias}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">MRN / Care Unit</span>
              <span className="text-slate-200 font-bold">{patient.id} · {patient.demographics.careUnit}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">Demographics</span>
              <span className="text-slate-200">{patient.demographics.age}yo {patient.demographics.gender}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">Code Status / LOS</span>
              <span className="text-slate-200">{patient.demographics.codeStatus} · {patient.demographics.losHours}h</span>
            </div>
          </div>

          {/* CAS Evaluation Card */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono border-b border-slate-800/80 pb-2">
              <span className="text-cyan-300 font-bold uppercase tracking-wider">
                T-TFGN Actionability Prediction (t_pred -30m)
              </span>
              <span className={`px-2 py-0.5 rounded font-bold ${
                patient.band === "HIGH" ? "bg-rose-950 text-rose-300 border border-rose-800" : patient.band === "MEDIUM" ? "bg-amber-950 text-amber-300 border border-amber-800" : "bg-emerald-950 text-emerald-300 border border-emerald-800"
              }`}>
                CAS: {patient.cas.toFixed(3)} [{patient.band} PRIORITY]
              </span>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed pt-1">
              <strong>Clinical Directive:</strong> {patient.clinicalPostureRecommendation}
            </p>
          </div>

          {/* Vitals Snapshot */}
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Heart Rate</span>
              <span className="text-base font-bold text-slate-100">{patient.vitals.heartRate} bpm</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Blood Pressure</span>
              <span className="text-base font-bold text-slate-100">{patient.vitals.bloodPressure}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Mean Arterial</span>
              <span className="text-base font-bold text-cyan-400">{patient.vitals.map} mmHg</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">SpO2 / RR</span>
              <span className="text-base font-bold text-slate-100">{patient.vitals.spo2}% / {patient.vitals.respiratoryRate}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Core Temp</span>
              <span className="text-base font-bold text-slate-100">{patient.vitals.temperature}°C</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">GCS Score</span>
              <span className="text-base font-bold text-emerald-400">{patient.vitals.gcs} / 15</span>
            </div>
          </div>

          {/* Active Meds & Order Sets */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
              <div className="text-xs font-bold text-slate-200 font-mono flex items-center gap-1.5">
                <Pill className="h-4 w-4 text-cyan-400" /> Continuous Infusions & Active Meds
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                {patient.activeMedications.map((m, idx) => (
                  <div key={idx} className="p-2 rounded bg-slate-950 border border-slate-800/80 flex justify-between">
                    <span>{m.drug}</span>
                    <span className="text-slate-400">{m.dose}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
              <div className="text-xs font-bold text-slate-200 font-mono flex items-center gap-1.5">
                <Activity className="h-4 w-4 text-indigo-400" /> Pre-Populated CPOE Orders
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                {patient.orderSet.map((o) => (
                  <div key={o.id} className="p-2 rounded bg-slate-950 border border-slate-800/80 space-y-0.5">
                    <div className="flex justify-between font-bold text-slate-200">
                      <span>{o.title}</span>
                      {o.urgent && <span className="text-rose-400 text-[10px]">STAT</span>}
                    </div>
                    <div className="text-[11px] text-slate-400 font-sans">{o.details}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SBAR Grid */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3 text-xs">
            <div className="text-xs font-bold text-slate-200 font-mono uppercase tracking-wider">
              Structured SBAR Clinical Sign-Out
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-bold text-rose-400 font-mono block mb-1">S · Situation</span>
                <p className="text-slate-300 leading-relaxed font-sans">{patient.sbar.situation}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-bold text-amber-400 font-mono block mb-1">B · Background</span>
                <p className="text-slate-300 leading-relaxed font-sans">{patient.sbar.background}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-bold text-cyan-400 font-mono block mb-1">A · Assessment</span>
                <p className="text-slate-300 leading-relaxed font-sans">{patient.sbar.assessment}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-bold text-emerald-400 font-mono block mb-1">R · Recommendation</span>
                <p className="text-slate-300 leading-relaxed font-sans">{patient.sbar.recommendation}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/90 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition-colors"
          >
            Close Sheet
          </button>
        </div>
      </div>
    </div>
  );
};
