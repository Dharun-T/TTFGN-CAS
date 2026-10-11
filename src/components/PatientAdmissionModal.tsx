import React, { useState } from "react";
import {
  X,
  UserPlus,
  Flame,
  AlertCircle,
  CheckCircle2,
  Stethoscope,
  Activity,
  Pill,
  Sparkles,
} from "lucide-react";
import { PatientCase, createCustomPatientCase } from "../data/patientDatabase";

interface AdmissionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdmitPatient: (patient: PatientCase) => void;
}

const PRESET_CASES = [
  {
    name: "Sepsis AKI (Refractory Vasoplegia)",
    unit: "MICU" as const,
    bed: "MICU-18",
    alias: "Elena Rostova",
    age: 58,
    gender: "Female" as const,
    diagnosis: "Severe Sepsis / KDIGO Stage 3 AKI",
    analyte: "Creatinine" as const,
    recentValue: 2.7,
    deltaValue: 0.8,
    hasVasopressor: true,
    hasDiuretic: false,
    sofaRenal: 3,
  },
  {
    name: "DKA with Acidemic Hyperkalemia",
    unit: "SICU" as const,
    bed: "SICU-11",
    alias: "Darius Vance",
    age: 32,
    gender: "Male" as const,
    diagnosis: "Diabetic Ketoacidosis / Severe Met Acidosis",
    analyte: "Potassium" as const,
    recentValue: 5.6,
    deltaValue: 0.7,
    hasInsulin: true,
    hasVasopressor: false,
    sofaRenal: 2,
  },
  {
    name: "Severe Upper GI Bleed",
    unit: "MICU" as const,
    bed: "MICU-06",
    alias: "Robert Chen",
    age: 64,
    gender: "Male" as const,
    diagnosis: "Bleeding Duodenal Ulcer / Melena",
    analyte: "Hemoglobin" as const,
    recentValue: 7.4,
    deltaValue: -1.6,
    hasVasopressor: true,
    hasDiuretic: false,
    sofaRenal: 1,
  },
  {
    name: "Post-Sternotomy Hypokalemia",
    unit: "CVICU" as const,
    bed: "CVICU-04",
    alias: "Gordon Miller",
    age: 71,
    gender: "Male" as const,
    diagnosis: "Post-AVR & CABG / Rapid Diuresis",
    analyte: "Potassium" as const,
    recentValue: 3.4,
    deltaValue: -0.4,
    hasDiuretic: true,
    hasVasopressor: true,
    sofaRenal: 1,
  },
  {
    name: "SIADH / Hypotonic Encephalopathy",
    unit: "NICU" as const,
    bed: "NICU-07",
    alias: "Miriam Kelly",
    age: 61,
    gender: "Female" as const,
    diagnosis: "Glioblastoma / Cerebral Edema / SIADH",
    analyte: "Sodium" as const,
    recentValue: 125.0,
    deltaValue: -5.0,
    hasVasopressor: false,
    hasDiuretic: true,
    sofaRenal: 1,
  },
  {
    name: "CCU Stable Post-PCI Observation",
    unit: "CCU" as const,
    bed: "CCU-08",
    alias: "Harold Jenkins",
    age: 59,
    gender: "Male" as const,
    diagnosis: "NSTEMI Post-Stenting (LAD)",
    analyte: "Glucose" as const,
    recentValue: 122.0,
    deltaValue: 4.0,
    hasVasopressor: false,
    hasDiuretic: false,
    sofaRenal: 0,
  },
];

export const PatientAdmissionModal: React.FC<AdmissionModalProps> = ({
  isOpen,
  onClose,
  onAdmitPatient,
}) => {
  const [bed, setBed] = useState("MICU-20");
  const [nameAlias, setNameAlias] = useState("Jonathan Bell");
  const [age, setAge] = useState(63);
  const [gender, setGender] = useState<"Male" | "Female">("Male");
  const [careUnit, setCareUnit] = useState<"MICU" | "SICU" | "CCU" | "CVICU" | "TSICU" | "NICU">("MICU");
  const [admitDiagnosis, setAdmitDiagnosis] = useState("Acute Sepsis / Respiratory Failure");
  const [analyte, setAnalyte] = useState<"Potassium" | "Glucose" | "Hemoglobin" | "Sodium" | "Creatinine">("Potassium");
  const [recentValue, setRecentValue] = useState(5.4);
  const [deltaValue, setDeltaValue] = useState(0.6);
  const [hasVasopressor, setHasVasopressor] = useState(false);
  const [hasDiuretic, setHasDiuretic] = useState(true);
  const [hasInsulin, setHasInsulin] = useState(false);
  const [sofaRenal, setSofaRenal] = useState(2);

  if (!isOpen) return null;

  const handleApplyPreset = (preset: typeof PRESET_CASES[0]) => {
    setBed(preset.bed);
    setNameAlias(preset.alias);
    setAge(preset.age);
    setGender(preset.gender);
    setCareUnit(preset.unit);
    setAdmitDiagnosis(preset.diagnosis);
    setAnalyte(preset.analyte);
    setRecentValue(preset.recentValue);
    setDeltaValue(preset.deltaValue);
    setHasVasopressor(Boolean(preset.hasVasopressor));
    setHasDiuretic(Boolean(preset.hasDiuretic));
    setHasInsulin(Boolean(preset.hasInsulin));
    setSofaRenal(preset.sofaRenal);
  };

  const handleAdmit = () => {
    const newPatient = createCustomPatientCase({
      bed,
      nameAlias,
      age,
      gender,
      careUnit,
      admitDiagnosis,
      analyte,
      recentValue,
      deltaValue,
      hasDiuretic,
      hasVasopressor,
      hasInsulin,
      sofaRenal,
    });
    onAdmitPatient(newPatient);
    onClose();
  };

  // Preview CAS
  const previewTarget = Number((recentValue + deltaValue).toFixed(1));
  let estCas = 0.22;
  if (analyte === "Potassium") {
    if (previewTarget > 5.2) estCas += (previewTarget - 5.0) * 0.42;
    if (previewTarget < 3.4) estCas += (3.5 - previewTarget) * 0.45;
  } else if (analyte === "Creatinine") {
    if (previewTarget > 1.8) estCas += (previewTarget - 1.3) * 0.32;
  } else if (analyte === "Hemoglobin") {
    if (previewTarget < 8.0) estCas += (8.0 - previewTarget) * 0.35;
  } else if (analyte === "Glucose") {
    if (previewTarget > 190) estCas += (previewTarget - 180) * 0.003;
    if (previewTarget < 65) estCas += 0.55;
  } else if (analyte === "Sodium") {
    if (previewTarget < 128 || previewTarget > 150) estCas += 0.45;
  }
  if (hasVasopressor) estCas += 0.15;
  if (hasDiuretic) estCas += 0.12;
  if (sofaRenal >= 2) estCas += 0.14;
  estCas = Math.min(0.97, Math.max(0.08, Number(estCas.toFixed(3))));
  const estBand = estCas >= 0.75 ? "HIGH" : estCas >= 0.40 ? "MEDIUM" : "LOW";

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden my-6">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/90">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-lg bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400">
              <UserPlus className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                Admit New Patient Case to Ward Census
              </h2>
              <p className="text-xs text-slate-400">
                Instantiate custom clinical cases or load real clinical phenotypes for demonstration.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto font-sans">
          {/* Quick Phenotype Presets */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-2 flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              Quick Load Clinical Phenotypes:
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {PRESET_CASES.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleApplyPreset(preset)}
                  className="text-left p-2.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-cyan-600/80 hover:bg-slate-850 transition-all text-xs group"
                >
                  <div className="font-semibold text-slate-200 group-hover:text-cyan-300 truncate">
                    {preset.name}
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between mt-0.5">
                    <span>{preset.unit} · {preset.analyte}</span>
                    <span className="text-cyan-400 font-bold">{preset.recentValue}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Form Fields */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-slate-800">
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Bed Identifier</label>
              <input
                type="text"
                value={bed}
                onChange={(e) => setBed(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Patient Alias / Name</label>
              <input
                type="text"
                value={nameAlias}
                onChange={(e) => setNameAlias(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Target Care Unit</label>
              <select
                value={careUnit}
                onChange={(e) => setCareUnit(e.target.value as any)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
              >
                <option value="MICU">MICU (Medical ICU)</option>
                <option value="SICU">SICU (Surgical ICU)</option>
                <option value="CCU">CCU (Coronary Care)</option>
                <option value="CVICU">CVICU (Cardiothoracic)</option>
                <option value="TSICU">TSICU (Trauma ICU)</option>
                <option value="NICU">NICU (Neuro ICU)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Age & Gender</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  min="18"
                  max="99"
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="w-20 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono"
                />
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value as any)}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            <div className="sm:col-span-2">
              <label className="text-xs font-semibold text-slate-300 block mb-1">Admitting Clinical Diagnosis</label>
              <input
                type="text"
                value={admitDiagnosis}
                onChange={(e) => setAdmitDiagnosis(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              />
            </div>
          </div>

          {/* Laboratory Parameter Config */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-4">
            <div className="text-xs font-semibold text-cyan-300 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Pending Laboratory Investigation Configuration
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Target Analyte</label>
                <select
                  value={analyte}
                  onChange={(e) => {
                    const a = e.target.value as any;
                    setAnalyte(a);
                    if (a === "Glucose") { setRecentValue(160); setDeltaValue(30); }
                    else if (a === "Hemoglobin") { setRecentValue(8.2); setDeltaValue(-1.0); }
                    else if (a === "Sodium") { setRecentValue(130); setDeltaValue(-4); }
                    else if (a === "Creatinine") { setRecentValue(2.2); setDeltaValue(0.6); }
                    else { setRecentValue(5.3); setDeltaValue(0.5); }
                  }}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-cyan-300 font-mono"
                >
                  <option value="Potassium">Potassium (mEq/L)</option>
                  <option value="Creatinine">Creatinine (mg/dL)</option>
                  <option value="Glucose">Glucose (mg/dL)</option>
                  <option value="Hemoglobin">Hemoglobin (g/dL)</option>
                  <option value="Sodium">Sodium (mEq/L)</option>
                </select>
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">Most Recent Draw Value</label>
                <input
                  type="number"
                  step="0.1"
                  value={recentValue}
                  onChange={(e) => setRecentValue(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">Expected Trend / Delta (Δ)</label>
                <input
                  type="number"
                  step="0.1"
                  value={deltaValue}
                  onChange={(e) => setDeltaValue(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-amber-300 font-mono"
                />
              </div>
            </div>

            {/* Context Switches */}
            <div className="pt-2 border-t border-slate-800 flex flex-wrap gap-4 text-xs font-mono">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasVasopressor}
                  onChange={(e) => setHasVasopressor(e.target.checked)}
                  className="accent-cyan-400"
                />
                <span className="text-slate-300">Vasopressor Infusion</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasDiuretic}
                  onChange={(e) => setHasDiuretic(e.target.checked)}
                  className="accent-cyan-400"
                />
                <span className="text-slate-300">Loop Diuretic (Lasix)</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasInsulin}
                  onChange={(e) => setHasInsulin(e.target.checked)}
                  className="accent-cyan-400"
                />
                <span className="text-slate-300">Active Insulin Drip</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sofaRenal >= 2}
                  onChange={(e) => setSofaRenal(e.target.checked ? 3 : 1)}
                  className="accent-cyan-400"
                />
                <span className="text-slate-300">Renal SOFA ≥ 2</span>
              </label>
            </div>
          </div>

          {/* Real-Time Live CAS Preview Card */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-mono uppercase text-slate-500">Live T-TFGN Inference Preview</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-300">Projected Result: <strong className="font-mono text-cyan-300">{previewTarget}</strong></span>
                <span className="text-slate-700">·</span>
                <span className="text-xs text-slate-300">Model Actionability: <strong className="font-mono text-slate-100">CAS {estCas.toFixed(3)}</strong></span>
              </div>
            </div>

            <div>
              {estBand === "HIGH" ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-rose-950 text-rose-300 border border-rose-800">
                  <Flame className="h-3.5 w-3.5 text-rose-400" />
                  HIGH ACTIONABILITY (CAS ≥ 0.75)
                </span>
              ) : estBand === "MEDIUM" ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800">
                  <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
                  MODERATE (0.40 - 0.75)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                  STABLE / LOW (CAS &lt; 0.40)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-900/90 flex items-center justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleAdmit}
            className="px-5 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold font-mono transition-all shadow-md flex items-center gap-2"
          >
            <UserPlus className="h-4 w-4" />
            Admit Patient & Compute CAS
          </button>
        </div>
      </div>
    </div>
  );
};
