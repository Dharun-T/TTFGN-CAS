import React, { useState } from "react";
import { Sparkles, Layers, Info, CheckCircle2 } from "lucide-react";

interface MatrixCell {
  labDim: string;
  contextDim: string;
  weight: number; // 0.0 to 1.0
  synergyLevel: "High" | "Moderate" | "Mild";
  clinicalRationale: string;
  exampleScenario: string;
}

const TENSOR_INTERACTIONS: MatrixCell[] = [
  {
    labDim: "Velocity (d/dt)",
    contextDim: "Vasopressor Infusion",
    weight: 0.88,
    synergyLevel: "High",
    clinicalRationale:
      "Rapid lab shifting in a patient on norepinephrine/vasopressin signals acute end-organ perfusion compromise or ischemic cellular lysis.",
    exampleScenario: "Sudden rise in potassium (+1.0 mEq/L/4h) in septic shock on dual pressors immediately alerts for bowel ischemia or worsening shock state.",
  },
  {
    labDim: "Velocity (d/dt)",
    contextDim: "Loop Diuretic (Lasix)",
    weight: 0.94,
    synergyLevel: "High",
    clinicalRationale:
      "Steep negative electrolyte velocity paired with continuous loop diuresis accelerates dangerous hypokalemic cardiac vulnerability.",
    exampleScenario: "Potassium dropping from 4.0 to 3.2 mEq/L following 80mg IV Lasix in post-CABG yields maximal synergy score (0.94).",
  },
  {
    labDim: "Velocity (d/dt)",
    contextDim: "Renal Failure (SOFA ≥ 2)",
    weight: 0.82,
    synergyLevel: "High",
    clinicalRationale:
      "Accelerating lab changes cannot be compensated for when renal clearance mechanisms are failing (loss of homeostatic buffering).",
    exampleScenario: "Creatinine velocity doubling in oliguric patient triggers immediate prompt for renal dose adjustment and CRRT readiness.",
  },
  {
    labDim: "Velocity (d/dt)",
    contextDim: "Metabolic Acidosis",
    weight: 0.79,
    synergyLevel: "High",
    clinicalRationale:
      "Profound acidemia drives transcellular shift of potassium outward from intracellular space, exaggerating apparent hyperkalemic threat.",
    exampleScenario: "In DKA or severe lactic acidosis, K+ velocity is modulated by simultaneous pH shifts.",
  },
  {
    labDim: "Time-Decayed Stale Lab",
    contextDim: "Vasopressor Infusion",
    weight: 0.38,
    synergyLevel: "Mild",
    clinicalRationale:
      "Old lab results (> 18h prior) have exponentially reduced prognostic relevance in highly volatile vasopressor-dependent hemodynamics.",
    exampleScenario: "T-TFGN decay factor λ rapidly suppresses stale draws when pressors are actively titrated.",
  },
  {
    labDim: "Time-Decayed Stale Lab",
    contextDim: "Loop Diuretic (Lasix)",
    weight: 0.45,
    synergyLevel: "Moderate",
    clinicalRationale:
      "Stale baseline establishes pre-diuresis starting point, but recent velocity dominates immediate action probability.",
    exampleScenario: "Morning baseline potassium provides frame of reference for afternoon bolus response.",
  },
  {
    labDim: "Time-Decayed Stale Lab",
    contextDim: "Renal Failure (SOFA ≥ 2)",
    weight: 0.62,
    synergyLevel: "Moderate",
    clinicalRationale:
      "Baseline chronic kidney disease measurements remain informative for calculating baseline GFR vs acute injury delta.",
    exampleScenario: "Distinguishing acute-on-chronic kidney injury from sudden de novo renal failure.",
  },
  {
    labDim: "Time-Decayed Stale Lab",
    contextDim: "Metabolic Acidosis",
    weight: 0.35,
    synergyLevel: "Mild",
    clinicalRationale:
      "Acid-base derangements evolve rapidly; stale measurements carry minimal predictive weight for acute shifting.",
    exampleScenario: "Blood gas and electrolyte draws > 12h prior contribute low gating weights.",
  },
  {
    labDim: "Extreme Outlier Magnitude",
    contextDim: "Vasopressor Infusion",
    weight: 0.91,
    synergyLevel: "High",
    clinicalRationale:
      "Critical numerical panic thresholds (K+ > 6.0, Hgb < 7.0) combined with circulatory shock dictate immediate bedside interventions.",
    exampleScenario: "Hemoglobin 6.4 in patient with MAP 60 triggers immediate uncrossmatched blood transfusion protocol.",
  },
  {
    labDim: "Extreme Outlier Magnitude",
    contextDim: "Loop Diuretic (Lasix)",
    weight: 0.74,
    synergyLevel: "Moderate",
    clinicalRationale:
      "Severe baseline electrolyte derangements prompt immediate cessation or augmentation of diuretic therapy.",
    exampleScenario: "Sodium < 125 mEq/L mandates immediate cessation of thiazides and free water restriction.",
  },
  {
    labDim: "Extreme Outlier Magnitude",
    contextDim: "Renal Failure (SOFA ≥ 2)",
    weight: 0.89,
    synergyLevel: "High",
    clinicalRationale:
      "Severe azotemia and hyperkalemia in anuric renal failure cannot self-correct, necessitating dialysis or binder orders.",
    exampleScenario: "Potassium 6.2 with BUN 85 and creatinine 4.2 triggers STAT urgent hemodialysis setup.",
  },
  {
    labDim: "Extreme Outlier Magnitude",
    contextDim: "Metabolic Acidosis",
    weight: 0.85,
    synergyLevel: "High",
    clinicalRationale:
      "Concurrent extreme metabolic derangements compound arrhythmogenic risk and hemodynamic collapse.",
    exampleScenario: "Severe hyperkalemia with bicarbonate < 12 mEq/L warrants urgent IV sodium bicarbonate administration.",
  },
  {
    labDim: "Acceleration (d²t)",
    contextDim: "Vasopressor Infusion",
    weight: 0.86,
    synergyLevel: "High",
    clinicalRationale:
      "Second-derivative acceleration indicates runaway physiological destabilization requiring preemptive clinical counteraction.",
    exampleScenario: "Potassium rising with accelerating velocity over 3 consecutive draws signals imminent crash.",
  },
  {
    labDim: "Acceleration (d²t)",
    contextDim: "Loop Diuretic (Lasix)",
    weight: 0.68,
    synergyLevel: "Moderate",
    clinicalRationale:
      "Non-linear steepening of electrolyte loss following diuretic dose escalation.",
    exampleScenario: "Doubling diuretic dose causes non-linear potassium excretion acceleration.",
  },
  {
    labDim: "Acceleration (d²t)",
    contextDim: "Renal Failure (SOFA ≥ 2)",
    weight: 0.78,
    synergyLevel: "High",
    clinicalRationale:
      "Accelerating creatinine trajectory signifies failure of renal autoregulation and acute tubular necrosis.",
    exampleScenario: "Rapid transition from KDIGO Stage 1 to Stage 3 within 24 hours.",
  },
  {
    labDim: "Acceleration (d²t)",
    contextDim: "Metabolic Acidosis",
    weight: 0.72,
    synergyLevel: "Moderate",
    clinicalRationale:
      "Worsening lactic acidosis or ketoacidosis shifts electrolyte dynamics non-linearly over time.",
    exampleScenario: "Accelerating base deficit in hemorrhagic shock prior to massive transfusion.",
  },
];

const LAB_DIMS = ["Velocity (d/dt)", "Time-Decayed Stale Lab", "Extreme Outlier Magnitude", "Acceleration (d²t)"];
const CONTEXT_DIMS = ["Vasopressor Infusion", "Loop Diuretic (Lasix)", "Renal Failure (SOFA ≥ 2)", "Metabolic Acidosis"];

export const TensorSynergyMatrix: React.FC = () => {
  const [selectedCell, setSelectedCell] = useState<MatrixCell>(TENSOR_INTERACTIONS[0]);

  const getCell = (lab: string, context: string) => {
    return TENSOR_INTERACTIONS.find((c) => c.labDim === lab && c.contextDim === context) || TENSOR_INTERACTIONS[0];
  };

  const getHeatmapColor = (weight: number) => {
    if (weight >= 0.85) return "bg-rose-950/80 text-rose-300 border-rose-700/80 hover:bg-rose-900/90";
    if (weight >= 0.70) return "bg-amber-950/80 text-amber-300 border-amber-700/80 hover:bg-amber-900/90";
    if (weight >= 0.50) return "bg-indigo-950/80 text-indigo-300 border-indigo-700/80 hover:bg-indigo-900/90";
    return "bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-900";
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-indigo-400" />
            <h3 className="text-base font-bold text-slate-100">
              Low-Rank Bilinear Tensor Interaction Matrix (Rank R = 16)
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Explaining cross-modal multiplicative interaction: h_tensor = (h_lab ⊗ h_context) W_tensor. Click any cell to inspect clinical rationale.
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400">
          Rank R = 16 Parameter Factorization
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 4x4 Heatmap Matrix */}
        <div className="lg:col-span-2 space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border border-slate-800 rounded-xl overflow-hidden font-mono">
              <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="p-3 text-slate-300">Lab Feature (h_lab) ↓ \ Context (h_ctx) →</th>
                  {CONTEXT_DIMS.map((ctx) => (
                    <th key={ctx} className="p-3 text-center text-slate-300">
                      {ctx}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {LAB_DIMS.map((lab) => (
                  <tr key={lab} className="bg-slate-900/50">
                    <td className="p-3 font-semibold text-slate-200 bg-slate-950/60 border-r border-slate-800">
                      {lab}
                    </td>
                    {CONTEXT_DIMS.map((ctx) => {
                      const cell = getCell(lab, ctx);
                      const isSelected = selectedCell.labDim === lab && selectedCell.contextDim === ctx;
                      return (
                        <td key={ctx} className="p-2 text-center">
                          <button
                            onClick={() => setSelectedCell(cell)}
                            className={`w-full py-2.5 px-2 rounded-lg border text-xs font-bold transition-all ${getHeatmapColor(
                              cell.weight
                            )} ${isSelected ? "ring-2 ring-cyan-400 scale-105 shadow-md" : ""}`}
                          >
                            <span className="block">{cell.weight.toFixed(2)}</span>
                            <span className="text-[9px] uppercase tracking-wider block opacity-75">
                              {cell.synergyLevel}
                            </span>
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded bg-rose-600"></span> High Synergy (&gt; 0.75)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded bg-amber-600"></span> Moderate Synergy (0.50 - 0.75)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded bg-slate-700"></span> Mild / Low Synergy (&lt; 0.50)
            </span>
          </div>
        </div>

        {/* Selected Interaction Deep-Dive */}
        <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 border-b border-slate-800 pb-2">
              <span className="uppercase tracking-wider">Inspected Bilinear Interaction</span>
              <span className={`px-2 py-0.5 rounded font-bold ${
                selectedCell.synergyLevel === "High" ? "bg-rose-950 text-rose-300 border border-rose-800" : "bg-amber-950 text-amber-300 border border-amber-800"
              }`}>
                Score: {selectedCell.weight.toFixed(2)}
              </span>
            </div>

            <div className="mt-3 space-y-2">
              <div className="text-sm font-bold text-cyan-300 font-mono">
                {selectedCell.labDim} ⊗ {selectedCell.contextDim}
              </div>

              <div className="space-y-1">
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wider">
                  Physiological Rationale:
                </span>
                <p className="text-xs text-slate-200 leading-relaxed font-sans">
                  {selectedCell.clinicalRationale}
                </p>
              </div>

              <div className="space-y-1 pt-2 border-t border-slate-800/80">
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wider">
                  Critical Care Example:
                </span>
                <p className="text-xs text-amber-200/90 leading-relaxed font-sans bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  {selectedCell.exampleScenario}
                </p>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-400">
            <span>Mathematical Formulation: </span>
            <span className="text-slate-200 block font-bold mt-0.5">
              (h_lab^T · W_r) · (W_r^T · h_ctx) with R = 16
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
