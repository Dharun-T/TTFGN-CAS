import React, { useState, useMemo } from "react";
import { Clock, Eye, Sparkles, TrendingUp, Info } from "lucide-react";
import { LookbackPoint, PatientCase } from "../data/patientDatabase";

interface TrajectoryChartProps {
  patient: PatientCase;
  analyteName: string;
  unit: string;
  normalRange: { min: number; max: number };
  trajectory: LookbackPoint[];
  targetReport: { time: string; value: number; status: string };
  isPrimary: boolean;
}

export const TrajectoryChart: React.FC<TrajectoryChartProps> = ({
  patient,
  analyteName,
  unit,
  normalRange,
  trajectory,
  targetReport,
  isPrimary,
}) => {
  const [showTimeDecayCurve, setShowTimeDecayCurve] = useState(true);
  const [showCounterfactual, setShowCounterfactual] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
    label: string;
    value: number;
    time: string;
    delta?: string;
  } | null>(null);

  // Mathematical Coordinate Mapping
  const chartBounds = useMemo(() => {
    const allValues = [
      ...trajectory.map((p) => p.value),
      targetReport.value,
      normalRange.min,
      normalRange.max,
    ];
    const valMin = Math.min(...allValues);
    const valMax = Math.max(...allValues);
    const margin = (valMax - valMin) * 0.25 || 1.0;

    const yDomainMin = Math.max(0, valMin - margin);
    const yDomainMax = valMax + margin;

    // SVG coordinate constants
    const svgWidth = 840;
    const svgHeight = 280;
    const plotLeft = 60;
    const plotRight = 800;
    const plotTop = 35;
    const plotBottom = 230;

    // Time domain: -48h (left) to +12h (right) -> total 60h span
    // t_pred = -0.5h, t_lab = 0.0h
    const timeToX = (tHours: number) => {
      // tHours from -48 to +12
      const clamped = Math.max(-48, Math.min(12, tHours));
      const frac = (clamped + 48) / 60; // 0 at -48h, 0.8 at 0h, 1.0 at +12h
      return plotLeft + frac * (plotRight - plotLeft);
    };

    const valToY = (val: number) => {
      const frac = (val - yDomainMin) / (yDomainMax - yDomainMin);
      return plotBottom - frac * (plotBottom - plotTop);
    };

    return {
      yDomainMin,
      yDomainMax,
      plotLeft,
      plotRight,
      plotTop,
      plotBottom,
      timeToX,
      valToY,
    };
  }, [trajectory, targetReport.value, normalRange]);

  // Points for historical polyline
  const polylinePoints = useMemo(() => {
    return trajectory
      .map((p) => `${chartBounds.timeToX(p.timeHoursAgo)},${chartBounds.valToY(p.value)}`)
      .join(" ");
  }, [trajectory, chartBounds]);

  // Projected counterfactual path
  const counterfactualTargetY = useMemo(() => {
    const baseline = targetReport.value;
    if (analyteName === "Potassium") {
      return baseline > 5.0 ? chartBounds.valToY(4.5) : chartBounds.valToY(4.2);
    }
    if (analyteName === "Glucose") {
      return chartBounds.valToY(130);
    }
    if (analyteName === "Hemoglobin") {
      return chartBounds.valToY(baseline + 1.8);
    }
    if (analyteName === "Sodium") {
      return chartBounds.valToY(136);
    }
    return chartBounds.valToY(1.4);
  }, [targetReport.value, analyteName, chartBounds]);

  const tPredX = chartBounds.timeToX(-0.5);
  const tLabX = chartBounds.timeToX(0.0);
  const tActionX = chartBounds.timeToX(4.0);

  const normalTopY = chartBounds.valToY(normalRange.max);
  const normalBottomY = chartBounds.valToY(normalRange.min);
  const normalHeight = Math.max(4, normalBottomY - normalTopY);

  return (
    <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 space-y-4">
      {/* Controls & Legend Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-3">
        <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-400"></span>
            <span className="text-slate-300">Observed Trajectory</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded bg-sky-500/20 border border-sky-400"></span>
            <span className="text-sky-300">Normal Band ({normalRange.min} - {normalRange.max} {unit})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400"></span>
            <span className="text-amber-300">t_pred (-30m) Cutoff</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded bg-emerald-500/20 border border-emerald-500"></span>
            <span className="text-emerald-300">+12h Action Window</span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <button
            onClick={() => setShowTimeDecayCurve(!showTimeDecayCurve)}
            className={`px-2.5 py-1 rounded border transition-all flex items-center gap-1.5 ${
              showTimeDecayCurve
                ? "bg-amber-950/80 text-amber-300 border-amber-700 font-semibold"
                : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
            }`}
          >
            <Eye className="h-3.5 w-3.5" />
            Time Decay Weighting λ(t)
          </button>

          <button
            onClick={() => setShowCounterfactual(!showCounterfactual)}
            className={`px-2.5 py-1 rounded border transition-all flex items-center gap-1.5 ${
              showCounterfactual
                ? "bg-indigo-950/80 text-indigo-300 border-indigo-700 font-semibold"
                : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Projected Post-Intervention Path
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative h-80 w-full flex items-center">
        <svg
          className="w-full h-full select-none"
          viewBox="0 0 840 280"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <defs>
            <linearGradient id="trajGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="actionWindowGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.04" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line x1={chartBounds.plotLeft} y1={chartBounds.plotTop} x2={chartBounds.plotRight} y2={chartBounds.plotTop} stroke="#1e293b" strokeDasharray="3 3" />
          <line x1={chartBounds.plotLeft} y1={(chartBounds.plotTop + chartBounds.plotBottom) / 2} x2={chartBounds.plotRight} y2={(chartBounds.plotTop + chartBounds.plotBottom) / 2} stroke="#1e293b" strokeDasharray="3 3" />
          <line x1={chartBounds.plotLeft} y1={chartBounds.plotBottom} x2={chartBounds.plotRight} y2={chartBounds.plotBottom} stroke="#334155" />

          {/* Normal Reference Range Shaded Band */}
          <rect
            x={chartBounds.plotLeft}
            y={normalTopY}
            width={chartBounds.plotRight - chartBounds.plotLeft}
            height={normalHeight}
            fill="#0284c7"
            fillOpacity="0.09"
          />
          <line
            x1={chartBounds.plotLeft}
            y1={normalTopY}
            x2={chartBounds.plotRight}
            y2={normalTopY}
            stroke="#38bdf8"
            strokeWidth="1"
            strokeDasharray="4 3"
            opacity="0.45"
          />
          <line
            x1={chartBounds.plotLeft}
            y1={normalBottomY}
            x2={chartBounds.plotRight}
            y2={normalBottomY}
            stroke="#38bdf8"
            strokeWidth="1"
            strokeDasharray="4 3"
            opacity="0.45"
          />
          <text
            x={chartBounds.plotLeft + 8}
            y={normalTopY + 12}
            fill="#38bdf8"
            fontSize="9"
            fontFamily="monospace"
            opacity="0.9"
          >
            Normal Ref: {normalRange.min} - {normalRange.max} {unit}
          </text>

          {/* Observation Action Window (+12h) */}
          <rect
            x={tLabX}
            y={chartBounds.plotTop}
            width={chartBounds.plotRight - tLabX}
            height={chartBounds.plotBottom - chartBounds.plotTop}
            fill="url(#actionWindowGradient)"
            rx="4"
          />
          <text
            x={tLabX + 10}
            y={chartBounds.plotTop + 16}
            fill="#10b981"
            fontSize="10"
            fontWeight="bold"
            fontFamily="monospace"
          >
            Action Window (t_lab, t_lab + 12h]
          </text>

          {/* Time-Decay Continuous Weighting Curve λ(t) */}
          {showTimeDecayCurve && (
            <g opacity="0.65">
              <path
                d={`M ${chartBounds.timeToX(-48)} ${chartBounds.plotBottom - 20} Q ${chartBounds.timeToX(-24)} ${chartBounds.plotBottom - 45} ${tPredX} ${chartBounds.plotTop + 30}`}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2"
                strokeDasharray="4 4"
              />
              <text
                x={chartBounds.timeToX(-36)}
                y={chartBounds.plotBottom - 45}
                fill="#f59e0b"
                fontSize="9"
                fontFamily="monospace"
              >
                Learnable Decay: λ_k(Δt) = exp(-γ Δt)
              </text>
            </g>
          )}

          {/* Prediction Point t_pred (-30m) Vertical Divider */}
          <line
            x1={tPredX}
            y1={chartBounds.plotTop - 10}
            x2={tPredX}
            y2={chartBounds.plotBottom + 15}
            stroke="#f59e0b"
            strokeWidth="2"
            strokeDasharray="4 4"
          />
          <text
            x={tPredX - 70}
            y={chartBounds.plotTop + 14}
            fill="#f59e0b"
            fontSize="9.5"
            fontFamily="monospace"
            fontWeight="bold"
          >
            t_pred (-30m)
          </text>

          {/* Target Lab t_lab (0h) Divider */}
          <line
            x1={tLabX}
            y1={chartBounds.plotTop - 10}
            x2={tLabX}
            y2={chartBounds.plotBottom + 15}
            stroke="#06b6d4"
            strokeWidth="2"
          />
          <text
            x={tLabX + 5}
            y={chartBounds.plotBottom + 26}
            fill="#06b6d4"
            fontSize="9.5"
            fontFamily="monospace"
            fontWeight="bold"
          >
            t_lab (0h)
          </text>

          {/* Area Fill Under Historical Polyline */}
          <polygon
            points={`${chartBounds.timeToX(-48)},${chartBounds.plotBottom} ${polylinePoints} ${chartBounds.timeToX(trajectory[trajectory.length - 1]?.timeHoursAgo || -0.5)},${chartBounds.plotBottom}`}
            fill="url(#trajGradient)"
          />

          {/* Historical Trajectory Polyline */}
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="3.5"
            points={polylinePoints}
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Target Value Link (Dashed Line from latest to target) */}
          {trajectory.length > 0 && (
            <line
              x1={chartBounds.timeToX(trajectory[trajectory.length - 1].timeHoursAgo)}
              y1={chartBounds.valToY(trajectory[trajectory.length - 1].value)}
              x2={tLabX}
              y2={chartBounds.valToY(targetReport.value)}
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeDasharray="3 3"
            />
          )}

          {/* Projected Counterfactual Path */}
          {showCounterfactual && (
            <g>
              <line
                x1={tLabX}
                y1={chartBounds.valToY(targetReport.value)}
                x2={chartBounds.timeToX(8)}
                y2={counterfactualTargetY}
                stroke="#a855f7"
                strokeWidth="2.5"
                strokeDasharray="4 3"
              />
              <circle
                cx={chartBounds.timeToX(8)}
                cy={counterfactualTargetY}
                r="6"
                fill="#a855f7"
                stroke="#ffffff"
                strokeWidth="1.5"
              />
              <text
                x={chartBounds.timeToX(8) - 40}
                y={counterfactualTargetY - 12}
                fill="#d8b4fe"
                fontSize="9"
                fontFamily="monospace"
                fontWeight="bold"
              >
                Projected Post-Intervention
              </text>
            </g>
          )}

          {/* Historical Points (Clickable / Hoverable) */}
          {trajectory.map((pt, idx) => {
            const cx = chartBounds.timeToX(pt.timeHoursAgo);
            const cy = chartBounds.valToY(pt.value);
            const prevVal = idx > 0 ? trajectory[idx - 1].value : undefined;
            const deltaStr = prevVal !== undefined ? (pt.value - prevVal >= 0 ? `+${(pt.value - prevVal).toFixed(1)}` : (pt.value - prevVal).toFixed(1)) : undefined;

            return (
              <g
                key={idx}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHoveredPoint({
                    x: cx,
                    y: cy,
                    label: pt.label,
                    value: pt.value,
                    time: `${Math.abs(pt.timeHoursAgo)}h prior`,
                    delta: deltaStr,
                  })
                }
              >
                <circle cx={cx} cy={cy} r="8" fill="#0369a1" opacity="0.3" />
                <circle
                  cx={cx}
                  cy={cy}
                  r="5"
                  fill="#38bdf8"
                  stroke="#0f172a"
                  strokeWidth="2"
                  className="transition-transform hover:scale-125"
                />
                <text
                  x={cx}
                  y={cy - 10}
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="monospace"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {pt.value}
                </text>
              </g>
            );
          })}

          {/* Target Result Point */}
          <g
            className="cursor-pointer"
            onMouseEnter={() =>
              setHoveredPoint({
                x: tLabX,
                y: chartBounds.valToY(targetReport.value),
                label: "Target Report (t_lab)",
                value: targetReport.value,
                time: "0.0h (Target)",
                delta: `${targetReport.status}`,
              })
            }
          >
            <circle
              cx={tLabX}
              cy={chartBounds.valToY(targetReport.value)}
              r="10"
              fill="#06b6d4"
              opacity="0.3"
              className="animate-ping"
            />
            <circle
              cx={tLabX}
              cy={chartBounds.valToY(targetReport.value)}
              r="7"
              fill="#06b6d4"
              stroke="#ffffff"
              strokeWidth="2"
            />
            <text
              x={tLabX + 12}
              y={chartBounds.valToY(targetReport.value) - 6}
              fill="#22d3ee"
              fontSize="11"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Result: {targetReport.value}
            </text>
          </g>

          {/* Action Marker if documented */}
          {patient.observedAction && isPrimary && (
            <g>
              <circle
                cx={tActionX}
                cy={chartBounds.valToY(targetReport.value)}
                r="7"
                fill="#10b981"
                stroke="#ffffff"
                strokeWidth="2"
              />
              <line
                x1={tActionX}
                y1={chartBounds.valToY(targetReport.value) + 7}
                x2={tActionX}
                y2={chartBounds.plotBottom - 10}
                stroke="#10b981"
                strokeWidth="1.5"
                strokeDasharray="2 2"
              />
              <rect
                x={tActionX - 80}
                y={chartBounds.plotBottom - 45}
                width="170"
                height="38"
                fill="#0f172a"
                stroke="#10b981"
                strokeWidth="1.5"
                rx="5"
              />
              <text
                x={tActionX - 72}
                y={chartBounds.plotBottom - 30}
                fill="#34d399"
                fontSize="9.5"
                fontWeight="bold"
                fontFamily="monospace"
              >
                ACTION: +{patient.timeToAction}
              </text>
              <text
                x={tActionX - 72}
                y={chartBounds.plotBottom - 16}
                fill="#94a3b8"
                fontSize="8"
                fontFamily="monospace"
              >
                {patient.actionType.slice(0, 24)}...
              </text>
            </g>
          )}

          {/* Time Axis Labels */}
          <text x={chartBounds.timeToX(-48)} y={chartBounds.plotBottom + 16} fill="#64748b" fontSize="9" fontFamily="monospace">-48h</text>
          <text x={chartBounds.timeToX(-24)} y={chartBounds.plotBottom + 16} fill="#64748b" fontSize="9" fontFamily="monospace">-24h</text>
          <text x={chartBounds.timeToX(-12)} y={chartBounds.plotBottom + 16} fill="#64748b" fontSize="9" fontFamily="monospace">-12h</text>
          <text x={chartBounds.timeToX(6)} y={chartBounds.plotBottom + 16} fill="#64748b" fontSize="9" fontFamily="monospace">+6h</text>
          <text x={chartBounds.timeToX(12) - 25} y={chartBounds.plotBottom + 16} fill="#64748b" fontSize="9" fontFamily="monospace">+12h</text>
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div
            className="absolute z-20 pointer-events-none p-2.5 rounded-lg bg-slate-900 border border-cyan-700/80 shadow-2xl text-xs font-mono space-y-0.5 animate-in fade-in"
            style={{
              left: Math.min(hoveredPoint.x, 620),
              top: Math.max(10, hoveredPoint.y - 70),
            }}
          >
            <div className="text-cyan-300 font-bold">{hoveredPoint.label}</div>
            <div className="text-slate-200">
              Value: <span className="font-bold text-white">{hoveredPoint.value} {unit}</span>
            </div>
            <div className="text-[11px] text-slate-400">Time: {hoveredPoint.time}</div>
            {hoveredPoint.delta && (
              <div className="text-[11px] text-amber-300">Delta / Note: {hoveredPoint.delta}</div>
            )}
          </div>
        )}
      </div>

      <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between border-t border-slate-800 pt-3">
        <span className="flex items-center gap-1.5">
          <Info className="h-3.5 w-3.5 text-cyan-400" />
          Strict Temporal Enforcement: No future data past t_pred (-30m) ingested during model forward pass.
        </span>
        <span className="text-slate-500">MIMIC-IV Continuous Phlebotomy Aligned</span>
      </div>
    </div>
  );
};
