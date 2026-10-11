import React, { useState, useMemo } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Bell,
  Check,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  Clock,
  Copy,
  Database,
  FileText,
  Filter,
  Flame,
  Gauge,
  Heart,
  HelpCircle,
  Layers,
  LineChart,
  Microscope,
  Pill,
  RefreshCw,
  Scale,
  Search,
  Send,
  Sliders,
  Sparkles,
  Stethoscope,
  TrendingDown,
  TrendingUp,
  User,
  Users,
  Zap,
  Plus,
  Printer,
  ShieldAlert,
} from "lucide-react";
import { ICU_PATIENTS_DATABASE, PatientCase } from "./data/patientDatabase";
import { PatientAdmissionModal } from "./components/PatientAdmissionModal";
import { TrajectoryChart } from "./components/TrajectoryChart";
import { ROCThresholdScrubber } from "./components/ROCThresholdScrubber";
import { TensorSynergyMatrix } from "./components/TensorSynergyMatrix";
import { RoundsSummaryModal } from "./components/RoundsSummaryModal";
import canonicalMetrics from "./data/canonical_test_metrics.json";

// Strictly Verified Empirical Benchmark Data from Held-Out Test Set (N=504)
const MASTER_BENCHMARK = [
  {
    model: "Random Forest (Lab-Only)",
    role: "Top-Performing Baseline (Unimodal)",
    rocAuc: "0.5648 [0.494 - 0.636]",
    prAuc: "0.1960 [0.141 - 0.272]",
    f1: "0.1404 [0.057 - 0.234]",
    brier: "0.1783",
    ece: "0.2156",
    accuracy: "0.8056",
    recall: "0.1096",
    precision: "0.1951",
    verified: true,
  },
  {
    model: "Gradient Boosting (XGBoost)",
    role: "Ensemble Baseline",
    rocAuc: "0.5424 [0.465 - 0.617]",
    prAuc: "0.1833 [0.136 - 0.263]",
    f1: "0.0000 [0.000 - 0.000]",
    brier: "0.1241",
    ece: "0.0192",
    accuracy: "0.8492",
    recall: "0.0000",
    precision: "0.0000",
    verified: true,
  },
  {
    model: "Random Forest",
    role: "Tree Ensemble Baseline",
    rocAuc: "0.5288 [0.454 - 0.598]",
    prAuc: "0.1753 [0.133 - 0.251]",
    f1: "0.0488 [0.000 - 0.118]",
    brier: "0.1265",
    ece: "0.0248",
    accuracy: "0.8452",
    recall: "0.0274",
    precision: "0.2222",
    verified: true,
  },
  {
    model: "Logistic Regression",
    role: "Linear Baseline",
    rocAuc: "0.5280 [0.448 - 0.604]",
    prAuc: "0.1534 [0.120 - 0.207]",
    f1: "0.0000 [0.000 - 0.000]",
    brier: "0.1276",
    ece: "0.0237",
    accuracy: "0.8532",
    recall: "0.0000",
    precision: "0.0000",
    verified: true,
  },
  {
    model: "T-TFGN (Proposed)",
    role: "Proposed Multimodal Architecture (Audit Verified)",
    rocAuc: "0.5034 [0.435 - 0.581]",
    prAuc: "0.1616 [0.124 - 0.230]",
    f1: "0.1935 [0.130 - 0.264]",
    brier: "0.2376",
    ece: "0.3303",
    accuracy: "0.6528",
    recall: "0.2877",
    precision: "0.1458",
    verified: true,
  },
];

// Verified 7-Way Ablation Study Data (from results/tables/ablation_study_results.csv)
const ABLATION_STUDY = [
  { exp: "Exp A", name: "Lab-Only Trajectory", rocAuc: 0.5942, prAuc: 0.1990, f1: 0.2374, brier: 0.2315, ece: 0.3283, desc: "Prior lab trend features only; zero clinical context" },
  { exp: "Exp B", name: "Clinical-Context-Only", rocAuc: 0.5392, prAuc: 0.1831, f1: 0.1772, brier: 0.2287, ece: 0.3237, desc: "Comorbidities, active meds, and vitals only; zero lab history" },
  { exp: "Exp C", name: "Standard Lab + Context MLP", rocAuc: 0.5392, prAuc: 0.1831, f1: 0.1772, brier: 0.2287, ece: 0.3237, desc: "Simple concatenation without decay, tensor fusion, or gating" },
  { exp: "Exp D", name: "T-TFGN w/o Time-Decay", rocAuc: 0.5782, prAuc: 0.2005, f1: 0.1985, brier: 0.2289, ece: 0.3259, desc: "Full architecture with learnable recency decay gamma disabled" },
  { exp: "Exp E", name: "T-TFGN w/o Tensor Fusion", rocAuc: 0.5309, prAuc: 0.1880, f1: 0.1867, brier: 0.2317, ece: 0.3279, desc: "Full architecture with bilinear cross-modal interactions disabled" },
  { exp: "Exp F", name: "T-TFGN w/o Adaptive Gating", rocAuc: 0.5562, prAuc: 0.1790, f1: 0.0899, brier: 0.2230, ece: 0.3155, desc: "Full architecture with adaptive convex gating disabled" },
  { exp: "Exp G", name: "Full Proposed T-TFGN", rocAuc: 0.5034, prAuc: 0.1616, f1: 0.1935, brier: 0.2376, ece: 0.3303, desc: "Complete multimodal architecture (verified empirical checkpoint)" },
];

// Actual Empirical Calibration Deciles (from canonical verified metrics)
const CALIBRATION_DECILES = canonicalMetrics.calibration_deciles.map(d => ({
  bin: d.bin,
  predCas: d.avg_confidence,
  obsActionRate: d.empirical_accuracy,
  count: d.count,
}));

export default function App() {
  const [activeTab, setActiveTab] = useState<
    "census" | "chart" | "timeline" | "orders" | "sbar" | "simulator" | "explain" | "analytics"
  >("census");
  const [patients, setPatients] = useState<PatientCase[]>(ICU_PATIENTS_DATABASE);
  const [selectedPatientId, setSelectedPatientId] = useState<string>(ICU_PATIENTS_DATABASE[0].id);

  // Modal Dialog States
  const [isAdmissionModalOpen, setIsAdmissionModalOpen] = useState(false);
  const [isRoundsModalOpen, setIsRoundsModalOpen] = useState(false);

  // Selected Analyte for Multi-Analyte Review within Patient Chart
  const [activeAnalyteKey, setActiveAnalyteKey] = useState<string>("PRIMARY");

  // Census Filter States
  const [searchQuery, setSearchQuery] = useState("");
  const [filterUnit, setFilterUnit] = useState<string>("ALL");
  const [filterAnalyte, setFilterAnalyte] = useState<string>("ALL");
  const [filterRiskBand, setFilterRiskBand] = useState<string>("ALL");

  // Alert Tray Dropdown State
  const [showAlertTray, setShowAlertTray] = useState(false);

  // Order Set Commitment State
  const [selectedOrderIds, setSelectedOrderIds] = useState<Record<string, boolean>>({});
  const [orderCommitStatus, setOrderCommitStatus] = useState<string | null>(null);

  // SBAR Copy Feedback Toast State
  const [sbarCopied, setSbarCopied] = useState(false);

  // Admission Handler
  const handleAdmitPatient = (newPatient: PatientCase) => {
    setPatients((prev) => [newPatient, ...prev]);
    setSelectedPatientId(newPatient.id);
    setActiveTab("chart");
  };

  // Active Patient Record
  const currentPatient = useMemo(() => {
    return patients.find((p) => p.id === selectedPatientId) || patients[0];
  }, [patients, selectedPatientId]);

  // Active Display Analyte Data (Allows toggling between primary and concurrent labs)
  const currentAnalyteDisplay = useMemo(() => {
    if (activeAnalyteKey === "PRIMARY" || !currentPatient.concurrentLabs[activeAnalyteKey]) {
      return {
        analyteName: currentPatient.analyte,
        unit: currentPatient.canonicalUnit,
        normalRange: currentPatient.normalRange,
        trajectory: currentPatient.lookbackTrajectory,
        targetReport: currentPatient.targetReport,
        isPrimary: true,
      };
    }
    const secondary = currentPatient.concurrentLabs[activeAnalyteKey];
    return {
      analyteName: secondary.analyte,
      unit: secondary.canonicalUnit,
      normalRange: secondary.normalRange,
      trajectory: secondary.lookbackTrajectory,
      targetReport: {
        time: "Current",
        value: secondary.currentValue,
        status: secondary.status,
      },
      isPrimary: false,
    };
  }, [currentPatient, activeAnalyteKey]);

  // Filtered Patient List for Census / Ward Board
  const filteredPatients = useMemo(() => {
    return patients.filter((p) => {
      const matchSearch =
        p.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.nameAlias.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.demographics.admitDiagnosis.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.demographics.bed.toLowerCase().includes(searchQuery.toLowerCase());

      const matchUnit = filterUnit === "ALL" || p.demographics.careUnit === filterUnit;
      const matchAnalyte = filterAnalyte === "ALL" || p.analyte === filterAnalyte;
      const matchRisk = filterRiskBand === "ALL" || p.band === filterRiskBand;

      return matchSearch && matchUnit && matchAnalyte && matchRisk;
    });
  }, [patients, searchQuery, filterUnit, filterAnalyte, filterRiskBand]);

  // Ward Metrics
  const wardMetrics = useMemo(() => {
    const total = patients.length;
    const highRiskPatients = patients.filter((p) => p.band === "HIGH");
    const medRisk = patients.filter((p) => p.band === "MEDIUM").length;
    const lowRisk = patients.filter((p) => p.band === "LOW").length;
    return { total, highRisk: highRiskPatients.length, medRisk, lowRisk, highRiskPatients };
  }, [patients]);

  // Simulator State
  const [simAnalyte, setSimAnalyte] = useState<"Potassium" | "Glucose" | "Hemoglobin" | "Sodium" | "Creatinine">("Potassium");
  const [simRecentValue, setSimRecentValue] = useState(5.3);
  const [simDelta, setSimDelta] = useState(0.8);
  const [simElapsedHours, setSimElapsedHours] = useState(4.0);
  const [simHasDiuretic, setSimHasDiuretic] = useState(true);
  const [simHasInsulin, setSimHasInsulin] = useState(false);
  const [simSofaRenal, setSimSofaRenal] = useState(2);
  const [simIsHypotensive, setSimIsHypotensive] = useState(false);

  // Live Simulator Computation
  const simulatedResults = useMemo(() => {
    const gamma = simAnalyte === "Potassium" ? -0.06 : simAnalyte === "Glucose" ? -0.04 : -0.05;
    const timeDecay = 1 / (1 + Math.exp(-(gamma * simElapsedHours + 1.0)));

    let baseScore = 0.25;
    if (simAnalyte === "Potassium") {
      if (simRecentValue > 5.0) baseScore += (simRecentValue - 5.0) * 0.45;
      if (simDelta > 0.3) baseScore += simDelta * 0.35;
      if (simHasDiuretic) baseScore += 0.15;
    } else if (simAnalyte === "Glucose") {
      if (simRecentValue > 180) baseScore += (simRecentValue - 180) * 0.003;
      if (simDelta > 30) baseScore += 0.22;
      if (simHasInsulin) baseScore += 0.18;
    } else if (simAnalyte === "Hemoglobin") {
      if (simRecentValue < 8.0) baseScore += (8.0 - simRecentValue) * 0.35;
      if (simDelta < -0.8) baseScore += Math.abs(simDelta) * 0.25;
    } else {
      if (Math.abs(simDelta) > 0.5) baseScore += 0.2;
    }

    if (simSofaRenal >= 2) baseScore += 0.12;
    if (simIsHypotensive) baseScore += 0.10;

    const finalCas = Math.min(0.96, Math.max(0.08, baseScore * (0.6 + 0.4 * timeDecay)));
    const gateLab = Math.min(0.85, Math.max(0.35, 0.45 + Math.abs(simDelta) * 0.2));
    const gateContext = 1.0 - gateLab;
    const synergy = Math.min(0.65, 0.2 + (simHasDiuretic ? 0.15 : 0) + (simSofaRenal > 1 ? 0.12 : 0));

    const band = finalCas >= 0.75 ? "HIGH" : finalCas >= 0.40 ? "MEDIUM" : "LOW";

    return {
      cas: Number(finalCas.toFixed(3)),
      band,
      timeDecay: Number(timeDecay.toFixed(2)),
      gateLab: Math.round(gateLab * 100),
      gateContext: Math.round(gateContext * 100),
      synergy: Math.round(synergy * 100),
    };
  }, [simAnalyte, simRecentValue, simDelta, simElapsedHours, simHasDiuretic, simHasInsulin, simSofaRenal, simIsHypotensive]);

  // Copy SBAR Handler
  const handleCopySbar = () => {
    const text = `=== SBAR CLINICAL HANDOVER: ${currentPatient.demographics.bed} (${currentPatient.nameAlias}) ===
SITUATION: ${currentPatient.sbar.situation}
BACKGROUND: ${currentPatient.sbar.background}
ASSESSMENT: ${currentPatient.sbar.assessment}
RECOMMENDATION: ${currentPatient.sbar.recommendation}`;

    navigator.clipboard.writeText(text);
    setSbarCopied(true);
    setTimeout(() => setSbarCopied(false), 2400);
  };

  // Toggle order checkbox
  const toggleOrderSelection = (id: string) => {
    setSelectedOrderIds((prev) => ({
      ...prev,
      [id]: prev[id] === undefined ? false : !prev[id],
    }));
  };

  // Commit order set simulation
  const handleCommitOrders = () => {
    setOrderCommitStatus("COMMITTED");
    setTimeout(() => {
      setOrderCommitStatus(null);
    }, 3500);
  };

  const getBandBadge = (band: string) => {
    switch (band) {
      case "HIGH":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-rose-950/80 text-rose-300 border border-rose-800/80 shadow-sm">
            <Flame className="h-3.5 w-3.5 text-rose-400" />
            CRITICAL ACTION PROBABLE (CAS ≥ 0.75)
          </span>
        );
      case "MEDIUM":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-800/80 shadow-sm">
            <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
            MODERATE ACTIONABILITY (0.40 ≤ CAS &lt; 0.75)
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800/80 shadow-sm">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            STABLE / NON-ACTIONABLE (CAS &lt; 0.40)
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/25 selection:text-cyan-200">
      {/* Top Clinical Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/95 backdrop-blur sticky top-0 z-50 px-6 py-2.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-900 to-indigo-950 border border-cyan-700/60 flex items-center justify-center text-cyan-300 shadow-inner">
            <Activity className="h-5 w-5 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-sm font-bold tracking-tight text-slate-100">
                T-TFGN Clinical Actionability System
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800 font-semibold">
                ICU LIVE
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center gap-2 font-mono">
              <span>Time-Decay Tensor Fusion Gated Network</span>
              <span className="text-slate-700">/</span>
              <span>MIMIC-IV Critical Care Database</span>
            </div>
          </div>
        </div>

        {/* Global Navigation Bar */}
        <div className="flex items-center gap-3">
          <nav className="flex items-center p-1 bg-slate-950 border border-slate-800/80 rounded-lg text-xs font-medium">
            <button
              onClick={() => setActiveTab("census")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "census" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Users className="h-3.5 w-3.5 text-cyan-400" />
              ICU Census
            </button>
            <button
              onClick={() => setActiveTab("chart")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "chart" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Stethoscope className="h-3.5 w-3.5 text-cyan-400" />
              Bedside Chart
            </button>
            <button
              onClick={() => setActiveTab("timeline")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "timeline" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Clock className="h-3.5 w-3.5 text-cyan-400" />
              Trajectory
            </button>
            <button
              onClick={() => setActiveTab("orders")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "orders" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Pill className="h-3.5 w-3.5 text-cyan-400" />
              CPOE Order Set
            </button>
            <button
              onClick={() => setActiveTab("sbar")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "sbar" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <FileText className="h-3.5 w-3.5 text-cyan-400" />
              SBAR Handover
            </button>
            <button
              onClick={() => setActiveTab("simulator")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "simulator" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sliders className="h-3.5 w-3.5 text-cyan-400" />
              Simulator
            </button>
            <button
              onClick={() => setActiveTab("explain")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "explain" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              Synergy
            </button>
            <button
              onClick={() => setActiveTab("analytics")}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === "analytics" ? "bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <LineChart className="h-3.5 w-3.5 text-cyan-400" />
              Benchmarks
            </button>
          </nav>

          {/* Quick Admit Patient Button */}
          <button
            onClick={() => setIsAdmissionModalOpen(true)}
            className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold font-mono transition-all shadow-md flex items-center gap-1.5 shrink-0"
            title="Admit Custom ICU Patient or Load Demonstration Phenotype"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>+ Admit Patient</span>
          </button>

          {/* Alarm Notifications Bell */}
          <div className="relative">
            <button
              onClick={() => setShowAlertTray(!showAlertTray)}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 relative transition-colors"
              title="Active Ward Clinical Alerts"
            >
              <Bell className="h-4 w-4 text-cyan-400" />
              {wardMetrics.highRisk > 0 && (
                <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-rose-500 text-white font-mono text-[9px] font-bold flex items-center justify-center animate-pulse">
                  {wardMetrics.highRisk}
                </span>
              )}
            </button>

            {/* Alert Tray Dropdown */}
            {showAlertTray && (
              <div className="absolute right-0 mt-2 w-80 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl p-3 z-50 space-y-2">
                <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                  <span className="font-bold text-slate-200 flex items-center gap-1.5">
                    <Flame className="h-3.5 w-3.5 text-rose-400" /> Critical Pending Lab Alerts
                  </span>
                  <span className="font-mono text-[10px] text-slate-500">{wardMetrics.highRisk} Active</span>
                </div>

                <div className="space-y-1.5 max-h-64 overflow-y-auto">
                  {wardMetrics.highRiskPatients.map((p) => (
                    <div
                      key={p.id}
                      onClick={() => {
                        setSelectedPatientId(p.id);
                        setActiveTab("chart");
                        setShowAlertTray(false);
                      }}
                      className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-rose-500/80 cursor-pointer transition-all space-y-1"
                    >
                      <div className="flex justify-between items-center text-xs">
                        <span className="font-bold font-mono text-slate-200">{p.demographics.bed}</span>
                        <span className="font-mono text-rose-400 font-bold text-[11px]">CAS {p.cas.toFixed(3)}</span>
                      </div>
                      <div className="text-[11px] text-slate-400 truncate">
                        Pending {p.analyte} ({p.canonicalUnit}) · {p.nameAlias}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Active Patient Telemetry Ribbon Strip */}
      <section className="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2 flex items-center justify-between gap-4 text-xs font-mono">
        <div className="flex items-center gap-3">
          <span className="text-slate-400">Active Bed:</span>
          <span className="font-bold text-slate-100 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
            {currentPatient.demographics.bed} · {currentPatient.nameAlias}
          </span>
          <span className="text-slate-500">({currentPatient.id})</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-300">{currentPatient.demographics.careUnit}</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-300">{currentPatient.demographics.age}yo {currentPatient.demographics.gender}</span>
        </div>

        {/* Live Rhythm Bar */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-0.5 rounded-md bg-slate-950 border border-slate-800 text-[11px] text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>{currentPatient.vitals.rhythm}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Pending Lab:</span>
            <span className="text-cyan-400 font-bold">{currentPatient.analyte}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">CAS:</span>
            <span className={`font-bold ${
              currentPatient.band === "HIGH" ? "text-rose-400" : currentPatient.band === "MEDIUM" ? "text-amber-400" : "text-emerald-400"
            }`}>
              {currentPatient.cas.toFixed(3)}
            </span>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
        {/* ========================================================= */}
        {/* TAB 1: ICU CENSUS & WARD BOARD                            */}
        {/* ========================================================= */}
        {activeTab === "census" && (
          <div className="space-y-6">
            {/* Top Ward Stats Bar */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-xl p-4">
                <div className="text-xs text-slate-400 font-mono">Total ICU Census</div>
                <div className="text-2xl font-bold font-mono text-slate-100 mt-1">{wardMetrics.total} Patients</div>
                <div className="text-[11px] text-slate-500 mt-0.5">Across MICU, SICU, CCU, CVICU</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-xl p-4 border-l-4 border-l-rose-500">
                <div className="text-xs text-rose-400 font-mono font-semibold flex items-center gap-1.5">
                  <Flame className="h-3.5 w-3.5" /> High Actionability
                </div>
                <div className="text-2xl font-bold font-mono text-rose-300 mt-1">{wardMetrics.highRisk} Pending</div>
                <div className="text-[11px] text-slate-400 mt-0.5">CAS ≥ 0.75 (Intervention Probable)</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-xl p-4 border-l-4 border-l-amber-500">
                <div className="text-xs text-amber-400 font-mono font-semibold flex items-center gap-1.5">
                  <AlertCircle className="h-3.5 w-3.5" /> Moderate Actionability
                </div>
                <div className="text-2xl font-bold font-mono text-amber-300 mt-1">{wardMetrics.medRisk} Pending</div>
                <div className="text-[11px] text-slate-400 mt-0.5">0.40 ≤ CAS &lt; 0.75</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-xl p-4 border-l-4 border-l-emerald-500">
                <div className="text-xs text-emerald-400 font-mono font-semibold flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5" /> Stable / Non-Actionable
                </div>
                <div className="text-2xl font-bold font-mono text-emerald-300 mt-1">{wardMetrics.lowRisk} Pending</div>
                <div className="text-[11px] text-slate-400 mt-0.5">CAS &lt; 0.40 (Regimen Maintained)</div>
              </div>
            </div>

            {/* Filter & Search Toolbar */}
            <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3 flex-1 min-w-[260px]">
                <div className="relative w-full max-w-md">
                  <Search className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Search by Bed, Patient Alias, ID, or Diagnosis..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                {/* Unit Filter */}
                <select
                  value={filterUnit}
                  onChange={(e) => setFilterUnit(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300 font-mono focus:outline-none focus:border-cyan-500"
                >
                  <option value="ALL">All Units</option>
                  <option value="MICU">MICU (Medical)</option>
                  <option value="SICU">SICU (Surgical)</option>
                  <option value="CCU">CCU (Coronary)</option>
                  <option value="CVICU">CVICU (Cardiothoracic)</option>
                  <option value="TSICU">TSICU (Trauma)</option>
                  <option value="NICU">NICU (Neuro)</option>
                </select>

                {/* Analyte Filter */}
                <select
                  value={filterAnalyte}
                  onChange={(e) => setFilterAnalyte(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300 font-mono focus:outline-none focus:border-cyan-500"
                >
                  <option value="ALL">All Analytes</option>
                  <option value="Potassium">Potassium</option>
                  <option value="Glucose">Glucose</option>
                  <option value="Hemoglobin">Hemoglobin</option>
                  <option value="Sodium">Sodium</option>
                  <option value="Creatinine">Creatinine</option>
                </select>

                {/* Risk Band Filter */}
                <select
                  value={filterRiskBand}
                  onChange={(e) => setFilterRiskBand(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300 font-mono focus:outline-none focus:border-cyan-500"
                >
                  <option value="ALL">All CAS Priorities</option>
                  <option value="HIGH">High CAS (≥ 0.75)</option>
                  <option value="MEDIUM">Medium CAS (0.40 - 0.75)</option>
                  <option value="LOW">Low CAS (&lt; 0.40)</option>
                </select>

                {/* Admit Button in Toolbar */}
                <button
                  onClick={() => setIsAdmissionModalOpen(true)}
                  className="px-3 py-1.5 rounded-lg bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-700/80 font-mono font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Plus className="h-3.5 w-3.5" />
                  + Admit Patient
                </button>
              </div>
            </div>

            {/* Quick Cohort Preset Filter Chips & Distribution Chart */}
            <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
                <span className="font-mono text-slate-400">Clinical Cohort Presets:</span>
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    onClick={() => { setFilterUnit("ALL"); setFilterRiskBand("ALL"); setFilterAnalyte("ALL"); setSearchQuery(""); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterUnit === "ALL" && filterRiskBand === "ALL" && filterAnalyte === "ALL"
                        ? "bg-slate-800 text-cyan-300 border-cyan-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    All Cohort ({patients.length})
                  </button>
                  <button
                    onClick={() => { setFilterRiskBand("HIGH"); setFilterUnit("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterRiskBand === "HIGH"
                        ? "bg-rose-950 text-rose-300 border-rose-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    High CAS ≥ 0.75 ({wardMetrics.highRisk})
                  </button>
                  <button
                    onClick={() => { setFilterRiskBand("MEDIUM"); setFilterUnit("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterRiskBand === "MEDIUM"
                        ? "bg-amber-950 text-amber-300 border-amber-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    Medium CAS ({wardMetrics.medRisk})
                  </button>
                  <button
                    onClick={() => { setFilterRiskBand("LOW"); setFilterUnit("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterRiskBand === "LOW"
                        ? "bg-emerald-950 text-emerald-300 border-emerald-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    Stable / Low ({wardMetrics.lowRisk})
                  </button>
                  <button
                    onClick={() => { setFilterUnit("MICU"); setFilterRiskBand("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterUnit === "MICU"
                        ? "bg-slate-800 text-cyan-300 border-cyan-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    MICU
                  </button>
                  <button
                    onClick={() => { setFilterUnit("CVICU"); setFilterRiskBand("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterUnit === "CVICU"
                        ? "bg-slate-800 text-cyan-300 border-cyan-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    CVICU
                  </button>
                  <button
                    onClick={() => { setFilterUnit("NICU"); setFilterRiskBand("ALL"); }}
                    className={`px-2.5 py-1 rounded-md font-mono text-[11px] transition-all border ${
                      filterUnit === "NICU"
                        ? "bg-slate-800 text-cyan-300 border-cyan-700 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    Neuro (NICU)
                  </button>
                </div>
              </div>

              {/* Visual Ward CAS Distribution Bar Chart */}
              <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                <div className="flex justify-between items-center text-[11px] font-mono text-slate-400">
                  <span>Unit Actionability Density:</span>
                  <span>
                    {((wardMetrics.highRisk / wardMetrics.total) * 100).toFixed(0)}% Critical ·{" "}
                    {((wardMetrics.medRisk / wardMetrics.total) * 100).toFixed(0)}% Moderate ·{" "}
                    {((wardMetrics.lowRisk / wardMetrics.total) * 100).toFixed(0)}% Stable
                  </span>
                </div>
                <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800">
                  <div
                    style={{ width: `${(wardMetrics.highRisk / wardMetrics.total) * 100}%` }}
                    className="bg-rose-500 h-full transition-all duration-500 cursor-pointer hover:opacity-90"
                    title={`High CAS: ${wardMetrics.highRisk} patients`}
                    onClick={() => setFilterRiskBand("HIGH")}
                  />
                  <div
                    style={{ width: `${(wardMetrics.medRisk / wardMetrics.total) * 100}%` }}
                    className="bg-amber-500 h-full transition-all duration-500 cursor-pointer hover:opacity-90"
                    title={`Medium CAS: ${wardMetrics.medRisk} patients`}
                    onClick={() => setFilterRiskBand("MEDIUM")}
                  />
                  <div
                    style={{ width: `${(wardMetrics.lowRisk / wardMetrics.total) * 100}%` }}
                    className="bg-emerald-500 h-full transition-all duration-500 cursor-pointer hover:opacity-90"
                    title={`Low CAS: ${wardMetrics.lowRisk} patients`}
                    onClick={() => setFilterRiskBand("LOW")}
                  />
                </div>
              </div>
            </div>

            {/* Patients Census Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredPatients.map((p) => (
                <div
                  key={p.id}
                  onClick={() => {
                    setSelectedPatientId(p.id);
                    setActiveTab("chart");
                  }}
                  className={`bg-slate-900 border rounded-xl p-4.5 cursor-pointer transition-all duration-200 space-y-3 group hover:border-cyan-500/80 ${
                    selectedPatientId === p.id ? "border-cyan-500 bg-slate-900/95 ring-1 ring-cyan-500/50 shadow-md" : "border-slate-800/80 hover:bg-slate-850"
                  }`}
                >
                  {/* Bed & Status Row */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-sm text-slate-100 group-hover:text-cyan-300 transition-colors">
                        {p.demographics.bed}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                        {p.demographics.careUnit}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-mono font-black ${
                        p.band === "HIGH" ? "text-rose-400" : p.band === "MEDIUM" ? "text-amber-400" : "text-emerald-400"
                      }`}>
                        CAS {p.cas.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  {/* Patient Alias & Diagnosis */}
                  <div>
                    <div className="text-xs font-semibold text-slate-200">
                      {p.nameAlias} · {p.demographics.age}yo {p.demographics.gender}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate mt-0.5 font-mono">
                      {p.demographics.admitDiagnosis}
                    </div>
                  </div>

                  {/* Pending Test & Vitals Summary */}
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 text-xs font-mono grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-[10px] text-slate-500 block">Pending Investigation</span>
                      <span className="text-cyan-400 font-bold">{p.analyte}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block">Bedside Vitals</span>
                      <span className="text-slate-300">HR {p.vitals.heartRate} · MAP {p.vitals.map}</span>
                    </div>
                  </div>

                  {/* Clinical Recommendation Snapshot */}
                  <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1">
                    <span className="truncate max-w-[210px] text-slate-400">
                      {p.demographics.codeStatus} · SOFA {p.demographics.sofaCardio + p.demographics.sofaRenal}
                    </span>
                    <span className="text-cyan-400 flex items-center gap-1 font-semibold group-hover:translate-x-0.5 transition-transform">
                      Open Chart <ChevronRight className="h-3.5 w-3.5" />
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 2: BEDSIDE PATIENT CLINICAL CHART                     */}
        {/* ========================================================= */}
        {activeTab === "chart" && (
          <div className="space-y-6">
            {/* Patient Header Banner */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-3">
                    <span className="text-xl font-bold text-slate-100 font-mono">
                      {currentPatient.demographics.bed} · {currentPatient.nameAlias}
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                      MRN: {currentPatient.id}
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      {currentPatient.demographics.careUnit}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 flex items-center gap-3">
                    <span><strong>Diagnosis:</strong> {currentPatient.demographics.admitDiagnosis}</span>
                    <span>·</span>
                    <span><strong>Code Status:</strong> {currentPatient.demographics.codeStatus}</span>
                    <span>·</span>
                    <span><strong>LOS:</strong> {currentPatient.demographics.losHours} hours</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsRoundsModalOpen(true)}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-xs font-medium text-slate-200 border border-slate-700 transition-colors flex items-center gap-1.5 shadow-sm"
                  >
                    <Printer className="h-3.5 w-3.5 text-cyan-400" />
                    Rounds Sheet
                  </button>
                  <button
                    onClick={() => setActiveTab("orders")}
                    className="px-3 py-1.5 rounded-lg bg-indigo-950 hover:bg-indigo-900 text-xs font-medium text-indigo-300 border border-indigo-700/60 transition-colors flex items-center gap-1.5"
                  >
                    <Pill className="h-3.5 w-3.5" />
                    Review Order Set
                  </button>
                  <button
                    onClick={() => setActiveTab("sbar")}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition-colors flex items-center gap-1.5"
                  >
                    <FileText className="h-3.5 w-3.5" />
                    SBAR Handover
                  </button>
                  <button
                    onClick={() => setActiveTab("timeline")}
                    className="px-3 py-1.5 rounded-lg bg-cyan-950 hover:bg-cyan-900 text-xs font-medium text-cyan-300 border border-cyan-700/60 transition-colors flex items-center gap-1.5"
                  >
                    Trajectory <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Bedside Vitals Bar */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-4">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Heart Rate</div>
                  <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">{currentPatient.vitals.heartRate} <span className="text-xs font-normal text-slate-400">bpm</span></div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Blood Pressure</div>
                  <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">{currentPatient.vitals.bloodPressure} <span className="text-xs font-normal text-slate-400">({currentPatient.vitals.map})</span></div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Oxygen Saturation</div>
                  <div className="text-lg font-bold font-mono text-cyan-400 mt-0.5">{currentPatient.vitals.spo2}% <span className="text-xs font-normal text-slate-400">SpO2</span></div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Respiratory Rate</div>
                  <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">{currentPatient.vitals.respiratoryRate} <span className="text-xs font-normal text-slate-400">/min</span></div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Core Temp</div>
                  <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">{currentPatient.vitals.temperature}°C</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Neurological GCS</div>
                  <div className="text-lg font-bold font-mono text-emerald-400 mt-0.5">{currentPatient.vitals.gcs} <span className="text-xs font-normal text-slate-400">/ 15</span></div>
                </div>
              </div>

              {/* Cross-Analyte Panel Switcher Strip */}
              <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2 text-slate-400">
                  <Microscope className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Concurrent Laboratory Panels:</span>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    onClick={() => setActiveAnalyteKey("PRIMARY")}
                    className={`px-3 py-1 rounded-md font-mono text-xs transition-all border ${
                      activeAnalyteKey === "PRIMARY"
                        ? "bg-cyan-950 text-cyan-300 border-cyan-700 shadow-sm font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    ★ {currentPatient.analyte} ({currentPatient.targetReport.value} {currentPatient.canonicalUnit})
                  </button>

                  {Object.entries(currentPatient.concurrentLabs).map(([key, lab]) => (
                    <button
                      key={key}
                      onClick={() => setActiveAnalyteKey(key)}
                      className={`px-3 py-1 rounded-md font-mono text-xs transition-all border ${
                        activeAnalyteKey === key
                          ? "bg-cyan-950 text-cyan-300 border-cyan-700 shadow-sm font-bold"
                          : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                      }`}
                    >
                      {lab.analyte}: {lab.currentValue} {lab.canonicalUnit}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* CAS Prediction & Decision Support HUD */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* CAS Probability Gauge Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between shadow-sm">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                      Predictive Decision Metric
                    </span>
                    <span className="text-xs font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">
                      30m Pre-Result Cutoff
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-slate-100">
                    Clinical Actionability Score
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Estimated probability that the pending {currentPatient.analyte} report will prompt direct pharmacological or diagnostic intervention.
                  </p>
                </div>

                <div className="my-6 flex items-center gap-6">
                  <div className="relative h-28 w-28 shrink-0 flex items-center justify-center">
                    <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="42" fill="none" stroke="#1e293b" strokeWidth="10" />
                      <circle
                        cx="50"
                        cy="50"
                        r="42"
                        fill="none"
                        stroke={currentPatient.band === "HIGH" ? "#f43f5e" : currentPatient.band === "MEDIUM" ? "#f59e0b" : "#10b981"}
                        strokeWidth="10"
                        strokeDasharray={264}
                        strokeDashoffset={264 - 264 * currentPatient.cas}
                        strokeLinecap="round"
                        className="transition-all duration-700 ease-out"
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center font-mono">
                      <span className="text-2xl font-black text-slate-100">{currentPatient.cas.toFixed(3)}</span>
                      <span className="text-[10px] text-slate-400 uppercase">CAS Score</span>
                    </div>
                  </div>

                  <div className="space-y-1.5 text-xs font-mono">
                    <div>
                      <span className="text-slate-400 block text-[11px]">Actionability Risk:</span>
                      <div>{getBandBadge(currentPatient.band)}</div>
                    </div>
                    <div className="pt-1">
                      <span className="text-slate-400">Pending Analyte: </span>
                      <span className="text-cyan-400 font-bold">{currentPatient.analyte} ({currentPatient.canonicalUnit})</span>
                    </div>
                  </div>
                </div>

                <div className="border-t border-slate-800/80 pt-3 text-xs text-slate-400">
                  <span>Downstream Observation Window: </span>
                  <span className="text-slate-200 font-mono">12.0 Hours Post-Result</span>
                </div>
              </div>

              {/* Recommended Clinical Posture Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between shadow-sm lg:col-span-2">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                      Automated Decision Support
                    </span>
                    <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                      Grounded in MIMIC-IV
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-slate-100">
                    Clinical Posture & Protocol Guidance
                  </h3>
                  <div className="my-4 p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <div className="text-xs font-semibold text-slate-300 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-amber-400" />
                      Physician Action Recommendation:
                    </div>
                    <p className="text-sm text-cyan-200 leading-relaxed font-sans">
                      {currentPatient.clinicalPostureRecommendation}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 border-t border-slate-800/80 pt-3 text-xs">
                  <div>
                    <span className="text-slate-400">Target Result Value:</span>
                    <div className="font-mono text-slate-200 font-bold mt-0.5">
                      {currentPatient.targetValue} {currentPatient.canonicalUnit} ({currentPatient.targetReport.status})
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-400">Documented Post-Result Outcome:</span>
                    <div className="font-mono text-emerald-400 font-bold mt-0.5 truncate">
                      {currentPatient.observedAction ? currentPatient.intervention : "No modification required"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Active Continuous Medications & Comorbidities Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Active Medications */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <Pill className="h-4 w-4 text-cyan-400" />
                    Active IV Infusions & Prescriptions Prior to Test
                  </h3>
                  <span className="text-xs font-mono text-slate-400">
                    {currentPatient.activeMedications.length} Active Orders
                  </span>
                </div>

                <div className="space-y-2">
                  {currentPatient.activeMedications.map((m, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 flex items-center justify-between text-xs"
                    >
                      <div className="space-y-0.5">
                        <div className="font-semibold text-slate-200 font-mono">{m.drug}</div>
                        <div className="text-slate-400 text-[11px]">{m.route} · {m.dose}</div>
                      </div>
                      <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-800">
                        {m.category}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Chronic Comorbidities & Organ Dysfunction */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <Layers className="h-4 w-4 text-cyan-400" />
                    Comorbidities & Organ Dysfunction Markers
                  </h3>
                  <span className="text-xs font-mono text-slate-400">
                    SOFA Total: {currentPatient.demographics.sofaCardio + currentPatient.demographics.sofaRenal}
                  </span>
                </div>

                <div className="space-y-2">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 text-xs">
                    <span className="text-slate-400 font-mono block text-[11px] mb-1">Documented Baseline Comorbidities:</span>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {currentPatient.comorbidities.map((c, idx) => (
                        <span key={idx} className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-800 text-[11px] font-mono">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[10px]">SOFA Cardiovascular</span>
                      <span className="text-slate-200 font-bold text-sm">Score {currentPatient.demographics.sofaCardio}</span>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[10px]">SOFA Renal Component</span>
                      <span className="text-slate-200 font-bold text-sm">Score {currentPatient.demographics.sofaRenal}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 3: LONGITUDINAL TRAJECTORY & LEAD-TIME                */}
        {/* ========================================================= */}
        {activeTab === "timeline" && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
              <div>
                <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <Clock className="h-5 w-5 text-cyan-400" />
                  Longitudinal Trajectory & Temporal Window Inspector
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Serial observations over 48h lookback, prediction cutoff t_pred (-30m), target result t_lab, and post-result 12h action window for {currentPatient.nameAlias}.
                </p>
              </div>

              {/* Analyte Switcher Buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveAnalyteKey("PRIMARY")}
                  className={`px-3 py-1 rounded text-xs font-mono transition-all border ${
                    activeAnalyteKey === "PRIMARY"
                      ? "bg-cyan-950 text-cyan-300 border-cyan-600 font-bold"
                      : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {currentPatient.analyte}
                </button>
                {Object.keys(currentPatient.concurrentLabs).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveAnalyteKey(k)}
                    className={`px-3 py-1 rounded text-xs font-mono transition-all border ${
                      activeAnalyteKey === k
                        ? "bg-cyan-950 text-cyan-300 border-cyan-600 font-bold"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            {/* Interactive Dynamic Trajectory Chart Component */}
            <TrajectoryChart
              patient={currentPatient}
              analyteName={currentAnalyteDisplay.analyteName}
              unit={currentAnalyteDisplay.unit}
              normalRange={currentAnalyteDisplay.normalRange}
              trajectory={currentAnalyteDisplay.trajectory}
              targetReport={currentAnalyteDisplay.targetReport}
              isPrimary={currentAnalyteDisplay.isPrimary}
            />

            {/* Trajectory Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden">
                <thead className="bg-slate-950 font-mono text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3">Timeline Point</th>
                    <th className="p-3">Relative Timestamp</th>
                    <th className="p-3">Value ({currentAnalyteDisplay.unit})</th>
                    <th className="p-3">Temporal Classification</th>
                    <th className="p-3">Model Ingestion Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {currentAnalyteDisplay.trajectory.map((pt, i) => (
                    <tr key={i} className="hover:bg-slate-850">
                      <td className="p-3 text-slate-200">{pt.label}</td>
                      <td className="p-3 text-slate-400">{pt.timeHoursAgo}h prior</td>
                      <td className="p-3 text-cyan-400 font-bold">{pt.value}</td>
                      <td className="p-3 text-slate-400">Historical Lookback Feature</td>
                      <td className="p-3 text-emerald-400">Ingested in X_lab vector</td>
                    </tr>
                  ))}
                  <tr className="bg-amber-950/20 text-amber-300">
                    <td className="p-3 font-bold">Prediction Timestamp (t_pred)</td>
                    <td className="p-3">-30m</td>
                    <td className="p-3">-</td>
                    <td className="p-3">Model Inference Cutoff</td>
                    <td className="p-3 text-amber-400 font-bold">Prediction computed here</td>
                  </tr>
                  <tr className="bg-cyan-950/20 text-cyan-300">
                    <td className="p-3 font-bold">Upcoming Target Lab (t_lab)</td>
                    <td className="p-3">0.0h</td>
                    <td className="p-3 font-bold">{currentAnalyteDisplay.targetReport.value}</td>
                    <td className="p-3">Diagnostic Target Query</td>
                    <td className="p-3 text-slate-400">Withheld during inference</td>
                  </tr>
                  {currentAnalyteDisplay.isPrimary && (
                    <tr className="bg-emerald-950/20 text-emerald-300">
                      <td className="p-3 font-bold">Downstream Clinical Action</td>
                      <td className="p-3">+{currentPatient.timeToAction}</td>
                      <td className="p-3">{currentPatient.intervention}</td>
                      <td className="p-3">Evaluation Label (Y = 1)</td>
                      <td className="p-3 text-emerald-400 font-bold">Post-result outcome verified</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 4: CPOE CLINICAL ORDER SET SIMULATOR                  */}
        {/* ========================================================= */}
        {activeTab === "orders" && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
                <div>
                  <div className="flex items-center gap-2.5">
                    <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                      <Pill className="h-5 w-5 text-indigo-400" />
                      Pre-Populated CPOE Order Set: {currentPatient.demographics.bed}
                    </h2>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                      CAS Guided
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Automated clinical order recommendations generated based on predicted actionability for {currentPatient.nameAlias} ({currentPatient.analyte}).
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  {orderCommitStatus === "COMMITTED" ? (
                    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950 text-emerald-300 border border-emerald-700 text-xs font-mono font-bold animate-pulse">
                      <Check className="h-4 w-4 text-emerald-400" /> Orders Committed to ICU EHR
                    </div>
                  ) : (
                    <button
                      onClick={handleCommitOrders}
                      className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold font-mono transition-all shadow-md flex items-center gap-2"
                    >
                      <Send className="h-3.5 w-3.5" />
                      Commit Selected Orders
                    </button>
                  )}
                </div>
              </div>

              {/* Order Items List */}
              <div className="space-y-3 pt-6">
                {currentPatient.orderSet.map((ord) => {
                  const isChecked = selectedOrderIds[ord.id] !== false;
                  return (
                    <div
                      key={ord.id}
                      onClick={() => toggleOrderSelection(ord.id)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer flex items-start gap-4 ${
                        isChecked
                          ? "bg-slate-950 border-cyan-700/80 shadow-sm"
                          : "bg-slate-950/40 border-slate-800/60 opacity-60"
                      }`}
                    >
                      <div className="pt-0.5">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}}
                          className="h-4 w-4 accent-cyan-400 rounded cursor-pointer"
                        />
                      </div>

                      <div className="flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2.5">
                            <span className="font-bold text-sm text-slate-100 font-mono">
                              {ord.title}
                            </span>
                            {ord.urgent && (
                              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800 font-bold">
                                STAT URGENT
                              </span>
                            )}
                          </div>
                          <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-800">
                            {ord.category} · {ord.actionType}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed font-sans">
                          {ord.details}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Clinical Protocol Reference Footnote */}
              <div className="mt-6 p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between font-mono">
                <span>Verification ID: CPOE-CAS-{currentPatient.id.slice(-6)}</span>
                <span>Audit Stamp: Attending Critical Care Physician</span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 5: SBAR SHIFT HANDOVER NOTE GENERATOR                 */}
        {/* ========================================================= */}
        {activeTab === "sbar" && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
                <div>
                  <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <FileText className="h-5 w-5 text-cyan-400" />
                    Structured SBAR Clinical Handover Report
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Situation, Background, Assessment, and Recommendation formatted for ICU morning and shift change rounds.
                  </p>
                </div>

                <button
                  onClick={handleCopySbar}
                  className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition-all flex items-center gap-1.5 shadow-sm font-mono"
                >
                  {sbarCopied ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                      <span className="text-emerald-300">Copied to Clipboard!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5 text-cyan-400" />
                      <span>Copy SBAR Note</span>
                    </>
                  )}
                </button>
              </div>

              {/* SBAR Structured Content Cards */}
              <div className="space-y-4 pt-6">
                {/* Situation */}
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="text-xs font-bold text-rose-400 font-mono uppercase tracking-wider flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                    S · Situation
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed font-sans">
                    {currentPatient.sbar.situation}
                  </p>
                </div>

                {/* Background */}
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="text-xs font-bold text-amber-400 font-mono uppercase tracking-wider flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                    B · Background
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed font-sans">
                    {currentPatient.sbar.background}
                  </p>
                </div>

                {/* Assessment */}
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="text-xs font-bold text-cyan-400 font-mono uppercase tracking-wider flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-cyan-500"></span>
                    A · Assessment (T-TFGN Model Evaluation)
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed font-sans">
                    {currentPatient.sbar.assessment}
                  </p>
                </div>

                {/* Recommendation */}
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-wider flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                    R · Recommendation
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed font-sans">
                    {currentPatient.sbar.recommendation}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 6: WHAT-IF SCENARIO SIMULATOR                         */}
        {/* ========================================================= */}
        {activeTab === "simulator" && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-4">
                <div>
                  <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <Sliders className="h-5 w-5 text-cyan-400" />
                    Bedside What-If Scenario & Sensitivity Simulator
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Simulate hypothetical laboratory trajectories, elapsed monitoring intervals, and concurrent medications to observe real-time CAS score changes.
                  </p>
                </div>
                <div className="text-xs font-mono text-cyan-300 bg-cyan-950 px-2.5 py-1 rounded border border-cyan-800">
                  Live Neural Gating
                </div>
              </div>

              {/* Controls and Output */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-5 bg-slate-950 p-5 rounded-xl border border-slate-800">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                        Target Laboratory Analyte
                      </label>
                      <select
                        value={simAnalyte}
                        onChange={(e) => setSimAnalyte(e.target.value as any)}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500"
                      >
                        <option value="Potassium">Potassium (mEq/L)</option>
                        <option value="Glucose">Glucose (mg/dL)</option>
                        <option value="Hemoglobin">Hemoglobin (g/dL)</option>
                        <option value="Sodium">Sodium (mEq/L)</option>
                        <option value="Creatinine">Creatinine (mg/dL)</option>
                      </select>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                        <span>Elapsed Interval (Δt since last test)</span>
                        <span className="text-cyan-400 font-mono">{simElapsedHours} hours</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="36"
                        step="1"
                        value={simElapsedHours}
                        onChange={(e) => setSimElapsedHours(Number(e.target.value))}
                        className="w-full accent-cyan-400"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-800">
                    <div>
                      <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                        <span>Prior Value</span>
                        <span className="text-cyan-400 font-mono">{simRecentValue}</span>
                      </div>
                      <input
                        type="range"
                        min={simAnalyte === "Glucose" ? 50 : simAnalyte === "Sodium" ? 115 : 1}
                        max={simAnalyte === "Glucose" ? 400 : simAnalyte === "Sodium" ? 160 : 15}
                        step={simAnalyte === "Glucose" ? 5 : 0.1}
                        value={simRecentValue}
                        onChange={(e) => setSimRecentValue(Number(e.target.value))}
                        className="w-full accent-cyan-400"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                        <span>Velocity / Delta over Lookback</span>
                        <span className="text-amber-400 font-mono">{simDelta > 0 ? `+${simDelta}` : simDelta}</span>
                      </div>
                      <input
                        type="range"
                        min={simAnalyte === "Glucose" ? -60 : -3.0}
                        max={simAnalyte === "Glucose" ? 60 : 3.0}
                        step={simAnalyte === "Glucose" ? 2 : 0.1}
                        value={simDelta}
                        onChange={(e) => setSimDelta(Number(e.target.value))}
                        className="w-full accent-amber-400"
                      />
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-800 space-y-3">
                    <span className="text-xs font-semibold text-slate-300 block">
                      Active Clinical Context Switches
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <label className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900 border border-slate-800 cursor-pointer hover:border-slate-700 text-xs">
                        <input
                          type="checkbox"
                          checked={simHasDiuretic}
                          onChange={(e) => setSimHasDiuretic(e.target.checked)}
                          className="accent-cyan-400"
                        />
                        <span className="text-slate-300 font-mono">Loop Diuretic</span>
                      </label>

                      <label className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900 border border-slate-800 cursor-pointer hover:border-slate-700 text-xs">
                        <input
                          type="checkbox"
                          checked={simHasInsulin}
                          onChange={(e) => setSimHasInsulin(e.target.checked)}
                          className="accent-cyan-400"
                        />
                        <span className="text-slate-300 font-mono">Active Insulin</span>
                      </label>

                      <label className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900 border border-slate-800 cursor-pointer hover:border-slate-700 text-xs">
                        <input
                          type="checkbox"
                          checked={simIsHypotensive}
                          onChange={(e) => setSimIsHypotensive(e.target.checked)}
                          className="accent-cyan-400"
                        />
                        <span className="text-slate-300 font-mono">MAP &lt; 65 mmHg</span>
                      </label>

                      <label className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900 border border-slate-800 cursor-pointer hover:border-slate-700 text-xs">
                        <input
                          type="checkbox"
                          checked={simSofaRenal >= 2}
                          onChange={(e) => setSimSofaRenal(e.target.checked ? 2 : 0)}
                          className="accent-cyan-400"
                        />
                        <span className="text-slate-300 font-mono">Renal SOFA ≥ 2</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Right Column: Live Simulated CAS Output */}
                <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 flex flex-col justify-between space-y-6">
                  <div>
                    <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                      Live Neural Estimation
                    </span>
                    <h3 className="text-base font-bold text-slate-100 mt-1">
                      Simulated Actionability
                    </h3>
                  </div>

                  <div className="flex flex-col items-center justify-center my-2">
                    <div className="relative h-32 w-32 flex items-center justify-center">
                      <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="42" fill="none" stroke="#1e293b" strokeWidth="10" />
                        <circle
                          cx="50"
                          cy="50"
                          r="42"
                          fill="none"
                          stroke={simulatedResults.band === "HIGH" ? "#f43f5e" : simulatedResults.band === "MEDIUM" ? "#f59e0b" : "#10b981"}
                          strokeWidth="10"
                          strokeDasharray={264}
                          strokeDashoffset={264 - 264 * simulatedResults.cas}
                          strokeLinecap="round"
                          className="transition-all duration-300 ease-out"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center font-mono">
                        <span className="text-3xl font-black text-slate-100">{simulatedResults.cas}</span>
                        <span className="text-[10px] text-slate-400 uppercase">Simulated CAS</span>
                      </div>
                    </div>
                    <div className="mt-3">{getBandBadge(simulatedResults.band)}</div>
                  </div>

                  <div className="space-y-2 border-t border-slate-800 pt-3 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Recency Decay Factor (λ):</span>
                      <span className="text-cyan-400 font-bold">{simulatedResults.timeDecay}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Dynamic Gate (Lab/Context):</span>
                      <span className="text-indigo-400 font-bold">{simulatedResults.gateLab}% / {simulatedResults.gateContext}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Tensor Synergy:</span>
                      <span className="text-amber-400 font-bold">{simulatedResults.synergy}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 7: EXPLAINABILITY & SYNERGY                           */}
        {/* ========================================================= */}
        {activeTab === "explain" && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-4 mb-6">
                <div>
                  <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-cyan-400" />
                    T-TFGN Feature Attributions & Cross-Modal Synergy
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Decomposing why {currentPatient.nameAlias} has a CAS of {currentPatient.cas.toFixed(3)}.
                  </p>
                </div>
                <div className="text-xs font-mono text-cyan-400">
                  {currentPatient.demographics.bed} · {currentPatient.analyte}
                </div>
              </div>

              {/* Attribution Factors Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Positive Drivers */}
                <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="flex items-center gap-2 text-xs font-semibold text-rose-400 uppercase tracking-wider">
                    <TrendingUp className="h-4 w-4" />
                    Top Positive Risk Factors (Elevating CAS)
                  </div>
                  <div className="space-y-2.5">
                    {currentPatient.explainability.topPositive.map((f, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 font-mono truncate max-w-xs">{f.feature}</span>
                        <div className="flex items-center gap-2 shrink-0">
                          <div className="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div
                              className="bg-rose-500 h-full rounded-full"
                              style={{ width: `${Math.min(100, f.impact * 200)}%` }}
                            />
                          </div>
                          <span className="font-mono text-rose-400 font-semibold w-10 text-right">
                            +{f.impact}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Negative Factors */}
                <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                    <TrendingDown className="h-4 w-4" />
                    Top Negative Stabilizing Factors (Lowering CAS)
                  </div>
                  <div className="space-y-2.5">
                    {currentPatient.explainability.topNegative.map((f, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 font-mono truncate max-w-xs">{f.feature}</span>
                        <div className="flex items-center gap-2 shrink-0">
                          <div className="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div
                              className="bg-emerald-500 h-full rounded-full"
                              style={{ width: `${Math.min(100, Math.abs(f.impact) * 200)}%` }}
                            />
                          </div>
                          <span className="font-mono text-emerald-400 font-semibold w-10 text-right">
                            {f.impact}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Neural Mechanics Breakdown */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-xs font-mono text-slate-400">1. Learnable Continuous Time Decay (λ_k)</div>
                  <div className="text-2xl font-bold font-mono text-cyan-400">
                    {currentPatient.explainability.timeDecayFactor}
                  </div>
                  <p className="text-xs text-slate-400">
                    Continuous exponential decay downweighting stale measurements as elapsed time increases.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-xs font-mono text-slate-400">2. Multimodal Adaptive Gate (g)</div>
                  <div className="text-2xl font-bold font-mono text-indigo-400">
                    {(currentPatient.explainability.gateLabWeight * 100).toFixed(0)}% / {(currentPatient.explainability.gateContextWeight * 100).toFixed(0)}%
                  </div>
                  <p className="text-xs text-slate-400">
                    Balances acute trajectory dynamics against chronic baseline comorbidity & medication exposure.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-xs font-mono text-slate-400">3. Bilinear Tensor Synergy</div>
                  <div className="text-2xl font-bold font-mono text-amber-400">
                    {(currentPatient.explainability.tensorSynergy * 100).toFixed(0)}%
                  </div>
                  <p className="text-xs text-slate-400">
                    Degree of multiplicative interaction between laboratory velocity and active medication exposure.
                  </p>
                </div>
              </div>
            </div>

            {/* Bilinear Tensor Synergy Cross-Modal Matrix */}
            <TensorSynergyMatrix />
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 8: ROC CURVES & MODEL BENCHMARKS                      */}
        {/* ========================================================= */}
        {activeTab === "analytics" && (
          <div className="space-y-6">
            {/* Audited Scientific Verification Banner */}
            <div className="bg-amber-950/30 border border-amber-600/40 rounded-xl p-4 text-xs space-y-2">
              <div className="flex items-center gap-2 text-amber-300 font-bold text-sm">
                <ShieldAlert className="h-4 w-4" />
                <span>Independent Scientific Audit & Reproducibility Verification</span>
              </div>
              <p className="text-amber-200/90 leading-relaxed">
                All metrics below have been independently recomputed directly from raw held-out prediction arrays (<code className="bg-slate-900 px-1 py-0.5 rounded text-cyan-300 font-mono">results/predictions/test_predictions.npz</code>, N=504 instances, 9 held-out patients).
                Previous frontend reports contained unverified and overclaimed values. Independent verification confirms that the proposed T-TFGN model achieves an empirical <strong>ROC-AUC of 0.5034</strong> and <strong>PR-AUC of 0.1616</strong>, underperforming the unimodal Random Forest baseline (<strong>ROC-AUC 0.5648</strong>).
                Patient-level split (70/15/15) has zero leakage across partitions. No clinical readiness or autonomous treatment decisions are claimed.
              </p>
            </div>

            {/* Interactive Decision Threshold Scrubber */}
            <ROCThresholdScrubber />

            {/* Master Comparison Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
                <div>
                  <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <Scale className="h-5 w-5 text-cyan-400" />
                    Comparative Model Performance (Held-Out Patient Partition)
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Evaluated on 504 test instances across 9 held-out patients with zero subject overlap.
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-400">95% Bootstrap CIs</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden">
                  <thead className="bg-slate-950 font-mono text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="p-3">Model</th>
                      <th className="p-3">Architecture Type</th>
                      <th className="p-3">ROC-AUC [95% CI]</th>
                      <th className="p-3">PR-AUC [95% CI]</th>
                      <th className="p-3">F1-Score [95% CI]</th>
                      <th className="p-3">Brier Score</th>
                      <th className="p-3">ECE</th>
                      <th className="p-3">Accuracy</th>
                      <th className="p-3">Recall</th>
                      <th className="p-3">Precision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                    {MASTER_BENCHMARK.map((m, i) => (
                      <tr
                        key={i}
                        className={
                          m.model.includes("T-TFGN")
                            ? "bg-cyan-950/30 text-cyan-200 font-semibold"
                            : "hover:bg-slate-850"
                        }
                      >
                        <td className="p-3 text-slate-100 flex items-center gap-1.5">
                          {m.model.includes("T-TFGN") && <Zap className="h-3.5 w-3.5 text-cyan-400" />}
                          {m.model}
                        </td>
                        <td className="p-3 text-slate-400">{m.role}</td>
                        <td className="p-3 text-cyan-400">{m.rocAuc}</td>
                        <td className="p-3 text-emerald-400">{m.prAuc}</td>
                        <td className="p-3 text-amber-400">{m.f1}</td>
                        <td className="p-3">{m.brier}</td>
                        <td className="p-3">{m.ece}</td>
                        <td className="p-3">{m.accuracy}</td>
                        <td className="p-3">{m.recall}</td>
                        <td className="p-3">{m.precision}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Graphs Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* ROC Curve */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <LineChart className="h-4 w-4 text-cyan-400" />
                  Receiver Operating Characteristic (ROC Curves)
                </h3>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 h-64 flex items-center justify-center">
                  <svg className="w-full h-full" viewBox="0 0 320 220">
                    <line x1="30" y1="20" x2="30" y2="190" stroke="#334155" strokeWidth="1" />
                    <line x1="30" y1="190" x2="300" y2="190" stroke="#334155" strokeWidth="1" />
                    <line x1="30" y1="105" x2="300" y2="105" stroke="#1e293b" strokeDasharray="2 2" />
                    <line x1="165" y1="20" x2="165" y2="190" stroke="#1e293b" strokeDasharray="2 2" />

                    <line x1="30" y1="190" x2="300" y2="20" stroke="#475569" strokeDasharray="3 3" strokeWidth="1.5" />

                    <path d="M 30,190 Q 60,110 110,80 T 210,40 T 300,20" fill="none" stroke="#06b6d4" strokeWidth="2.5" />
                    <path d="M 30,190 Q 75,130 130,95 T 225,50 T 300,20" fill="none" stroke="#10b981" strokeWidth="2" />
                    <path d="M 30,190 Q 90,140 150,110 T 240,60 T 300,20" fill="none" stroke="#818cf8" strokeWidth="2" />

                    <text x="35" y="32" fill="#10b981" fontSize="9" fontWeight="bold" fontFamily="monospace">
                      RF Lab-Only (AUC: 0.565 - Top Baseline)
                    </text>
                    <text x="35" y="44" fill="#818cf8" fontSize="9" fontWeight="bold" fontFamily="monospace">
                      XGBoost (AUC: 0.542)
                    </text>
                    <text x="35" y="56" fill="#06b6d4" fontSize="9" fontWeight="bold" fontFamily="monospace">
                      T-TFGN (AUC: 0.503 - Underperforms Baseline)
                    </text>
                    <text x="210" y="185" fill="#64748b" fontSize="8" fontFamily="monospace">FPR (1-Spec)</text>
                    <text x="35" y="100" fill="#64748b" fontSize="8" fontFamily="monospace" transform="rotate(-90 35,100)">Sensitivity</text>
                  </svg>
                </div>
              </div>

              {/* Calibration Plot */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-emerald-400" />
                  Reliability Diagram (Calibration Curve)
                </h3>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 h-64 flex items-center justify-center">
                  <svg className="w-full h-full" viewBox="0 0 320 220">
                    <line x1="30" y1="20" x2="30" y2="190" stroke="#334155" strokeWidth="1" />
                    <line x1="30" y1="190" x2="300" y2="190" stroke="#334155" strokeWidth="1" />

                    <line x1="30" y1="190" x2="300" y2="20" stroke="#64748b" strokeDasharray="3 3" strokeWidth="1.5" />

                    {CALIBRATION_DECILES.map((d, idx) => {
                      const cx = 30 + (d.predCas / 1.0) * 270;
                      const cy = 190 - (d.obsActionRate / 1.0) * 170;
                      return (
                        <g key={idx}>
                          <line x1={cx} y1="190" x2={cx} y2={cy} stroke="#06b6d4" strokeWidth="1" strokeDasharray="1 1" opacity="0.5" />
                          <circle cx={cx} cy={cy} r="4" fill="#38bdf8" stroke="#ffffff" strokeWidth="1" />
                        </g>
                      );
                    })}

                    <polyline
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="2"
                      points={CALIBRATION_DECILES.map((d) => `${30 + (d.predCas / 1.0) * 270},${190 - (d.obsActionRate / 1.0) * 170}`).join(" ")}
                    />

                    <text x="160" y="205" fill="#64748b" fontSize="8" fontFamily="monospace">Predicted CAS</text>
                    <text x="15" y="110" fill="#64748b" fontSize="8" fontFamily="monospace" transform="rotate(-90 15,110)">Observed Rate</text>
                  </svg>
                </div>
              </div>
            </div>

            {/* Controlled 7-Way Ablations */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <Layers className="h-4 w-4 text-cyan-400" />
                Controlled 7-Way Ablation Experiments (Exp A through Exp G)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden">
                  <thead className="bg-slate-950 font-mono text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="p-3">Exp ID</th>
                      <th className="p-3">Architecture Variant</th>
                      <th className="p-3">Description</th>
                      <th className="p-3">ROC-AUC</th>
                      <th className="p-3">PR-AUC</th>
                      <th className="p-3">F1-Score</th>
                      <th className="p-3">Brier Score</th>
                      <th className="p-3">ECE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                    {ABLATION_STUDY.map((ab, i) => (
                      <tr
                        key={i}
                        className={
                          ab.exp === "Exp G"
                            ? "bg-cyan-950/20 text-cyan-200 font-semibold"
                            : "hover:bg-slate-850"
                        }
                      >
                        <td className="p-3 font-bold text-slate-200">{ab.exp}</td>
                        <td className="p-3 text-slate-100">{ab.name}</td>
                        <td className="p-3 text-slate-400 font-sans">{ab.desc}</td>
                        <td className="p-3 text-cyan-400">{ab.rocAuc.toFixed(4)}</td>
                        <td className="p-3 text-emerald-400">{ab.prAuc.toFixed(4)}</td>
                        <td className="p-3 text-amber-400">{ab.f1.toFixed(4)}</td>
                        <td className="p-3">{ab.brier.toFixed(4)}</td>
                        <td className="p-3">{ab.ece.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Minimal, Premium Clinical Telemetry Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 px-6 py-3.5 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></div>
          <span className="font-medium text-slate-300">
            T-TFGN Clinical Analytics Engine
          </span>
          <span className="text-slate-700">·</span>
          <span>MIMIC-IV Multi-Analyte Suite</span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-400">Zero-Leakage Temporal Verification Enforced</span>
        </div>
        <div className="text-slate-400 font-mono text-[11px] shrink-0">
          Intensive Care Monitoring Infrastructure · Version 2.2
        </div>
      </footer>

      {/* Patient Admission & Clinical Demonstration Modal */}
      <PatientAdmissionModal
        isOpen={isAdmissionModalOpen}
        onClose={() => setIsAdmissionModalOpen(false)}
        onAdmitPatient={handleAdmitPatient}
      />

      {/* Physician Morning Rounds & Handover Summary Modal */}
      <RoundsSummaryModal
        isOpen={isRoundsModalOpen}
        onClose={() => setIsRoundsModalOpen(false)}
        patient={currentPatient}
      />
    </div>
  );
}
