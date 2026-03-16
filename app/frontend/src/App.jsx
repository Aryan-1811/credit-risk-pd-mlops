import { useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const RISK_CONFIG = {
  LOW:        { color: "#00e5a0", bg: "rgba(0,229,160,0.08)", border: "rgba(0,229,160,0.25)"  },
  MEDIUM:     { color: "#f5c518", bg: "rgba(245,197,24,0.08)",  border: "rgba(245,197,24,0.25)"  },
  HIGH:       { color: "#ff7b2c", bg: "rgba(255,123,44,0.08)",  border: "rgba(255,123,44,0.25)"  },
  "VERY HIGH":{ color: "#ff3b5c", bg: "rgba(255,59,92,0.08)",   border: "rgba(255,59,92,0.25)"   },
};

const DEFAULT_RISK = { color: "#4a7a9b", bg: "rgba(74,122,155,0.08)", border: "rgba(74,122,155,0.25)" };

function Gauge({ value, color }) {
  const radius = 80;
  const cx = 110, cy = 110;
  const startAngle = 210;
  const endAngle = 330;
  const totalAngle = 360 - startAngle + endAngle;
  const valueAngle = startAngle + (value / 100) * totalAngle;

  const toRad = (deg) => (deg * Math.PI) / 180;
  const arcPath = (start, end, r) => {
    const s = { x: cx + r * Math.cos(toRad(start)), y: cy + r * Math.sin(toRad(start)) };
    const e = { x: cx + r * Math.cos(toRad(end)),   y: cy + r * Math.sin(toRad(end))   };
    const large = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  };

  const needleX = cx + (radius - 10) * Math.cos(toRad(valueAngle));
  const needleY = cy + (radius - 10) * Math.sin(toRad(valueAngle));

  const zones = [
    { start: startAngle, end: startAngle + totalAngle * 0.05,  color: "#00e5a0" },
    { start: startAngle + totalAngle * 0.05, end: startAngle + totalAngle * 0.15, color: "#f5c518" },
    { start: startAngle + totalAngle * 0.15, end: startAngle + totalAngle * 0.30, color: "#ff7b2c" },
    { start: startAngle + totalAngle * 0.30, end: startAngle + totalAngle,        color: "#ff3b5c" },
  ];

  return (
    <svg viewBox="0 0 220 150" style={{ width: "100%", maxWidth: 300 }}>
      {/* Track */}
      <path d={arcPath(startAngle, startAngle + totalAngle, radius)}
        fill="none" stroke="#0f2236" strokeWidth="14" strokeLinecap="round" />
      {/* Zones */}
      {zones.map((z, i) => (
        <path key={i} d={arcPath(z.start, z.end, radius)}
          fill="none" stroke={z.color} strokeWidth="14" strokeLinecap="butt" opacity="0.25" />
      ))}
      {/* Active arc */}
      <path d={arcPath(startAngle, valueAngle, radius)}
        fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
        style={{ transition: "all 0.5s cubic-bezier(0.34,1.56,0.64,1)" }} />
      {/* Needle dot */}
      <circle cx={needleX} cy={needleY} r="5" fill={color}
        style={{ transition: "all 0.5s cubic-bezier(0.34,1.56,0.64,1)", filter: `drop-shadow(0 0 6px ${color})` }} />
      {/* Center value */}
      <text x={cx} y={cy + 10} textAnchor="middle"
        style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 28, fontWeight: 500, fill: color,
                 transition: "fill 0.3s" }}>
        {value.toFixed(2)}%
      </text>
      <text x={cx} y={cy + 30} textAnchor="middle"
        style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9, fill: "#1e4d6b", letterSpacing: "0.15em" }}>
        PROBABILITY OF DEFAULT
      </text>
    </svg>
  );
}

function Slider({ label, value, min, max, step, onChange, format }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: "1.1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
        <span style={{ fontFamily: "'Syne',sans-serif", fontSize: "0.78rem", color: "#4a7a9b" }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.78rem", color: "#94b8d4" }}>
          {format ? format(value) : value}
        </span>
      </div>
      <div style={{ position: "relative", height: 4, background: "#0f2236", borderRadius: 2 }}>
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "100%",
                      background: "linear-gradient(90deg,#1a5276,#2980b9)", borderRadius: 2,
                      transition: "width 0.1s" }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", inset: 0, width: "100%", opacity: 0,
                   cursor: "pointer", height: "100%", margin: 0 }} />
        <div style={{ position: "absolute", left: `${pct}%`, top: "50%",
                      transform: "translate(-50%,-50%)", width: 14, height: 14,
                      background: "#2980b9", borderRadius: "50%", border: "2px solid #05080f",
                      boxShadow: "0 0 8px rgba(41,128,185,0.6)", transition: "left 0.1s",
                      pointerEvents: "none" }} />
      </div>
    </div>
  );
}

function StressBar({ label, value, maxVal }) {
  const cfg = value < 5 ? RISK_CONFIG.LOW : value < 15 ? RISK_CONFIG.MEDIUM :
              value < 30 ? RISK_CONFIG.HIGH : RISK_CONFIG["VERY HIGH"];
  return (
    <div style={{ marginBottom: "0.7rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.65rem", color: "#2a5a7b" }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.65rem", color: cfg.color }}>{value.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "#0a1520", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${(value / maxVal) * 100}%`, background: cfg.color,
                      borderRadius: 3, opacity: 0.7, transition: "width 0.5s cubic-bezier(0.34,1.56,0.64,1)" }} />
      </div>
    </div>
  );
}

export default function App() {
  const [inputs, setInputs] = useState({
    revolving_utilization: 0.30,
    age: 40,
    late_30_59: 0,
    debt_ratio: 0.35,
    monthly_income: 5000,
    open_credit_lines: 4,
    dependents: 0,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchScore = useCallback(async (data) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const t = setTimeout(() => fetchScore(inputs), 300);
    return () => clearTimeout(t);
  }, [inputs, fetchScore]);

  const set = (key) => (val) => setInputs(p => ({ ...p, [key]: val }));

  const risk = result ? (RISK_CONFIG[result.risk_band] || DEFAULT_RISK) : DEFAULT_RISK;
  const pdPct = result?.pd_percent ?? 0;
  const stress = result?.stress;

  return (
    <div style={{ minHeight: "100vh", background: "#05080f", color: "#c8dde8",
                  fontFamily: "'Syne',sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #05080f; }
        ::-webkit-scrollbar-thumb { background: #0f2236; border-radius: 2px; }
      `}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "2.5rem 2rem", display: "grid",
                    gridTemplateColumns: "320px 1fr", gap: "2rem", alignItems: "start" }}>

        {/* ── LEFT PANEL ── */}
        <div style={{ background: "#080d18", border: "1px solid #0f2236", borderRadius: 16,
                      padding: "1.8rem 1.5rem", position: "sticky", top: "2rem" }}>
          {/* Header */}
          <div style={{ marginBottom: "1.8rem" }}>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem",
                          color: "#1e4d6b", letterSpacing: "0.2em", textTransform: "uppercase",
                          marginBottom: "0.5rem" }}>
              Credit Risk · Basel II IRB
            </div>
            <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#f0f8ff", letterSpacing: "-0.03em" }}>
              PD Scorer
            </div>
            <div style={{ fontSize: "0.75rem", color: "#2a5a7b", marginTop: "0.3rem" }}>
              Adjust sliders to score a borrower
            </div>
          </div>

          <div style={{ height: 1, background: "#0f2236", marginBottom: "1.5rem" }} />

          <Slider label="Credit Utilisation" value={inputs.revolving_utilization}
            min={0} max={1} step={0.01} onChange={set("revolving_utilization")}
            format={v => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Age" value={inputs.age} min={18} max={80} step={1}
            onChange={set("age")} format={v => `${v} yrs`} />
          <Slider label="Times 30–59 Days Late" value={inputs.late_30_59}
            min={0} max={10} step={1} onChange={set("late_30_59")} />
          <Slider label="Debt Ratio" value={inputs.debt_ratio}
            min={0} max={2} step={0.01} onChange={set("debt_ratio")}
            format={v => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Monthly Income (£)" value={inputs.monthly_income}
            min={0} max={20000} step={100} onChange={set("monthly_income")}
            format={v => `£${v.toLocaleString()}`} />
          <Slider label="Open Credit Lines" value={inputs.open_credit_lines}
            min={0} max={30} step={1} onChange={set("open_credit_lines")} />
          <Slider label="Dependents" value={inputs.dependents}
            min={0} max={10} step={1} onChange={set("dependents")} />

          {/* Model info */}
          <div style={{ height: 1, background: "#0f2236", margin: "1.5rem 0 1.2rem" }} />
          <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.62rem",
                        color: "#1e4d6b", lineHeight: 2.2 }}>
            <span style={{ color: "#1e6b9e" }}>MODEL &nbsp;&nbsp;</span> LightGBM · isotonic cal.<br/>
            <span style={{ color: "#1e6b9e" }}>FEATURES </span> WoE · IV≥0.02 · 7 vars<br/>
            <span style={{ color: "#1e6b9e" }}>TRACKING </span> MLflow Registry v1<br/>
            <span style={{ color: "#1e6b9e" }}>PIPELINE </span> Prefect · GitHub Actions
          </div>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>

          {/* Top row: gauge + KPIs */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.2rem" }}>

            {/* Gauge card */}
            <div style={{ background: "#080d18", border: `1px solid ${risk.border}`,
                          borderRadius: 16, padding: "1.8rem 1.5rem",
                          display: "flex", flexDirection: "column", alignItems: "center",
                          transition: "border-color 0.3s" }}>
              <Gauge value={pdPct} color={risk.color} />
              {result && (
                <div style={{ marginTop: "0.8rem", padding: "0.5rem 1.4rem",
                              background: risk.bg, border: `1px solid ${risk.border}`,
                              borderRadius: 6, fontFamily: "'JetBrains Mono',monospace",
                              fontSize: "0.85rem", fontWeight: 500, color: risk.color,
                              letterSpacing: "0.1em", transition: "all 0.3s" }}>
                  {result.risk_band}
                </div>
              )}
            </div>

            {/* KPIs */}
            <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
              {[
                { label: "Probability of Default", value: result ? `${pdPct.toFixed(2)}%` : "—" },
                { label: "Expected Loss (per £1)", value: result ? `£${result.expected_loss.toFixed(4)}` : "—",
                  sub: "EL = PD × LGD × EAD" },
                { label: "LGD (Basel II Floor)", value: "45.00%", sub: "Unsecured retail" },
              ].map(({ label, value, sub }) => (
                <div key={label} style={{ background: "#080d18", border: "1px solid #0f2236",
                                          borderRadius: 12, padding: "1rem 1.2rem", flex: 1 }}>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.58rem",
                                letterSpacing: "0.18em", color: "#1e4d6b", textTransform: "uppercase",
                                marginBottom: "0.4rem" }}>{label}</div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "1.4rem",
                                fontWeight: 500, color: risk.color, transition: "color 0.3s" }}>{value}</div>
                  {sub && <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem",
                                        color: "#1e4d6b", marginTop: "0.2rem" }}>{sub}</div>}
                </div>
              ))}
            </div>
          </div>

          {/* Stress test */}
          <div style={{ background: "#080d18", border: "1px solid #0f2236",
                        borderRadius: 16, padding: "1.5rem 1.5rem" }}>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem",
                          letterSpacing: "0.2em", color: "#1e4d6b", textTransform: "uppercase",
                          marginBottom: "1.1rem" }}>Macro Stress Testing</div>
            {stress ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 2rem" }}>
                {[
                  ["Baseline",          stress.baseline * 100],
                  ["Mild Stress ×1.3",  stress.mild_stress * 100],
                  ["Adverse ×1.7",      stress.adverse * 100],
                  ["Severely Adverse ×2.5", stress.severely_adverse * 100],
                ].map(([label, val]) => (
                  <StressBar key={label} label={label} value={val}
                    maxVal={Math.max(stress.severely_adverse * 100, 5)} />
                ))}
              </div>
            ) : (
              <div style={{ color: "#1e4d6b", fontFamily: "'JetBrains Mono',monospace",
                            fontSize: "0.75rem" }}>Loading...</div>
            )}
          </div>

          {/* Decision + formula */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.2rem" }}>
            {/* Decision */}
            <div style={{ background: risk.bg, border: `1px solid ${risk.border}`,
                          borderRadius: 16, padding: "1.4rem 1.5rem", transition: "all 0.3s" }}>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem",
                            letterSpacing: "0.2em", color: risk.color, opacity: 0.7,
                            textTransform: "uppercase", marginBottom: "0.7rem" }}>Credit Decision</div>
              <div style={{ fontFamily: "'Syne',sans-serif", fontSize: "1rem", fontWeight: 700,
                            color: risk.color, lineHeight: 1.4 }}>
                {result?.decision ?? "—"}
              </div>
              <div style={{ marginTop: "0.8rem", fontFamily: "'JetBrains Mono',monospace",
                            fontSize: "0.65rem", color: "#2a5a7b", lineHeight: 1.8 }}>
                {result?.risk_band === "LOW" && "Strong borrower. Standard rate applies."}
                {result?.risk_band === "MEDIUM" && "Acceptable risk. Consider risk-adjusted pricing."}
                {result?.risk_band === "HIGH" && "Elevated risk. Manual underwriter review required."}
                {result?.risk_band === "VERY HIGH" && "PD exceeds risk appetite. Decline application."}
              </div>
            </div>

            {/* Formula */}
            <div style={{ background: "#080d18", border: "1px solid #0f2236",
                          borderLeft: "3px solid #1e6b9e", borderRadius: "0 16px 16px 0",
                          padding: "1.4rem 1.5rem" }}>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem",
                            letterSpacing: "0.2em", color: "#1e4d6b", textTransform: "uppercase",
                            marginBottom: "0.8rem" }}>Basel II Formula</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.85rem",
                            color: "#4a7a9b", lineHeight: 2.4 }}>
                EL &nbsp;=&nbsp; PD &nbsp;×&nbsp; LGD &nbsp;×&nbsp; EAD<br/>
                <span style={{ color: "#2a5a7b" }}>
                  &nbsp;&nbsp;&nbsp;=&nbsp;{" "}
                  <span style={{ color: risk.color }}>{result ? pdPct.toFixed(2) : "0.00"}%</span>
                  {" "}×{" "}
                  <span style={{ color: "#94b8d4" }}>45%</span>
                  {" "}×{" "}
                  <span style={{ color: "#94b8d4" }}>1.00</span>
                </span><br/>
                <span style={{ color: risk.color, fontWeight: 500 }}>
                  &nbsp;&nbsp;&nbsp;= £{result ? result.expected_loss.toFixed(4) : "0.0000"}
                </span>
              </div>
            </div>
          </div>

          {error && (
            <div style={{ background: "rgba(255,59,92,0.08)", border: "1px solid rgba(255,59,92,0.3)",
                          borderRadius: 12, padding: "1rem 1.2rem", fontFamily: "'JetBrains Mono',monospace",
                          fontSize: "0.75rem", color: "#ff3b5c" }}>
              ⚠ API Error: {error}. Make sure the FastAPI backend is running.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
