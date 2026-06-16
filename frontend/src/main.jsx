import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Database,
  Gauge,
  HeartPulse,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  ShieldCheck,
  Sun,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API_URL = import.meta.env.VITE_STUNTGUARD_API_URL || "http://localhost:8000";

const fallbackData = {
  summary: {
    rows: 297,
    regions: 27,
    year_min: 2014,
    year_max: 2024,
    average_stunting: 15.42,
    best_model: "random_forest",
    best_macro_f1: 0.844779,
  },
  yearly_trend: [
    { tahun: 2014, persentase_stunting: 29.5 },
    { tahun: 2016, persentase_stunting: 26.1 },
    { tahun: 2018, persentase_stunting: 22.7 },
    { tahun: 2020, persentase_stunting: 18.9 },
    { tahun: 2022, persentase_stunting: 14.2 },
    { tahun: 2024, persentase_stunting: 15.4 },
  ],
  risk_distribution: [
    { risk_label: "Rendah", jumlah: 75 },
    { risk_label: "Sedang", jumlah: 148 },
    { risk_label: "Tinggi", jumlah: 74 },
  ],
  model_comparison: [
    { model: "decision_tree", accuracy: 0.85, f1_macro: 0.842126 },
    { model: "logistic_regression", accuracy: 0.683333, f1_macro: 0.670911 },
    { model: "random_forest", accuracy: 0.85, f1_macro: 0.844779 },
    { model: "xgboost", accuracy: 0.766667, f1_macro: 0.760417 },
  ],
  feature_importance: [
    { feature: "rolling_mean_3y", importance: 0.39 },
    { feature: "lag_1_stunting", importance: 0.31 },
    { feature: "trend_stunting", importance: 0.18 },
    { feature: "tahun", importance: 0.12 },
  ],
  sample_rows: [
    {
      nama_kabupaten_kota: "Kabupaten Bogor",
      tahun: 2024,
      persentase_stunting: 18.3,
      risk_label: "Tinggi",
    },
    {
      nama_kabupaten_kota: "Kota Bandung",
      tahun: 2024,
      persentase_stunting: 8.1,
      risk_label: "Rendah",
    },
    {
      nama_kabupaten_kota: "Kabupaten Garut",
      tahun: 2024,
      persentase_stunting: 14.7,
      risk_label: "Sedang",
    },
  ],
};

const navigation = [
  ["overview", "Overview", BarChart3],
  ["dataset", "Dataset", Database],
  ["training", "Training", Play],
  ["prediction", "Prediction", Gauge],
  ["evaluation", "Evaluation", ShieldCheck],
];

const riskColors = {
  Rendah: "var(--success)",
  Sedang: "var(--warning)",
  Tinggi: "var(--error)",
};

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") {
    return value.toLocaleString("id-ID", { maximumFractionDigits: digits });
  }
  return value;
}

function App() {
  const [data, setData] = useState(fallbackData);
  const [source, setSource] = useState("fallback");
  const [activeTab, setActiveTab] = useState("overview");
  const [theme, setTheme] = useState("dark");
  const [sidebarWidth, setSidebarWidth] = useState(248);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    fetch("/stuntguard-dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("No exported dashboard data");
        return response.json();
      })
      .then((payload) => {
        setData(payload);
        setSource("pipeline");
      })
      .catch(() => setSource("fallback"));
  }, []);

  const bestModel = useMemo(() => {
    return [...(data.model_comparison || [])].sort(
      (a, b) => (b.f1_macro || 0) - (a.f1_macro || 0),
    )[0];
  }, [data]);

  function startResize(event) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = sidebarWidth;

    function onMove(moveEvent) {
      const nextWidth = Math.min(340, Math.max(188, startWidth + moveEvent.clientX - startX));
      setSidebarWidth(nextWidth);
    }

    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  const layoutWidth = sidebarCollapsed ? 72 : sidebarWidth;

  return (
    <main
      className={sidebarCollapsed ? "app-shell collapsed" : "app-shell"}
      data-theme={theme}
      style={{ "--sidebar-width": `${layoutWidth}px` }}
    >
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <HeartPulse size={22} />
          </div>
          {!sidebarCollapsed && (
            <div className="brand-copy">
              <p className="overline">STUNTGUARD</p>
              <h1>Jabar ML</h1>
            </div>
          )}
        </div>

        <nav className="nav-list">
          {navigation.map(([id, label, Icon]) => (
            <button
              key={id}
              className={activeTab === id ? "nav-item active" : "nav-item"}
              onClick={() => setActiveTab(id)}
              title={label}
            >
              <Icon size={16} />
              {!sidebarCollapsed && <span>{label}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-actions">
          <button
            className="icon-button"
            onClick={() => setSidebarCollapsed((value) => !value)}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
          {!sidebarCollapsed && (
            <button
              className="icon-button"
              onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
              title="Toggle theme"
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          )}
        </div>
        {!sidebarCollapsed && <div className="sidebar-resizer" onMouseDown={startResize} />}
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="overline">Risk Classification Dashboard</p>
            <h2>Stunting Surveillance Jawa Barat</h2>
          </div>
          <div className="topbar-actions">
            <span className={source === "pipeline" ? "chip info" : "chip warning"}>
              {source === "pipeline" ? "Pipeline Data" : "Demo Fallback"}
            </span>
            <button
              className="icon-button desktop-theme"
              onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
              title="Toggle theme"
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
        </header>

        <section className="hero-panel">
          <div>
            <p className="overline">Clinical data density</p>
            <h3>Early warning view for regional stunting risk</h3>
            <p>
              React dashboard ini mengikuti token warna, spacing, card, chip,
              alert hierarchy, light/dark mode, dan responsive behavior dari DESIGN.md.
            </p>
          </div>
          <div className="hero-status">
            <AlertTriangle size={20} />
            <span>Prioritaskan recall kelas Tinggi</span>
          </div>
        </section>

        {activeTab === "overview" && <Overview data={data} bestModel={bestModel} />}
        {activeTab === "dataset" && <Dataset data={data} />}
        {activeTab === "training" && <Training data={data} />}
        {activeTab === "prediction" && <Prediction />}
        {activeTab === "evaluation" && <Evaluation data={data} />}
      </section>

      <nav className="mobile-nav">
        {navigation.map(([id, label, Icon]) => (
          <button
            key={id}
            className={activeTab === id ? "mobile-nav-item active" : "mobile-nav-item"}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={17} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </main>
  );
}

function MetricCard({ label, value, tone = "info", detail }) {
  return (
    <article className={`metric-card ${tone}`}>
      <p className="metric-label">{label}</p>
      <strong>{value}</strong>
      {detail && <span>{detail}</span>}
    </article>
  );
}

function Panel({ title, children, action }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {action}
      </div>
      {children}
    </section>
  );
}

function Overview({ data, bestModel }) {
  return (
    <>
      <section className="metrics-grid">
        <MetricCard label="Jumlah Data" value={fmt(data.summary.rows, 0)} detail="observasi" />
        <MetricCard label="Wilayah" value={fmt(data.summary.regions, 0)} detail="kab/kota" />
        <MetricCard label="Rentang Tahun" value={`${data.summary.year_min}-${data.summary.year_max}`} />
        <MetricCard label="Macro F1" value={fmt(bestModel?.f1_macro, 3)} tone="success" detail={bestModel?.model} />
      </section>
      <section className="dashboard-grid">
        <Panel title="Tren Rata-rata Stunting">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.yearly_trend}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="tahun" stroke="var(--axis)" />
              <YAxis stroke="var(--axis)" />
              <Tooltip contentStyle={{ background: "var(--tooltip-bg)", border: "1px solid var(--surface-2)" }} />
              <Line type="monotone" dataKey="persentase_stunting" stroke="var(--info)" strokeWidth={3} dot />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Distribusi Risiko">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.risk_distribution}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="risk_label" stroke="var(--axis)" />
              <YAxis stroke="var(--axis)" />
              <Tooltip contentStyle={{ background: "var(--tooltip-bg)", border: "1px solid var(--surface-2)" }} />
              <Bar dataKey="jumlah" radius={[4, 4, 0, 0]}>
                {data.risk_distribution.map((item) => (
                  <Cell key={item.risk_label} fill={riskColors[item.risk_label]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </section>
    </>
  );
}

function Dataset({ data }) {
  return (
    <Panel title="Dataset Sample">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Wilayah</th>
              <th>Tahun</th>
              <th>Stunting</th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody>
            {data.sample_rows.map((row) => (
              <tr key={`${row.nama_kabupaten_kota}-${row.tahun}`}>
                <td>{row.nama_kabupaten_kota}</td>
                <td>{row.tahun}</td>
                <td>{fmt(row.persentase_stunting)}%</td>
                <td><span className={`chip ${row.risk_label.toLowerCase()}`}>{row.risk_label}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Training({ data }) {
  return (
    <Panel title="Model Comparison">
      <div className="model-list">
        {data.model_comparison.map((model) => (
          <div className="model-row" key={model.model}>
            <div>
              <strong>{model.model}</strong>
              <span>Accuracy {fmt(model.accuracy, 3)}</span>
            </div>
            <b>{fmt(model.f1_macro, 3)}</b>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function Prediction() {
  const [form, setForm] = useState({
    tahun: 2024,
    lag_1_stunting: 8.11,
    rolling_mean_3y: 12.07,
    trend_stunting: 8.97,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: Number(value) }));
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediction failed");
      setResult(payload);
    } catch (err) {
      setError(`${err.message}. Jalankan API: python -m uvicorn api.main:app --reload --port 8000`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Panel title="Prediction Console">
      <form onSubmit={submitPrediction}>
        <div className="prediction-grid">
          {[
            ["tahun", "Tahun"],
            ["lag_1_stunting", "Lag 1 Stunting"],
            ["rolling_mean_3y", "Rolling Mean 3Y"],
            ["trend_stunting", "Trend"],
          ].map(([name, label]) => (
            <label key={name}>
              {label}
              <input name={name} type="number" step="0.01" value={form[name]} onChange={updateField} />
            </label>
          ))}
        </div>
        <button className="primary-button" disabled={loading}>
          {loading ? "Predicting..." : "Predict Risk"}
          {loading ? <Activity size={16} /> : <ChevronRight size={16} />}
        </button>
      </form>

      {result && (
        <div className="result-strip">
          <Gauge size={18} />
          <span>Prediksi: <b>{result.risk_label}</b></span>
          <span>Confidence: <b>{fmt(result.probability, 3)}</b></span>
          <span>Model: <b>{result.model_name}</b></span>
        </div>
      )}
      {error && (
        <div className="result-strip error-strip">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}
    </Panel>
  );
}

function Evaluation({ data }) {
  return (
    <section className="dashboard-grid">
      <Panel title="Feature Importance">
        <div className="importance-list">
          {data.feature_importance.map((item) => (
            <div className="importance-row" key={item.feature}>
              <span>{item.feature}</span>
              <div><i style={{ width: `${Math.max(8, item.importance * 100)}%` }} /></div>
              <b>{fmt(item.importance, 3)}</b>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Operational Notes">
        <ul className="notes">
          <li><TrendingUp size={16} /> Macro F1 menjadi metrik utama.</li>
          <li><AlertTriangle size={16} /> Risiko Tinggi perlu prioritas recall.</li>
          <li><Activity size={16} /> React prediction terhubung ke FastAPI model server.</li>
        </ul>
      </Panel>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
