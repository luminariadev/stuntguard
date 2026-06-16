import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Database,
  Download,
  Gauge,
  HeartPulse,
  MapPin,
  Play,
  ShieldCheck,
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

const riskColors = {
  Rendah: "var(--success)",
  Sedang: "var(--warning)",
  Tinggi: "var(--error)",
};

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") return value.toLocaleString("id-ID", { maximumFractionDigits: digits });
  return value;
}

function App() {
  const [data, setData] = useState(fallbackData);
  const [source, setSource] = useState("fallback");
  const [activeTab, setActiveTab] = useState("overview");

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
    return [...(data.model_comparison || [])].sort((a, b) => (b.f1_macro || 0) - (a.f1_macro || 0))[0];
  }, [data]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><HeartPulse size={22} /></div>
          <div>
            <p className="overline">STUNTGUARD</p>
            <h1>Jabar ML</h1>
          </div>
        </div>
        <nav className="nav-list">
          {[
            ["overview", "Overview", BarChart3],
            ["dataset", "Dataset", Database],
            ["training", "Training", Play],
            ["prediction", "Prediction", Gauge],
            ["evaluation", "Evaluation", ShieldCheck],
          ].map(([id, label, Icon]) => (
            <button
              key={id}
              className={activeTab === id ? "nav-item active" : "nav-item"}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="overline">Risk Classification Dashboard</p>
            <h2>Stunting Surveillance Jawa Barat</h2>
          </div>
          <span className={source === "pipeline" ? "chip info" : "chip warning"}>
            {source === "pipeline" ? "Pipeline Data" : "Demo Fallback"}
          </span>
        </header>

        <section className="hero-panel">
          <div>
            <p className="overline">Clinical data density</p>
            <h3>Early warning view for regional stunting risk</h3>
            <p>
              Dashboard React responsif ini mengikuti token warna, spacing, card,
              chip, dan alert hierarchy dari DESIGN.md.
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
              <CartesianGrid stroke="#334155" />
              <XAxis dataKey="tahun" stroke="#8D99AE" />
              <YAxis stroke="#8D99AE" />
              <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid #334155" }} />
              <Line type="monotone" dataKey="persentase_stunting" stroke="#3B82F6" strokeWidth={3} dot />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Distribusi Risiko">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.risk_distribution}>
              <CartesianGrid stroke="#334155" />
              <XAxis dataKey="risk_label" stroke="#8D99AE" />
              <YAxis stroke="#8D99AE" />
              <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid #334155" }} />
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
    <Panel
      title="Dataset Sample"
      action={<button className="secondary-button"><Download size={15} /> CSV ready</button>}
    >
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
  return (
    <Panel title="Prediction Console">
      <div className="prediction-grid">
        <label>
          Tahun
          <input defaultValue="2024" />
        </label>
        <label>
          Lag 1 Stunting
          <input defaultValue="8.11" />
        </label>
        <label>
          Rolling Mean 3Y
          <input defaultValue="12.07" />
        </label>
        <label>
          Trend
          <input defaultValue="8.97" />
        </label>
      </div>
      <div className="result-strip">
        <MapPin size={18} />
        <span>Use Python API/FastAPI integration for live model scoring.</span>
      </div>
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
          <li><Activity size={16} /> Data baru dapat diekspor dari pipeline Python.</li>
        </ul>
      </Panel>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
