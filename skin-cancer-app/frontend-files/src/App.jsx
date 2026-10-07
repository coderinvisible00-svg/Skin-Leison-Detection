import { useEffect, useRef, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, Legend,
} from "recharts";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const GROUP_COLORS = {
  Malignant: "#dc2626",
  "Pre-cancerous": "#d97706",
  Benign: "#2563eb",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // Load the Kaggle training results once
  useEffect(() => {
    fetch(`${API}/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setError("Cannot reach the backend. Is it running on port 8000?"));
  }, []);

  // Free the preview URL when it changes
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function pick(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError("");
    setResult(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Prediction failed.");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>Skin Lesion Analysis Dashboard</h1>
        <p>Educational project: not a medical device and not a diagnosis.</p>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="grid two">
        {/* Upload */}
        <div className="card">
          <h2>1. Upload an image</h2>
          <div
            className={`drop ${dragging ? "drag" : ""}`}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
          >
            {preview ? (
              <img src={preview} alt="Selected lesion" />
            ) : (
              <span>Drag and drop an image here, or click to browse</span>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => pick(e.target.files[0])}
          />
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading ? "Analyzing..." : "Analyze image"}
          </button>
        </div>

        {/* Prediction */}
        <div className="card">
          <h2>2. Prediction</h2>
          {!result ? (
            <p className="muted">Results will appear here after you analyze an image.</p>
          ) : (
            <>
              <div className="pred" style={{ borderColor: GROUP_COLORS[result.group] }}>
                <span className="badge" style={{ background: GROUP_COLORS[result.group] }}>
                  {result.group}
                </span>
                <div className="pred-label">{result.label}</div>
                <div className="pred-conf">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ height: 260 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={result.probabilities.map((p) => ({ ...p, pct: +(p.probability * 100).toFixed(1) }))}
                    layout="vertical"
                    margin={{ left: 40, right: 20 }}
                  >
                    <XAxis type="number" domain={[0, 100]} unit="%" />
                    <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                      {result.probabilities.map((p) => (
                        <Cell key={p.code} fill={GROUP_COLORS[p.group]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      </section>

      {metrics && <ModelPerformance m={metrics} />}
    </div>
  );
}

function ModelPerformance({ m }) {
  const curves = m.history.train_loss.map((_, i) => ({
    epoch: i + 1,
    "train loss": m.history.train_loss[i],
    "val loss": m.history.val_loss[i],
    "train F1": m.history.train_f1[i],
    "val F1": m.history.val_f1[i],
  }));

  return (
    <>
      <h2 className="section-title">Model performance (from Kaggle training)</h2>

      <section className="grid stats">
        <Stat label="Model" value={m.model} />
        <Stat label="Test accuracy" value={`${(m.test_accuracy * 100).toFixed(1)}%`} />
        <Stat label="Test macro F1" value={m.test_macro_f1.toFixed(3)} />
        <Stat label="Epochs" value={m.epochs} />
      </section>

      <section className="grid two">
        <div className="card">
          <h3>Training curves</h3>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={curves}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train loss" stroke="#93c5fd" dot={false} />
                <Line type="monotone" dataKey="val loss" stroke="#2563eb" dot={false} />
                <Line type="monotone" dataKey="train F1" stroke="#86efac" dot={false} />
                <Line type="monotone" dataKey="val F1" stroke="#16a34a" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3>Confusion matrix (test set)</h3>
          <ConfusionMatrix names={m.class_names} cm={m.confusion_matrix} />
        </div>
      </section>

      <div className="card">
        <h3>Per-class results</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
              </tr>
            </thead>
            <tbody>
              {m.class_names.map((c) => {
                const r = m.per_class[c];
                return (
                  <tr key={c}>
                    <td>{m.class_full_names[c] || c} <span className="muted">({c})</span></td>
                    <td>{r.precision.toFixed(2)}</td>
                    <td>{r.recall.toFixed(2)}</td>
                    <td>{r["f1-score"].toFixed(2)}</td>
                    <td>{r.support}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function Stat({ label, value }) {
  return (
    <div className="card stat">
      <div className="stat-value">{value}</div>
      <div className="muted">{label}</div>
    </div>
  );
}

function ConfusionMatrix({ names, cm }) {
  const n = names.length;
  return (
    <div className="cm" style={{ gridTemplateColumns: `60px repeat(${n}, 1fr)` }}>
      <div />
      {names.map((c) => <div key={c} className="cm-head">{c}</div>)}
      {cm.map((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1;
        return [
          <div key={`h${i}`} className="cm-head cm-row">{names[i]}</div>,
          ...row.map((v, j) => (
            <div
              key={`${i}-${j}`}
              className="cm-cell"
              title={`True ${names[i]}, predicted ${names[j]}: ${v}`}
              style={{
                background: `rgba(37, 99, 235, ${v / total})`,
                color: v / total > 0.5 ? "#fff" : "#111",
              }}
            >
              {v}
            </div>
          )),
        ];
      })}
      <div className="cm-note">Rows: true class. Columns: predicted class. Shading is per row.</div>
    </div>
  );
}
