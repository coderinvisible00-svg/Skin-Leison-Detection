import { useEffect, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend,
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

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setError("Cannot reach the backend. Is it running on port 8000?"));
  }, []);

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
      <header className="hero">
        <h1>Skin Lesion Analysis</h1>
        <p>Upload a dermatoscopic image and see what the model predicts, with full training results below.</p>
        <span className="pill">Educational project, not a medical device</span>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="grid two">
        <div className="card">
          <h2><span className="step">1</span>Upload an image</h2>
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
              <>
                <div className="drop-icon">🖼️</div>
                <strong>Drag and drop an image</strong>
                <span>or click to browse (JPG, PNG)</span>
              </>
            )}
          </div>
          <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files[0])} />
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading && <span className="spinner" />}
            {loading ? "Analyzing..." : "Analyze image"}
          </button>
        </div>

        <div className="card">
          <h2><span className="step">2</span>Prediction</h2>
          {!result ? (
            <div className="empty">
              <div className="drop-icon">🔬</div>
              <p className="muted">Results will appear here after you analyze an image.</p>
            </div>
          ) : (
            <Result result={result} />
          )}
        </div>
      </section>

      {metrics && <ModelPerformance m={metrics} />}

      <footer>
        Trained on HAM10000. Predictions can be wrong; always consult a qualified doctor for any skin concern.
      </footer>
    </div>
  );
}

function Result({ result }) {
  const color = GROUP_COLORS[result.group];
  const pct = +(result.confidence * 100).toFixed(1);
  return (
    <>
      <div className="result-head">
        <div className="ring" style={{ "--p": pct, "--c": color }}>
          <span>{pct}%</span>
        </div>
        <div>
          <span className="badge" style={{ background: color }}>{result.group}</span>
          <div className="pred-label">{result.label}</div>
          <div className="muted">Model confidence</div>
        </div>
      </div>

      <div className="bars">
        {result.probabilities.map((p, i) => {
          const v = +(p.probability * 100).toFixed(1);
          return (
            <div key={p.code} className={`bar-row ${i === 0 ? "top" : ""}`}>
              <span>{p.label}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${v}%`, background: GROUP_COLORS[p.group] }} />
              </div>
              <span className="bar-pct">{v}%</span>
            </div>
          );
        })}
      </div>

      <div className="note">
        A high confidence score does not mean the result is correct. This tool cannot replace a medical examination.
      </div>
    </>
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
      <h2 className="section-title">Model performance</h2>

      <section className="grid stats">
        <Stat label="Model" value={m.model} />
        <Stat label="Test accuracy" value={`${(m.test_accuracy * 100).toFixed(1)}%`} />
        <Stat label="Test macro F1" value={m.test_macro_f1.toFixed(3)} />
        <Stat label="Epochs" value={m.epochs} />
      </section>

      <section className="grid two">
        <div className="card">
          <h3>Training curves</h3>
          <div style={{ height: 290 }}>
            <ResponsiveContainer>
              <LineChart data={curves}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train loss" stroke="#a5b4fc" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val loss" stroke="#4f46e5" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="train F1" stroke="#67e8f9" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val F1" stroke="#0891b2" strokeWidth={2} dot={false} />
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
              <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
            </thead>
            <tbody>
              {m.class_names.map((c) => {
                const r = m.per_class[c];
                return (
                  <tr key={c}>
                    <td>{m.class_full_names[c] || c} <span className="muted">({c})</span></td>
                    <td>{r.precision.toFixed(2)}</td>
                    <td>
                      <div className="mini">
                        {r.recall.toFixed(2)}
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${r.recall * 100}%`, background: "var(--accent)" }} />
                        </div>
                      </div>
                    </td>
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
  return (
    <div className="cm" style={{ gridTemplateColumns: `56px repeat(${names.length}, 1fr)` }}>
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
                background: `rgba(79, 70, 229, ${v / total})`,
                color: v / total > 0.5 ? "#fff" : "inherit",
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
