import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BrainViewer } from "./components/BrainViewer";
import { SliderPanel } from "./components/SliderPanel";
import { atlasManifest, getSubjects, inferSubject, listRuns } from "./lib/api";

type InferenceResponse = {
  metadata: Record<string, string | number | null>;
  inference: {
    n_processes: number;
    age_latent: number;
    process_latents: number[];
  };
  top_changes: Array<{
    roi_name: string;
    roi_full_name: string;
    delta: number;
  }>;
};

export function App() {
  const [runs, setRuns] = useState<Array<{ run_name: string }>>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [splitName, setSplitName] = useState("application");
  const [subjects, setSubjects] = useState<any[]>([]);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [sliderValues, setSliderValues] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<"slices" | "volume">("slices");
  const [atlasUrls, setAtlasUrls] = useState<{ atlas_image_url: string; atlas_segmentation_url: string } | null>(null);

  useEffect(() => {
    listRuns().then(setRuns).catch(console.error);
    atlasManifest().then(setAtlasUrls).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedRun) {
      return;
    }
    getSubjects(selectedRun, splitName)
      .then((rows) => {
        setSubjects(rows);
        if (rows.length > 0) {
          setSelectedRow(rows[0].row_index);
        }
      })
      .catch(console.error);
  }, [selectedRun, splitName]);

  useEffect(() => {
    if (!selectedRun || selectedRow === null) {
      return;
    }
    inferSubject(selectedRun, splitName, selectedRow)
      .then((payload: any) => {
        setInference(payload);
        setSliderValues(payload.inference.process_latents);
      })
      .catch(console.error);
  }, [selectedRun, splitName, selectedRow]);

  const topChanges = useMemo(
    () =>
      (inference?.top_changes ?? []).map((row) => ({
        name: row.roi_name,
        delta: row.delta,
      })),
    [inference],
  );

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">Age-Decoupled SurrealGAN</p>
          <h1>ROI Latent Explorer</h1>
          <p>Inspect age latents, process latents, and ROI-level shifts on top of the IXI atlas.</p>
        </div>

        <div className="control-group">
          <label>Run</label>
          <select value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
            <option value="">Select a run</option>
            {runs.map((run) => (
              <option key={run.run_name} value={run.run_name}>
                {run.run_name}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Split</label>
          <select value={splitName} onChange={(event) => setSplitName(event.target.value)}>
            <option value="application">Application</option>
            <option value="ood_test">OOD test</option>
            <option value="id_test">ID test</option>
            <option value="val">Validation</option>
            <option value="train">Train</option>
          </select>
        </div>

        <div className="control-group">
          <label>Subject</label>
          <select
            value={selectedRow ?? ""}
            onChange={(event) => setSelectedRow(Number(event.target.value))}
          >
            {subjects.map((subject) => (
              <option key={subject.row_index} value={subject.row_index}>
                {subject.subject_id} | {subject.study} | {subject.diagnosis_group}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>View mode</label>
          <div className="segmented-control">
            <button
              className={viewMode === "slices" ? "active" : ""}
              onClick={() => setViewMode("slices")}
            >
              2D slices
            </button>
            <button
              className={viewMode === "volume" ? "active" : ""}
              onClick={() => setViewMode("volume")}
            >
              3D volume
            </button>
          </div>
        </div>
      </aside>

      <section className="content">
        <div className="hero-grid">
          <section className="panel hero-card">
            <div className="panel-header">
              <h2>Latent summary</h2>
              <p>The process slider count is derived from the selected checkpoint metadata.</p>
            </div>
            <div className="metric-row">
              <div>
                <span>Age latent</span>
                <strong>{inference?.inference.age_latent.toFixed(3) ?? "0.000"}</strong>
              </div>
              <div>
                <span>Processes</span>
                <strong>{inference?.inference.n_processes ?? 0}</strong>
              </div>
            </div>
            <dl className="subject-meta">
              {Object.entries(inference?.metadata ?? {}).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{String(value ?? "")}</dd>
                </div>
              ))}
            </dl>
          </section>

          {atlasUrls ? (
            <BrainViewer
              atlasImageUrl={atlasUrls.atlas_image_url}
              atlasSegmentationUrl={atlasUrls.atlas_segmentation_url}
              mode={viewMode}
            />
          ) : null}
        </div>

        <SliderPanel processLatents={sliderValues} onChange={setSliderValues} />

        <section className="panel chart-panel">
          <div className="panel-header">
            <h3>Top ROI changes</h3>
            <p>Largest absolute synthetic changes for the selected subject.</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={topChanges}>
                <XAxis dataKey="name" hide />
                <YAxis />
                <Tooltip />
                <Bar dataKey="delta" fill="#1f6f8b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </section>
    </main>
  );
}
