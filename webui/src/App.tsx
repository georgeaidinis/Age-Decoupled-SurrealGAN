import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BrainViewer } from "./components/BrainViewer";
import { LatentSpaceExplorer } from "./components/LatentSpaceExplorer";
import { SliderPanel } from "./components/SliderPanel";
import {
  atlasManifest,
  getLatentSpace,
  getAtlasRoiMetadata,
  getPopulationPattern,
  getPopulationPatterns,
  getRunMetadata,
  getSubjectDefaults,
  getSubjects,
  inferSubject,
  listRuns,
  type InferenceResponse,
  type LatentSpaceResponse,
  type PopulationPatternManifest,
  type PopulationPatternResponse,
  type RoiMetadataRow,
  type SubjectRow,
} from "./lib/api";

type RunMetadata = {
  n_processes: number;
};

export function App() {
  const [runs, setRuns] = useState<Array<{ run_name: string }>>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [runMetadata, setRunMetadata] = useState<RunMetadata | null>(null);
  const [splitName, setSplitName] = useState("application");
  const [subjects, setSubjects] = useState<SubjectRow[]>([]);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [subjectMetadata, setSubjectMetadata] = useState<Record<string, string | number | null> | null>(null);
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [ageYears, setAgeYears] = useState(20);
  const [processLatents, setProcessLatents] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<"slices" | "volume">("slices");
  const [displayMode, setDisplayMode] = useState<"subject" | "population">("subject");
  const [overlayScaleMode, setOverlayScaleMode] = useState<"relative" | "absolute">("relative");
  const [overlayOpacity, setOverlayOpacity] = useState(0.38);
  const [absoluteOverlayScale, setAbsoluteOverlayScale] = useState(1.0);
  const [atlasUrls, setAtlasUrls] = useState<{ atlas_image_url: string; atlas_segmentation_url: string } | null>(null);
  const [roiMetadata, setRoiMetadata] = useState<RoiMetadataRow[]>([]);
  const [overlayImageUrl, setOverlayImageUrl] = useState<string | null>(null);
  const [populationManifest, setPopulationManifest] = useState<PopulationPatternManifest | null>(null);
  const [selectedPopulationPattern, setSelectedPopulationPattern] = useState("age");
  const [populationPattern, setPopulationPattern] = useState<PopulationPatternResponse | null>(null);
  const [latentSpace, setLatentSpace] = useState<LatentSpaceResponse | null>(null);
  const [loadingDefaults, setLoadingDefaults] = useState(false);
  const [loadingInference, setLoadingInference] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([listRuns(), atlasManifest()])
      .then(async ([loadedRuns, manifest]) => {
        const roiRows = manifest.roi_metadata_url ? await getAtlasRoiMetadata(manifest.roi_metadata_url) : [];
        setRuns(loadedRuns);
        setAtlasUrls({
          atlas_image_url: manifest.atlas_image_url,
          atlas_segmentation_url: manifest.atlas_segmentation_url,
        });
        setRoiMetadata(roiRows);
      })
      .catch((loadError) => {
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to initialize the application.");
      });
  }, []);

  useEffect(() => {
    if (!selectedRun) {
      setRunMetadata(null);
      setSubjects([]);
      setSelectedRow(null);
      setSubjectMetadata(null);
      setInference(null);
      setOverlayImageUrl(null);
      setAgeYears(20);
      setProcessLatents([]);
      setPopulationManifest(null);
      setSelectedPopulationPattern("age");
      setPopulationPattern(null);
      return;
    }
    let cancelled = false;
    setError(null);
    setInference(null);
    setOverlayImageUrl(null);
    getRunMetadata(selectedRun)
      .then((metadata: any) => {
        if (cancelled) {
          return;
        }
        setRunMetadata(metadata);
        setProcessLatents(Array.from({ length: metadata.n_processes ?? 0 }, () => 0));
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load run metadata.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedRun) {
      return;
    }
    let cancelled = false;
    setPopulationManifest(null);
    setPopulationPattern(null);
    getPopulationPatterns(selectedRun)
      .then((manifest) => {
        if (cancelled) {
          return;
        }
        setPopulationManifest(manifest);
        setSelectedPopulationPattern(manifest.patterns[0]?.key ?? "age");
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load population patterns.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedRun || !selectedPopulationPattern) {
      return;
    }
    let cancelled = false;
    getPopulationPattern(selectedRun, selectedPopulationPattern)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setPopulationPattern(payload);
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load the selected population pattern.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPopulationPattern, selectedRun]);

  useEffect(() => {
    if (!selectedRun) {
      return;
    }
    let cancelled = false;
    setLatentSpace(null);
    setSubjects([]);
    setSelectedRow(null);
    setSubjectMetadata(null);
    setInference(null);
    setOverlayImageUrl(null);
    setError(null);
    getSubjects(selectedRun, splitName)
      .then((rows) => {
        if (cancelled) {
          return;
        }
        setSubjects(rows);
        setSelectedRow(rows.length > 0 ? rows[0].row_index : null);
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load split subjects.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRun, splitName]);

  useEffect(() => {
    if (!selectedRun) {
      return;
    }
    let cancelled = false;
    getLatentSpace(selectedRun, splitName)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setLatentSpace(payload);
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load latent-space predictions.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRun, splitName]);

  useEffect(() => {
    if (!selectedRun || selectedRow === null) {
      return;
    }
    let cancelled = false;
    setLoadingDefaults(true);
    setInference(null);
    setOverlayImageUrl(null);
    setError(null);
    getSubjectDefaults(selectedRun, splitName, selectedRow)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setSubjectMetadata(payload.metadata);
        setAgeYears(payload.defaults.age_years);
        setProcessLatents(payload.defaults.process_latents);
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        console.error(loadError);
        setError(loadError instanceof Error ? loadError.message : "Failed to load subject defaults.");
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingDefaults(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRun, splitName, selectedRow]);

  const handleGenerate = () => {
    if (!selectedRun || selectedRow === null) {
      return;
    }
    setLoadingInference(true);
    setError(null);
    inferSubject(selectedRun, splitName, selectedRow, ageYears, processLatents)
      .then((payload) => {
        setInference(payload);
        setOverlayImageUrl(payload.overlay_image_url);
      })
      .catch((loadError) => {
        console.error(loadError);
        setInference(null);
        setOverlayImageUrl(null);
        setError(loadError instanceof Error ? loadError.message : "Inference failed.");
      })
      .finally(() => setLoadingInference(false));
  };

  const topChanges = useMemo(
    () =>
      ((displayMode === "population" ? populationPattern?.top_changes : inference?.top_changes) ?? []).map((row) => ({
        name: row.roi_name ?? row.roi_full_name ?? "Unknown ROI",
        percent_change: Number(row.percent_change ?? 0),
        delta: Number(row.delta ?? 0),
        baseline_value: Number(row.baseline_value ?? 0),
        predicted_value: Number(row.predicted_value ?? 0),
      })),
    [displayMode, inference, populationPattern],
  );

  const roiMetadataById = useMemo(
    () =>
      Object.fromEntries(
        roiMetadata
          .filter((row) => Number.isFinite(Number(row.roi_id)))
          .map((row) => [Number(row.roi_id), row]),
      ) as Record<number, RoiMetadataRow>,
    [roiMetadata],
  );

  const roiValueById = useMemo(
    () =>
      Object.fromEntries(
        ((displayMode === "population" ? populationPattern?.roi_table : inference?.roi_table) ?? [])
          .filter((row) => Number.isFinite(Number(row.roi_id)))
          .map((row) => [
            Number(row.roi_id),
            {
              delta: Number(row.delta ?? 0),
              percent_change: Number(row.percent_change ?? 0),
              baseline_value: Number(row.baseline_value ?? 0),
              predicted_value: Number(row.predicted_value ?? 0),
            },
          ]),
      ) as Record<number, { delta: number; percent_change: number; baseline_value: number; predicted_value: number }>,
    [displayMode, inference, populationPattern],
  );

  const chartMode = useMemo(() => {
    const maxPercent = topChanges.reduce((current, row) => Math.max(current, Math.abs(row.percent_change)), 0);
    return maxPercent >= 0.01 ? "percent" : "delta";
  }, [topChanges]);

  const orderedTopChanges = useMemo(
    () =>
      [...topChanges].sort(
        (lhs, rhs) =>
          Math.abs(chartMode === "percent" ? rhs.percent_change : rhs.delta) -
          Math.abs(chartMode === "percent" ? lhs.percent_change : lhs.delta),
      ),
    [chartMode, topChanges],
  );

  const overlayAbsMax = useMemo(() => {
    const values =
      displayMode === "population"
        ? (populationPattern?.roi_table ?? []).map((row) => Number(row.percent_change ?? 0))
        : (inference?.inference.percent_change ?? []);
    const maxValue = values.reduce((current, value) => Math.max(current, Math.abs(value)), 0);
    if (overlayScaleMode === "relative") {
      return Math.max(maxValue, 1e-4);
    }
    return Math.max(absoluteOverlayScale, 1e-4);
  }, [absoluteOverlayScale, displayMode, inference, overlayScaleMode, populationPattern]);

  const currentMetadata =
    displayMode === "population"
      ? {
          mode: "population",
          pattern: populationPattern?.label ?? selectedPopulationPattern,
          reference_split: populationManifest?.reference_split ?? "",
          reference_bucket: populationManifest?.reference_cohort_bucket ?? "",
          anchor_age: populationManifest?.process_anchor_age_years ?? "",
        }
      : inference?.metadata ?? subjectMetadata ?? {};
  const activeOverlayImageUrl = displayMode === "population" ? populationPattern?.overlay_image_url ?? null : overlayImageUrl;
  const controlsDisabled = !selectedRun || selectedRow === null || loadingDefaults || loadingInference;
  const formatMetadataValue = (key: string, value: string | number | null) => {
    if (value === null || value === undefined || value === "") {
      return "";
    }
    if (typeof value === "number") {
      if (key.toLowerCase() === "age") {
        return value.toFixed(2);
      }
      if (Number.isInteger(value)) {
        return String(value);
      }
      return value.toFixed(3);
    }
    return String(value);
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">Age-Decoupled SurrealGAN</p>
          <h1>ROI Latent Explorer</h1>
          <p>Generate ROI-space changes from age and process controls, then visualize them on the atlas.</p>
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
          <select value={splitName} onChange={(event) => setSplitName(event.target.value)} disabled={!selectedRun}>
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
            disabled={!selectedRun || subjects.length === 0}
            onChange={(event) => setSelectedRow(Number(event.target.value))}
          >
            {subjects.length === 0 ? <option value="">No subjects available</option> : null}
            {subjects.map((subject) => (
              <option key={subject.row_index} value={subject.row_index}>
                {subject.subject_id} | {subject.study} | {subject.diagnosis_group}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Display mode</label>
          <div className="segmented-control">
            <button
              className={displayMode === "subject" ? "active" : ""}
              type="button"
              onClick={() => setDisplayMode("subject")}
            >
              Subject
            </button>
            <button
              className={displayMode === "population" ? "active" : ""}
              type="button"
              onClick={() => setDisplayMode("population")}
              disabled={!populationManifest}
            >
              Population
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>View mode</label>
          <div className="segmented-control">
            <button
              className={viewMode === "slices" ? "active" : ""}
              type="button"
              onClick={() => setViewMode("slices")}
            >
              2D slices
            </button>
            <button
              className={viewMode === "volume" ? "active" : ""}
              type="button"
              onClick={() => setViewMode("volume")}
            >
              3D volume
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Overlay scaling</label>
          <div className="segmented-control">
            <button
              className={overlayScaleMode === "relative" ? "active" : ""}
              type="button"
              onClick={() => setOverlayScaleMode("relative")}
            >
              Relative
            </button>
            <button
              className={overlayScaleMode === "absolute" ? "active" : ""}
              type="button"
              onClick={() => setOverlayScaleMode("absolute")}
            >
              Absolute
            </button>
          </div>
          <p className="control-help">
            Relative rescales each generated overlay to its own max change. Absolute preserves a fixed percent scale so
            runs are comparable.
          </p>
        </div>

        {displayMode === "population" && populationManifest ? (
          <div className="control-group">
            <label>Population pattern</label>
            <select value={selectedPopulationPattern} onChange={(event) => setSelectedPopulationPattern(event.target.value)}>
              {populationManifest.patterns.map((pattern) => (
                <option key={pattern.key} value={pattern.key}>
                  {pattern.label}
                </option>
              ))}
            </select>
            <p className="control-help">{populationPattern?.description ?? "Select age or one of the process factors."}</p>
          </div>
        ) : null}

        {overlayScaleMode === "absolute" ? (
          <div className="control-group">
            <label>Absolute overlay range</label>
            <input
              type="range"
              min={0.1}
              max={10}
              step={0.1}
              value={absoluteOverlayScale}
              onChange={(event) => setAbsoluteOverlayScale(Number(event.target.value))}
            />
            <p className="control-help">Clip the blue/red overlay at ±{absoluteOverlayScale.toFixed(1)}% change.</p>
          </div>
        ) : null}

        <div className="control-group">
          <label>Overlay opacity</label>
          <input
            type="range"
            min={0.05}
            max={0.75}
            step={0.01}
            value={overlayOpacity}
            onChange={(event) => setOverlayOpacity(Number(event.target.value))}
          />
          <p className="control-help">Current opacity: {overlayOpacity.toFixed(2)}</p>
        </div>
      </aside>

      <section className="content">
        {error ? (
          <section className="panel status-panel">
            <div className="panel-header">
              <h3>Application error</h3>
              <p>{error}</p>
            </div>
          </section>
        ) : null}

        <div className="hero-grid">
          <section className="panel hero-card">
            <div className="panel-header">
              <h2>Latent summary</h2>
              <p>
                The atlas stays grayscale until you click Generate. The age slider uses years, while process sliders
                remain on a 0 to 1 scale.
              </p>
            </div>
            <div className="metric-row">
              <div>
                <span>Requested age</span>
                <strong>{ageYears.toFixed(2)} y</strong>
              </div>
              <div>
                <span>Processes</span>
                <strong>{runMetadata?.n_processes ?? processLatents.length ?? 0}</strong>
              </div>
            </div>
            <dl className="subject-meta">
              {Object.entries(currentMetadata).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{formatMetadataValue(key, value)}</dd>
                </div>
              ))}
            </dl>
          </section>

          {atlasUrls ? (
            <BrainViewer
              atlasImageUrl={atlasUrls.atlas_image_url}
              atlasSegmentationUrl={atlasUrls.atlas_segmentation_url}
              overlayImageUrl={activeOverlayImageUrl}
              overlayAbsMax={overlayAbsMax}
              overlayOpacity={overlayOpacity}
              mode={viewMode}
              overlayScaleMode={overlayScaleMode}
              roiMetadataById={roiMetadataById}
              roiValueById={roiValueById}
            />
          ) : null}
        </div>

        {(loadingDefaults || loadingInference) && !error && displayMode === "subject" ? (
          <section className="panel status-panel">
            <div className="panel-header">
              <h3>{loadingInference ? "Generating ROI changes" : "Loading subject defaults"}</h3>
              <p>
                {loadingInference
                  ? "Applying the selected age/process controls to the chosen subject."
                  : "Inferring default slider values from the selected subject."}
              </p>
            </div>
          </section>
        ) : null}

        {displayMode === "subject" ? (
          <SliderPanel
            ageYears={ageYears}
            processLatents={processLatents}
            disabled={controlsDisabled}
            onAgeChange={setAgeYears}
            onProcessChange={setProcessLatents}
            onGenerate={handleGenerate}
          />
        ) : (
          <section className="panel">
            <div className="panel-header">
              <h3>Population Pattern View</h3>
              <p>These overlays are precomputed per run from the selected checkpoint, so they load immediately.</p>
            </div>
            <p className="control-help">
              Showing: <strong>{populationPattern?.label ?? selectedPopulationPattern}</strong>
            </p>
          </section>
        )}

        <section className="panel chart-panel">
          <div className="panel-header">
            <h3>Top ROI changes</h3>
            <p>
              {displayMode === "population"
                ? chartMode === "percent"
                  ? "Bars show the average isolated population-level percent change for the selected factor. Percent is computed as 100 × Δ / |baseline ROI volume|."
                  : "Average percent changes are extremely small for this pattern, so the plot is showing raw ROI deltas instead."
                : chartMode === "percent"
                  ? "Bars show percent change relative to the selected subject's current ROI volume. Small baseline ROIs can therefore yield large percentages."
                  : "Percent changes are extremely small for this checkpoint, so the plot is showing raw ROI deltas instead."}
            </p>
          </div>
          <div className="chart-wrap">
            {orderedTopChanges.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={orderedTopChanges}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" hide />
                  <YAxis
                    tickFormatter={(value) =>
                      chartMode === "percent" ? `${Number(value).toFixed(1)}%` : Number(value).toFixed(1)
                    }
                  />
                  <Tooltip
                    formatter={(value: number, key: string, item: any) => {
                      if (key === "percent_change" || key === "delta") {
                        return [
                          chartMode === "percent"
                            ? `${Number(item.payload.percent_change).toFixed(1)}%`
                            : `${Number(item.payload.delta).toFixed(1)}`,
                          `Δ=${item.payload.delta.toFixed(1)} | %=${item.payload.percent_change.toFixed(1)} | baseline=${item.payload.baseline_value.toFixed(1)} | predicted=${item.payload.predicted_value.toFixed(1)}`,
                        ];
                      }
                      return [String(value), key];
                    }}
                  />
                  <Bar dataKey={chartMode === "percent" ? "percent_change" : "delta"} fill="#1f6f8b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-chart-state">
                {displayMode === "population"
                  ? "Select a population pattern to populate the top-ROI summary."
                  : "Select a subject, adjust sliders if needed, and click Generate to populate ROI changes."}
              </div>
            )}
          </div>
        </section>

        {latentSpace && runMetadata ? (
          <LatentSpaceExplorer
            rows={latentSpace.rows}
            nProcesses={runMetadata.n_processes}
            splitName={latentSpace.split_name}
            totalRows={latentSpace.total_rows}
            returnedRows={latentSpace.returned_rows}
          />
        ) : null}

        {displayMode === "subject" && inference?.debug ? (
          <section className="panel">
            <div className="panel-header">
              <h3>Generation Debug</h3>
              <p>Useful when a checkpoint appears weakly responsive to latent changes.</p>
            </div>
            {inference.timing ? (
              <p className="control-help">
                inference={inference.timing.inference_seconds.toFixed(2)}s | overlay={inference.timing.overlay_seconds.toFixed(2)}s
                {" | "}total={inference.timing.total_seconds.toFixed(2)}s
              </p>
            ) : null}
            <pre className="debug-block">{JSON.stringify(inference.debug, null, 2)}</pre>
          </section>
        ) : null}
      </section>
    </main>
  );
}
