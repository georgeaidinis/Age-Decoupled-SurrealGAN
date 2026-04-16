import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BrainViewer } from "./components/BrainViewer";
import { LatentSpaceExplorer } from "./components/LatentSpaceExplorer";
import { SliderPanel } from "./components/SliderPanel";
import {
  atlasManifest,
  getAtlasRoiMetadata,
  getLatentSpace,
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
  type SubjectDefaultsResponse,
  type SubjectRow,
} from "./lib/api";

type RunMetadata = {
  n_processes: number;
};

type ChartMetric = "delta_std" | "delta" | "percent_change";

type RoiChartRow = {
  roi_id?: number;
  name: string;
  percent_change: number;
  delta: number;
  delta_std: number;
  baseline_value: number;
  predicted_value: number;
};

function computePercentChange(baselineValue: number, predictedValue: number): number {
  const denominator = Math.max(Math.abs(baselineValue), 1.0);
  return ((predictedValue - baselineValue) / denominator) * 100.0;
}

function formatMetricValue(metric: ChartMetric, value: number): string {
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  if (metric === "percent_change") {
    return `${value.toFixed(1)}%`;
  }
  if (metric === "delta") {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function metricLabel(metric: ChartMetric): string {
  if (metric === "delta_std") {
    return "Standardized delta";
  }
  if (metric === "delta") {
    return "Raw delta";
  }
  return "Percent vs baseline";
}

export function App() {
  const [runs, setRuns] = useState<Array<{ run_name: string }>>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [runMetadata, setRunMetadata] = useState<RunMetadata | null>(null);
  const [splitName, setSplitName] = useState("application");
  const [subjects, setSubjects] = useState<SubjectRow[]>([]);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [subjectMetadata, setSubjectMetadata] = useState<Record<string, string | number | null> | null>(null);
  const [subjectDefaults, setSubjectDefaults] = useState<SubjectDefaultsResponse["defaults"] | null>(null);
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [ageYears, setAgeYears] = useState(20);
  const [processLatents, setProcessLatents] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<"slices" | "volume">("slices");
  const [displayMode, setDisplayMode] = useState<"subject" | "population">("subject");
  const [overlayScaleMode, setOverlayScaleMode] = useState<"relative" | "absolute">("relative");
  const [roiChartMetric, setRoiChartMetric] = useState<ChartMetric>("delta_std");
  const [overlayOpacity, setOverlayOpacity] = useState(0.38);
  const [absoluteOverlayScale, setAbsoluteOverlayScale] = useState(1.0);
  const [colorblindMode, setColorblindMode] = useState(false);
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
      setSubjectDefaults(null);
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
    setSubjectDefaults(null);
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
        setSubjectDefaults(payload.defaults);
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

  const roiRows = useMemo<RoiChartRow[]>(
    () =>
      ((displayMode === "population" ? populationPattern?.roi_table : inference?.roi_table) ?? []).map((row) => ({
        roi_id: Number.isFinite(Number(row.roi_id)) ? Number(row.roi_id) : undefined,
        name: row.roi_name ?? row.roi_full_name ?? "Unknown ROI",
        delta: Number(row.delta ?? 0),
        delta_std: Number(row.delta_std ?? 0),
        baseline_value: Number(row.baseline_value ?? 0),
        predicted_value: Number(row.predicted_value ?? 0),
        percent_change: computePercentChange(Number(row.baseline_value ?? 0), Number(row.predicted_value ?? 0)),
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
        roiRows
          .filter((row) => row.roi_id !== undefined)
          .map((row) => [
            Number(row.roi_id),
            {
              delta: row.delta,
              delta_std: row.delta_std,
              percent_change: row.percent_change,
              baseline_value: row.baseline_value,
              predicted_value: row.predicted_value,
            },
          ]),
      ) as Record<number, { delta: number; delta_std: number; percent_change: number; baseline_value: number; predicted_value: number }>,
    [roiRows],
  );

  const chartValueKey = roiChartMetric;

  const orderedTopChanges = useMemo(
    () =>
      [...roiRows]
        .sort((lhs, rhs) => Math.abs(Number(rhs[chartValueKey])) - Math.abs(Number(lhs[chartValueKey])))
        .slice(0, 25),
    [chartValueKey, roiRows],
  );

  const overlayAbsMax = useMemo(() => {
    const values = roiRows.map((row) => Math.abs(Number(row.delta_std ?? 0)));
    const maxValue = values.reduce((current, value) => Math.max(current, value), 0);
    if (overlayScaleMode === "relative") {
      return Math.max(maxValue, 1e-4);
    }
    return Math.max(absoluteOverlayScale, 1e-4);
  }, [absoluteOverlayScale, overlayScaleMode, roiRows]);

  const currentMetadata =
    displayMode === "population"
      ? {
          mode: "population",
          pattern: populationPattern?.label ?? selectedPopulationPattern,
          reference_split: populationManifest?.reference_split ?? "",
          reference_bucket: populationManifest?.reference_cohort_bucket ?? "",
          ...(selectedPopulationPattern === "age"
            ? {
                age_range:
                  populationManifest?.age_pattern != null
                    ? `${populationManifest.age_pattern.age_min_years} to ${populationManifest.age_pattern.age_max_years}`
                    : "",
              }
            : {
                process_anchor_age_years: populationManifest?.process_anchor_age_years ?? "",
              }),
        }
      : inference?.metadata ?? subjectMetadata ?? {};

  const activeOverlayImageUrl = displayMode === "population" ? populationPattern?.overlay_image_url ?? null : overlayImageUrl;
  const controlsDisabled = !selectedRun || selectedRow === null || loadingDefaults || loadingInference;
  const groundTruthAgeYears =
    displayMode === "subject"
      ? Number(inference?.inference.ground_truth_age_years ?? subjectMetadata?.age ?? ageYears)
      : null;
  const predictedAgeYears =
    displayMode === "subject"
      ? Number(inference?.inference.predicted_age_years ?? subjectDefaults?.predicted_age_years ?? Number.NaN)
      : null;
  const inferredProcessDefaults =
    displayMode === "subject" ? inference?.inference.default_process_latents ?? subjectDefaults?.process_latents ?? [] : [];

  const formatMetadataValue = (key: string, value: string | number | null) => {
    if (value === null || value === undefined || value === "") {
      return "";
    }
    if (typeof value === "number") {
      if (key.toLowerCase().includes("age")) {
        return value.toFixed(2);
      }
      if (Number.isInteger(value)) {
        return String(value);
      }
      return value.toFixed(3);
    }
    return String(value);
  };

  const chartDescription =
    displayMode === "population"
      ? roiChartMetric === "delta_std"
        ? "Bars show the isolated population-level effect in normalized ROI standard deviations. This is the most stable cross-ROI view."
        : roiChartMetric === "delta"
          ? "Bars show the isolated population-level mean raw ROI change."
          : "Bars show 100 × (predicted - baseline) / |baseline| using the displayed population-level baseline and predicted values."
      : roiChartMetric === "delta_std"
        ? "Bars show subject-level anchored edits in normalized ROI standard deviations. This suppresses the tiny-ROI percentage explosion that made qualitative inspection unreliable."
        : roiChartMetric === "delta"
          ? "Bars show subject-level anchored edits in raw ROI units."
          : "Bars show 100 × (predicted - baseline) / |baseline| for the selected subject. Tiny ROIs can still yield large percentages, so use this as a secondary diagnostic rather than the default ranking view.";

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
            <button className={viewMode === "slices" ? "active" : ""} type="button" onClick={() => setViewMode("slices")}>
              2D slices
            </button>
            <button className={viewMode === "volume" ? "active" : ""} type="button" onClick={() => setViewMode("volume")}>
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
            The atlas overlay is shown in standardized delta units, so the scale is less dominated by tiny ROI denominators.
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
              max={6}
              step={0.1}
              value={absoluteOverlayScale}
              onChange={(event) => setAbsoluteOverlayScale(Number(event.target.value))}
            />
            <p className="control-help">Clip the atlas overlay at ±{absoluteOverlayScale.toFixed(1)} SD of normalized ROI change.</p>
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

        <div className="control-group">
          <label>Bar chart metric</label>
          <div className="segmented-control segmented-control-compact">
            <button className={roiChartMetric === "delta_std" ? "active" : ""} type="button" onClick={() => setRoiChartMetric("delta_std")}>
              SD
            </button>
            <button className={roiChartMetric === "delta" ? "active" : ""} type="button" onClick={() => setRoiChartMetric("delta")}>
              Raw
            </button>
            <button
              className={roiChartMetric === "percent_change" ? "active" : ""}
              type="button"
              onClick={() => setRoiChartMetric("percent_change")}
            >
              %
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Color palette</label>
          <div className="segmented-control">
            <button className={!colorblindMode ? "active" : ""} type="button" onClick={() => setColorblindMode(false)}>
              Default
            </button>
            <button className={colorblindMode ? "active" : ""} type="button" onClick={() => setColorblindMode(true)}>
              Colorblind
            </button>
          </div>
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
                Subject mode shows anchored edits relative to the selected subject’s inferred latent state. Population mode shows
                precomputed isolated factors from the selected checkpoint.
              </p>
            </div>
            <div className="metric-row">
              <div>
                <span>Requested age</span>
                <strong>{ageYears.toFixed(1)} y</strong>
              </div>
              <div>
                <span>Ground truth age</span>
                <strong>{groundTruthAgeYears !== null && Number.isFinite(groundTruthAgeYears) ? `${groundTruthAgeYears.toFixed(1)} y` : "N/A"}</strong>
              </div>
            </div>
            <div className="metric-row">
              <div>
                <span>Model-predicted age</span>
                <strong>
                  {displayMode === "subject" && predictedAgeYears !== null && Number.isFinite(predictedAgeYears)
                    ? `${predictedAgeYears.toFixed(1)} y`
                    : "N/A"}
                </strong>
              </div>
              <div>
                <span>Processes</span>
                <strong>{runMetadata?.n_processes ?? processLatents.length ?? 0}</strong>
              </div>
            </div>
            {displayMode === "subject" && inferredProcessDefaults.length > 0 ? (
              <div className="latent-chip-group">
                {inferredProcessDefaults.map((value, index) => (
                  <span key={`latent-default-${index}`} className="latent-chip">
                    r{index + 1}={value.toFixed(2)}
                  </span>
                ))}
              </div>
            ) : null}
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
              colorblindMode={colorblindMode}
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
                  ? "Applying the selected age/process controls as anchored edits around the inferred subject state."
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
            <p>{chartDescription}</p>
          </div>
          <div className="chart-wrap">
            {orderedTopChanges.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={orderedTopChanges}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" hide />
                  <YAxis tickFormatter={(value) => formatMetricValue(roiChartMetric, Number(value))} />
                  <Tooltip
                    formatter={(value: number, key: string, item: any) => {
                      if (key === chartValueKey) {
                        return [formatMetricValue(roiChartMetric, Number(value)), metricLabel(roiChartMetric)];
                      }
                      return [String(value), key];
                    }}
                    content={({ active, payload }: any) => {
                      if (!active || !payload || payload.length === 0) {
                        return null;
                      }
                      const row = payload[0]?.payload as RoiChartRow | null;
                      if (!row) {
                        return null;
                      }
                      return (
                        <div className="latent-tooltip">
                          <strong>{row.name}</strong>
                          <p>
                            Δstd={row.delta_std.toFixed(2)} | Δ={row.delta.toFixed(1)} | %={row.percent_change.toFixed(1)}
                          </p>
                          <p>
                            baseline={row.baseline_value.toFixed(1)} | predicted={row.predicted_value.toFixed(1)}
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey={chartValueKey} fill="#1f6f8b" radius={[4, 4, 0, 0]} />
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
