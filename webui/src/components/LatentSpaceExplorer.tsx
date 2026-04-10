import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { LatentSpaceRow } from "../lib/api";

type AxisKey = "age" | "age_latent" | `r${number}`;
type ColorBy = "age" | "diagnosis_group" | "study" | "cohort_bucket";

type LatentSpaceExplorerProps = {
  rows: LatentSpaceRow[];
  nProcesses: number;
  splitName: string;
  totalRows: number;
  returnedRows: number;
};

const DISCRETE_PALETTE = [
  "#1f6f8b",
  "#d97430",
  "#2f8f6b",
  "#8a4fff",
  "#c64646",
  "#8c6c1d",
  "#147a96",
  "#6e5bd4",
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatAxisValue(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toFixed(2);
}

function continuousAgeColor(age: number | null, ages: number[]): string {
  if (age === null || !Number.isFinite(age) || ages.length === 0) {
    return "#5f7c8a";
  }
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const ratio = maxAge > minAge ? clamp01((age - minAge) / (maxAge - minAge)) : 0.5;
  const hue = 220 - ratio * 170;
  return `hsl(${hue.toFixed(0)} 70% 48%)`;
}

export function LatentSpaceExplorer({
  rows,
  nProcesses,
  splitName,
  totalRows,
  returnedRows,
}: LatentSpaceExplorerProps) {
  const axisOptions = useMemo<AxisKey[]>(
    () => ["age", "age_latent", ...Array.from({ length: nProcesses }, (_, index) => `r${index + 1}` as AxisKey)],
    [nProcesses],
  );
  const [xAxisKey, setXAxisKey] = useState<AxisKey>("r1");
  const [yAxisKey, setYAxisKey] = useState<AxisKey>(nProcesses >= 2 ? "r2" : "age_latent");
  const [colorBy, setColorBy] = useState<ColorBy>("age");

  const ages = useMemo(
    () =>
      rows
        .map((row) => (typeof row.age === "number" && Number.isFinite(row.age) ? row.age : null))
        .filter((value): value is number => value !== null),
    [rows],
  );

  const categoricalColors = useMemo(() => {
    const mapping: Record<ColorBy, Record<string, string>> = {
      age: {},
      diagnosis_group: {},
      study: {},
      cohort_bucket: {},
    };
    (["diagnosis_group", "study", "cohort_bucket"] as const).forEach((key) => {
      const values = Array.from(new Set(rows.map((row) => String(row[key] ?? "unknown")))).sort();
      mapping[key] = Object.fromEntries(values.map((value, index) => [value, DISCRETE_PALETTE[index % DISCRETE_PALETTE.length]]));
    });
    return mapping;
  }, [rows]);

  const chartRows = useMemo(
    () =>
      rows
        .map((row) => {
          const xValue = Number(row[xAxisKey] ?? NaN);
          const yValue = Number(row[yAxisKey] ?? NaN);
          if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) {
            return null;
          }
          const age = typeof row.age === "number" && Number.isFinite(row.age) ? row.age : null;
          const labelValue = colorBy === "age" ? (age === null ? "unknown" : `${age.toFixed(1)} y`) : String(row[colorBy] ?? "unknown");
          const fill =
            colorBy === "age"
              ? continuousAgeColor(age, ages)
              : categoricalColors[colorBy][String(row[colorBy] ?? "unknown")] ?? "#5f7c8a";
          return {
            ...row,
            xValue,
            yValue,
            fill,
            labelValue,
          };
        })
        .filter((row): row is LatentSpaceRow & { xValue: number; yValue: number; fill: string; labelValue: string } => row !== null),
    [ages, categoricalColors, colorBy, rows, xAxisKey, yAxisKey],
  );

  const discreteLegend = useMemo(() => {
    if (colorBy === "age") {
      return [];
    }
    return Object.entries(categoricalColors[colorBy]).map(([label, fill]) => ({ label, fill }));
  }, [categoricalColors, colorBy]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Latent Space Explorer</h3>
        <p>
          Cross-sectional view of the saved latent predictions for the <strong>{splitName}</strong> split. This is useful for
          seeing whether a process axis is really distinct or just shadowing age or another process.
        </p>
      </div>

      <div className="latent-controls">
        <div className="control-group compact">
          <label>X axis</label>
          <select value={xAxisKey} onChange={(event) => setXAxisKey(event.target.value as AxisKey)}>
            {axisOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label>Y axis</label>
          <select value={yAxisKey} onChange={(event) => setYAxisKey(event.target.value as AxisKey)}>
            {axisOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label>Color by</label>
          <select value={colorBy} onChange={(event) => setColorBy(event.target.value as ColorBy)}>
            <option value="age">Age</option>
            <option value="diagnosis_group">Diagnosis group</option>
            <option value="study">Study</option>
            <option value="cohort_bucket">Cohort bucket</option>
          </select>
        </div>
      </div>

      <p className="control-help">
        Showing {returnedRows.toLocaleString()} of {totalRows.toLocaleString()} subjects saved for this split.
      </p>

      {colorBy === "age" ? (
        <div className="latent-age-legend" aria-label="Age color legend">
          <span>{ages.length > 0 ? `${Math.min(...ages).toFixed(0)} y` : "young"}</span>
          <div className="latent-age-legend-bar" />
          <span>{ages.length > 0 ? `${Math.max(...ages).toFixed(0)} y` : "old"}</span>
        </div>
      ) : discreteLegend.length > 0 ? (
        <div className="latent-legend">
          {discreteLegend.map((entry) => (
            <span key={entry.label} className="latent-legend-chip">
              <i style={{ background: entry.fill }} />
              {entry.label}
            </span>
          ))}
        </div>
      ) : null}

      <div className="latent-chart-wrap">
        {chartRows.length > 0 ? (
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 16, right: 18, bottom: 12, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="xValue"
                name={xAxisKey}
                tickFormatter={(value) => formatAxisValue(Number(value))}
              />
              <YAxis
                type="number"
                dataKey="yValue"
                name={yAxisKey}
                tickFormatter={(value) => formatAxisValue(Number(value))}
              />
              <Tooltip
                cursor={{ strokeDasharray: "4 4" }}
                labelFormatter={() => ""}
                content={({ active, payload }: any) => {
                  if (!active || !payload || payload.length === 0) {
                    return null;
                  }
                  const row = payload[0]?.payload as (LatentSpaceRow & {
                    xValue: number;
                    yValue: number;
                    fill: string;
                    labelValue: string;
                  }) | null;
                  if (!row) {
                    return null;
                  }
                  return (
                    <div className="latent-tooltip">
                      <strong>{row.subject_id}</strong>
                      <p>
                        {xAxisKey}={formatAxisValue(row.xValue)} | {yAxisKey}={formatAxisValue(row.yValue)}
                      </p>
                      <p>
                        age={formatAxisValue(typeof row.age === "number" ? row.age : Number.NaN)} | age_latent=
                        {formatAxisValue(Number(row.age_latent))}
                      </p>
                      <p>
                        diagnosis={row.diagnosis_group} | study={row.study}
                      </p>
                      <p>color={row.labelValue}</p>
                    </div>
                  );
                }}
              />
              <Scatter data={chartRows}>
                {chartRows.map((row, index) => (
                  <Cell key={`${row.subject_id}-${index}`} fill={row.fill} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-chart-state">No saved latent rows are available for this split.</div>
        )}
      </div>
    </section>
  );
}
