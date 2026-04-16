import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

import type { LatentSpaceRow } from "../lib/api";

type LatentSpaceExplorerProps = {
  rows: LatentSpaceRow[];
  nProcesses: number;
  splitName: string;
  totalRows: number;
  returnedRows: number;
};

type ExplorerRow = LatentSpaceRow & {
  xValue: number;
  yValue: number;
  fill: string;
  colorValueLabel: string;
};

const DISCRETE_PALETTE = [
  "#1f6f8b",
  "#d97430",
  "#2f8f6b",
  "#7750c8",
  "#c64646",
  "#8c6c1d",
  "#1c7f98",
  "#5567d8",
  "#b55391",
  "#5a7f1d",
];

const AXIS_PRIORITY = [
  "age",
  "age_latent",
  "r1",
  "r2",
  "r3",
  "r4",
  "r5",
  "r6",
  "r7",
  "r8",
  "r9",
  "r10",
  "r11",
  "MMSE",
  "DSST",
  "SPARE_AD",
  "SPARE_BA",
  "Education_Years",
  "APOE4_Alleles",
  "DLICV",
  "BMI",
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatNumeric(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toFixed(2);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function rankKey(key: string): number {
  const index = AXIS_PRIORITY.indexOf(key);
  return index >= 0 ? index : AXIS_PRIORITY.length + key.charCodeAt(0);
}

function continuousColor(value: number | null, values: number[]): string {
  if (value === null || !Number.isFinite(value) || values.length === 0) {
    return "#5f7c8a";
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const ratio = maxValue > minValue ? clamp01((value - minValue) / (maxValue - minValue)) : 0.5;
  const hue = 220 - ratio * 185;
  return `hsl(${hue.toFixed(0)} 70% 48%)`;
}

export function LatentSpaceExplorer({
  rows,
  nProcesses,
  splitName,
  totalRows,
  returnedRows,
}: LatentSpaceExplorerProps) {
  const numericKeys = useMemo(() => {
    const discovered = new Set<string>(["age", "age_latent"]);
    for (let index = 1; index <= nProcesses; index += 1) {
      discovered.add(`r${index}`);
    }
    for (const row of rows) {
      for (const [key, value] of Object.entries(row)) {
        if (key === "subject_id") {
          continue;
        }
        if (typeof value === "number" && Number.isFinite(value)) {
          discovered.add(key);
          continue;
        }
        if (typeof value === "string" && value.trim() !== "") {
          const parsed = Number(value);
          if (Number.isFinite(parsed)) {
            discovered.add(key);
          }
        }
      }
    }
    return [...discovered].sort((lhs, rhs) => rankKey(lhs) - rankKey(rhs));
  }, [nProcesses, rows]);

  const categoricalKeys = useMemo(() => {
    const discovered = new Set<string>(["diagnosis_group", "study", "cohort_bucket", "sex"]);
    for (const row of rows) {
      for (const [key, value] of Object.entries(row)) {
        if (key === "subject_id" || numericKeys.includes(key)) {
          continue;
        }
        if (value !== null && value !== undefined && String(value).trim() !== "") {
          discovered.add(key);
        }
      }
    }
    return [...discovered].sort((lhs, rhs) => lhs.localeCompare(rhs));
  }, [numericKeys, rows]);

  const colorOptions = useMemo(() => [...numericKeys, ...categoricalKeys], [categoricalKeys, numericKeys]);

  const defaultYAxis = numericKeys.includes("r2") ? "r2" : numericKeys.find((key) => key !== "r1") ?? "age_latent";
  const [xAxisKey, setXAxisKey] = useState<string>(numericKeys.includes("r1") ? "r1" : numericKeys[0] ?? "age");
  const [yAxisKey, setYAxisKey] = useState<string>(defaultYAxis);
  const [colorBy, setColorBy] = useState<string>(numericKeys.includes("age") ? "age" : numericKeys[0] ?? "diagnosis_group");

  useEffect(() => {
    if (!numericKeys.includes(xAxisKey)) {
      setXAxisKey(numericKeys.includes("r1") ? "r1" : numericKeys[0] ?? "age");
    }
    if (!numericKeys.includes(yAxisKey)) {
      setYAxisKey(numericKeys.includes("r2") ? "r2" : numericKeys.find((key) => key !== xAxisKey) ?? numericKeys[0] ?? "age_latent");
    }
    if (!colorOptions.includes(colorBy)) {
      setColorBy(colorOptions[0] ?? "age");
    }
  }, [colorBy, colorOptions, numericKeys, xAxisKey, yAxisKey]);

  const numericColorValues = useMemo(
    () =>
      colorOptions.includes(colorBy) && numericKeys.includes(colorBy)
        ? rows
            .map((row) => {
              const value = row[colorBy];
              return typeof value === "number" ? value : Number(value);
            })
            .filter((value) => Number.isFinite(value))
        : [],
    [colorBy, colorOptions, numericKeys, rows],
  );

  const categoricalColors = useMemo(() => {
    if (!categoricalKeys.includes(colorBy)) {
      return {};
    }
    const values = Array.from(new Set(rows.map((row) => String(row[colorBy] ?? "unknown")))).sort();
    return Object.fromEntries(values.map((value, index) => [value, DISCRETE_PALETTE[index % DISCRETE_PALETTE.length]]));
  }, [categoricalKeys, colorBy, rows]);

  const chartRows = useMemo(
    () =>
      rows
        .map((row) => {
          const xCandidate = typeof row[xAxisKey] === "number" ? row[xAxisKey] : Number(row[xAxisKey]);
          const yCandidate = typeof row[yAxisKey] === "number" ? row[yAxisKey] : Number(row[yAxisKey]);
          if (!Number.isFinite(xCandidate) || !Number.isFinite(yCandidate)) {
            return null;
          }
          if (numericKeys.includes(colorBy)) {
            const numericValue = typeof row[colorBy] === "number" ? row[colorBy] : Number(row[colorBy]);
            return {
              ...row,
              xValue: xCandidate,
              yValue: yCandidate,
              fill: continuousColor(Number.isFinite(numericValue) ? numericValue : null, numericColorValues),
              colorValueLabel: Number.isFinite(numericValue) ? formatNumeric(numericValue) : "N/A",
            } satisfies ExplorerRow;
          }
          const categoricalValue = String(row[colorBy] ?? "unknown");
          return {
            ...row,
            xValue: xCandidate,
            yValue: yCandidate,
            fill: categoricalColors[categoricalValue] ?? "#5f7c8a",
            colorValueLabel: categoricalValue,
          } satisfies ExplorerRow;
        })
        .filter((row): row is ExplorerRow => row !== null),
    [categoricalColors, colorBy, numericColorValues, numericKeys, rows, xAxisKey, yAxisKey],
  );

  const discreteLegend = useMemo(() => {
    if (!categoricalKeys.includes(colorBy)) {
      return [];
    }
    return Object.entries(categoricalColors).map(([label, fill]) => ({ label, fill }));
  }, [categoricalColors, categoricalKeys, colorBy]);

  const continuousLegendBounds = useMemo(() => {
    if (!numericKeys.includes(colorBy) || numericColorValues.length === 0) {
      return null;
    }
    return {
      min: Math.min(...numericColorValues),
      max: Math.max(...numericColorValues),
    };
  }, [colorBy, numericColorValues, numericKeys]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Latent Space Explorer</h3>
        <p>
          Cross-sectional view of the saved latent predictions for the <strong>{splitName}</strong> split. Use this to test
          whether a process axis is distinct, age-linked, diagnosis-linked, or just shadowing another axis.
        </p>
      </div>

      <div className="latent-controls">
        <div className="control-group compact">
          <label>X axis</label>
          <select value={xAxisKey} onChange={(event) => setXAxisKey(event.target.value)}>
            {numericKeys.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label>Y axis</label>
          <select value={yAxisKey} onChange={(event) => setYAxisKey(event.target.value)}>
            {numericKeys.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label>Color by</label>
          <select value={colorBy} onChange={(event) => setColorBy(event.target.value)}>
            {colorOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      </div>

      <p className="control-help">
        Showing {returnedRows.toLocaleString()} of {totalRows.toLocaleString()} saved rows for this split.
      </p>

      {continuousLegendBounds ? (
        <div className="latent-age-legend" aria-label="Continuous color legend">
          <span>{formatNumeric(continuousLegendBounds.min)}</span>
          <div className="latent-age-legend-bar" />
          <span>{formatNumeric(continuousLegendBounds.max)}</span>
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
              <XAxis type="number" dataKey="xValue" name={xAxisKey} tickFormatter={(value) => formatNumeric(Number(value))} />
              <YAxis type="number" dataKey="yValue" name={yAxisKey} tickFormatter={(value) => formatNumeric(Number(value))} />
              <Tooltip
                cursor={{ strokeDasharray: "6 6" }}
                labelFormatter={() => ""}
                content={({ active, payload }: any) => {
                  if (!active || !payload || payload.length === 0) {
                    return null;
                  }
                  const row = payload[0]?.payload as ExplorerRow | null;
                  if (!row) {
                    return null;
                  }
                  return (
                    <div className="latent-tooltip">
                      <strong>{String(row.subject_id ?? "unknown")}</strong>
                      <p>
                        {xAxisKey}={formatNumeric(row.xValue)} | {yAxisKey}={formatNumeric(row.yValue)}
                      </p>
                      <p>
                        age={formatNumeric(row.age)} | age_latent={formatNumeric(row.age_latent)}
                      </p>
                      <p>
                        diagnosis={String(row.diagnosis_group ?? "N/A")} | study={String(row.study ?? "N/A")}
                      </p>
                      <p>
                        {colorBy}={row.colorValueLabel}
                      </p>
                    </div>
                  );
                }}
              />
              <Scatter
                data={chartRows}
                shape={(props: any) => (
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={2.1}
                    fill={props.payload.fill}
                    fillOpacity={0.78}
                    stroke="rgba(20, 36, 50, 0.12)"
                    strokeWidth={0.25}
                  />
                )}
              />
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-chart-state">No saved latent rows are available for this split.</div>
        )}
      </div>
    </section>
  );
}
