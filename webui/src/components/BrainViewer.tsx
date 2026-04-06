import { useEffect, useMemo, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

import type { RoiMetadataRow } from "../lib/api";

type RoiValueRow = {
  delta: number;
  percent_change: number;
  baseline_value?: number;
  predicted_value?: number;
};

type BrainViewerProps = {
  atlasImageUrl: string;
  atlasSegmentationUrl: string;
  overlayImageUrl?: string | null;
  overlayAbsMax?: number;
  overlayOpacity?: number;
  mode: "slices" | "volume";
  overlayScaleMode: "relative" | "absolute";
  roiMetadataById: Record<number, RoiMetadataRow>;
  roiValueById: Record<number, RoiValueRow>;
};

type LocationPayload = {
  values?: Array<{ name?: string; value?: number; id?: string | number }>;
  vox?: number[];
  mm?: number[];
};

type CrosshairInfo = {
  roiId: number | null;
  roiName: string;
  roiFullName: string;
  primaryLabel: string;
  primaryValue: string;
  secondaryLabel: string;
  secondaryValue: string;
  voxelLabel: string;
};

function formatSigned(value: number, suffix = ""): string {
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  const prefix = normalized > 0 ? "+" : "";
  return `${prefix}${normalized.toFixed(2)}${suffix}`;
}

function formatValue(value: number): string {
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  return normalized.toFixed(2);
}

export function BrainViewer({
  atlasImageUrl,
  atlasSegmentationUrl,
  overlayImageUrl,
  overlayAbsMax = 1,
  overlayOpacity = 0.38,
  mode,
  overlayScaleMode,
  roiMetadataById,
  roiValueById,
}: BrainViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [crosshairInfo, setCrosshairInfo] = useState<CrosshairInfo | null>(null);

  const defaultCrosshairInfo = useMemo<CrosshairInfo>(
    () => ({
      roiId: null,
      roiName: "Move the crosshairs",
      roiFullName: "Crosshair readout updates with the ROI label and the current change value.",
      primaryLabel: overlayScaleMode === "relative" ? "Relative overlay" : "Absolute change",
      primaryValue: "N/A",
      secondaryLabel: "ROI delta",
      secondaryValue: "N/A",
      voxelLabel: "Voxel: N/A",
    }),
    [overlayScaleMode],
  );

  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    setError(null);
    const nv = new Niivue({
      isResizeCanvas: true,
      show3Dcrosshair: mode === "volume",
      dragMode: 2,
    });
    let disposed = false;
    try {
      nv.attachToCanvas(canvasRef.current);
      nv.onLocationChange = (location: unknown) => {
        const payload = location as LocationPayload;
        const values = Array.isArray(payload.values) ? payload.values : [];
        const segmentationBasename = atlasSegmentationUrl.split("/").pop()?.toLowerCase() ?? "";
        const segmentationEntry =
          values.find((item) => {
            const name = String(item?.name ?? "").toLowerCase();
            return name.includes("segmentation") || (segmentationBasename.length > 0 && name.includes(segmentationBasename));
          }) ?? values.find((item) => String(item?.name ?? "").toLowerCase().includes("seg"));
        const segmentationValue = segmentationEntry ? Number(segmentationEntry.value) : NaN;
        const roiId = Number.isFinite(segmentationValue) ? Math.round(segmentationValue) : 0;
        const metadata = roiId > 0 ? roiMetadataById[roiId] : undefined;
        const roiValues = roiId > 0 ? roiValueById[roiId] : undefined;
        const absolutePercent = Number(roiValues?.percent_change ?? 0);
        const absoluteDelta = Number(roiValues?.delta ?? 0);
        const relativePercent = overlayAbsMax > 1e-6 ? (absolutePercent / overlayAbsMax) * 100.0 : 0.0;
        const vox = Array.isArray(payload.vox) ? payload.vox.slice(0, 3).map((value) => Math.round(Number(value))) : [];
        const voxelLabel = vox.length === 3 ? `Voxel: ${vox[0]}, ${vox[1]}, ${vox[2]}` : "Voxel: N/A";

        if (!metadata) {
          const rawValues = values
            .map((item) => {
              const rawName = String(item?.name ?? "unknown");
              const compactName = rawName.length > 28 ? `${rawName.slice(0, 28)}...` : rawName;
              return `${compactName}=${formatValue(Number(item?.value ?? 0))}`;
            })
            .join(" | ");
          setCrosshairInfo({
            roiId: null,
            roiName: "Background / unlabeled voxel",
            roiFullName: rawValues
              ? `The current crosshair location does not map to a retained MUSE ROI. Raw values: ${rawValues}`
              : "The current crosshair location does not map to a retained MUSE ROI.",
            primaryLabel: overlayScaleMode === "relative" ? "Relative overlay" : "Absolute change",
            primaryValue: "0.00",
            secondaryLabel: "ROI delta",
            secondaryValue: "0.00",
            voxelLabel,
          });
          return;
        }

        setCrosshairInfo({
          roiId,
          roiName: metadata.roi_name || metadata.roi_full_name || `ROI ${roiId}`,
          roiFullName: metadata.roi_full_name || metadata.roi_name || `ROI ${roiId}`,
          primaryLabel: overlayScaleMode === "relative" ? "Relative overlay" : "Absolute change",
          primaryValue:
            overlayScaleMode === "relative"
              ? `${formatSigned(relativePercent, "%")} of current display range`
              : formatSigned(absolutePercent, "%"),
          secondaryLabel: "ROI delta",
          secondaryValue: `${formatSigned(absoluteDelta)} raw units`,
          voxelLabel,
        });
      };
      nv.addColormap("roi-positive", {
        R: [0, 42, 88, 128],
        G: [0, 112, 178, 220],
        B: [0, 212, 255, 255],
        A: [0, 24, 92, 160],
        I: [0, 96, 180, 255],
      });
      nv.addColormap("roi-negative", {
        R: [0, 172, 224, 255],
        G: [0, 78, 66, 52],
        B: [0, 78, 52, 52],
        A: [0, 24, 92, 160],
        I: [0, 96, 180, 255],
      });
      const volumes: any[] = [
        { url: atlasImageUrl, opacity: 1, colormap: "gray" },
        { url: atlasSegmentationUrl, opacity: 0, colormap: "gray" },
      ];
      if (overlayImageUrl) {
        const scale = Math.max(overlayAbsMax, 1e-4);
        const threshold = Math.max(scale * 0.025, 1e-5);
        volumes.push({
          url: overlayImageUrl,
          opacity: mode === "volume" ? Math.min(overlayOpacity, 0.22) : overlayOpacity,
          colormap: "roi-positive",
          colormapNegative: "roi-negative",
          cal_min: threshold,
          cal_max: scale,
          cal_minNeg: -scale,
          cal_maxNeg: -threshold,
        });
      }
      void nv
        .loadVolumes(volumes)
        .then(() => {
          if (!disposed) {
            nv.setSliceType(mode === "volume" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
            nv.onLocationChange({
              values: [],
              vox: [0, 0, 0],
            });
          }
        })
        .catch((loadError) => {
          if (!disposed) {
            setError(loadError instanceof Error ? loadError.message : "Failed to load atlas volumes.");
          }
        });
    } catch (attachError) {
      setError(attachError instanceof Error ? attachError.message : "Failed to initialize brain viewer.");
    }
    return () => {
      disposed = true;
      if (canvasRef.current) {
        const context = canvasRef.current.getContext("2d");
        context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
  }, [
    atlasImageUrl,
    atlasSegmentationUrl,
    overlayImageUrl,
    overlayAbsMax,
    overlayOpacity,
    mode,
    overlayScaleMode,
    roiMetadataById,
    roiValueById,
  ]);

  const displayInfo = crosshairInfo ?? defaultCrosshairInfo;

  return (
    <section className="panel brain-panel">
      <div className="panel-header">
        <h3>Atlas Viewer</h3>
        <p>Axial, coronal, sagittal slices by default, with optional 3D volume rendering and signed ROI overlays.</p>
      </div>
      <div className="overlay-legend" aria-label="Overlay legend">
        <span className="overlay-legend-label negative">Red: negative change, ROI volume loss</span>
        <div className="overlay-legend-bar" />
        <span className="overlay-legend-label positive">Blue: positive change, ROI enlargement</span>
      </div>
      <div className="crosshair-readout">
        <div className="crosshair-readout-main">
          <span className="crosshair-label">{displayInfo.roiId !== null ? `ROI ${displayInfo.roiId}` : "ROI"}</span>
          <strong>{displayInfo.roiName}</strong>
          <p>{displayInfo.roiFullName}</p>
        </div>
        <div className="crosshair-stat">
          <span>{displayInfo.primaryLabel}</span>
          <strong>{displayInfo.primaryValue}</strong>
        </div>
        <div className="crosshair-stat">
          <span>{displayInfo.secondaryLabel}</span>
          <strong>{displayInfo.secondaryValue}</strong>
        </div>
        <div className="crosshair-stat">
          <span>Location</span>
          <strong>{displayInfo.voxelLabel}</strong>
        </div>
      </div>
      <div className="brain-canvas-shell">
        {error ? <div className="inline-error">{error}</div> : <canvas ref={canvasRef} className="brain-canvas" />}
      </div>
    </section>
  );
}
