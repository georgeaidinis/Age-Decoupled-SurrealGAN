import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

type BrainViewerProps = {
  atlasImageUrl: string;
  overlayImageUrl?: string | null;
  overlayAbsMax?: number;
  overlayOpacity?: number;
  mode: "slices" | "volume";
};

export function BrainViewer({
  atlasImageUrl,
  overlayImageUrl,
  overlayAbsMax = 1,
  overlayOpacity = 0.38,
  mode,
}: BrainViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      const volumes: any[] = [{ url: atlasImageUrl, opacity: 1, colormap: "gray" }];
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
  }, [atlasImageUrl, overlayImageUrl, overlayAbsMax, overlayOpacity, mode]);

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
      <div className="brain-canvas-shell">
        {error ? <div className="inline-error">{error}</div> : <canvas ref={canvasRef} className="brain-canvas" />}
      </div>
    </section>
  );
}
