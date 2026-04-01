import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

type BrainViewerProps = {
  atlasImageUrl: string;
  overlayImageUrl?: string | null;
  overlayAbsMax?: number;
  mode: "slices" | "volume";
};

export function BrainViewer({ atlasImageUrl, overlayImageUrl, overlayAbsMax = 1, mode }: BrainViewerProps) {
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
      const volumes: any[] = [{ url: atlasImageUrl, opacity: 1, colormap: "gray" }];
      if (overlayImageUrl) {
        volumes.push({
          url: overlayImageUrl,
          opacity: 0.7,
          colormap: "blue",
          colormapNegative: "red",
          cal_min: 0,
          cal_max: Math.max(overlayAbsMax, 1),
          cal_minNeg: -Math.max(overlayAbsMax, 1),
          cal_maxNeg: 0,
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
  }, [atlasImageUrl, overlayImageUrl, overlayAbsMax, mode]);

  return (
    <section className="panel brain-panel">
      <div className="panel-header">
        <h3>Atlas Viewer</h3>
        <p>Axial, coronal, sagittal slices by default, with optional 3D volume rendering.</p>
      </div>
      <div className="brain-canvas-shell">
        {error ? <div className="inline-error">{error}</div> : <canvas ref={canvasRef} className="brain-canvas" />}
      </div>
    </section>
  );
}
