import { useEffect, useRef } from "react";
import { Niivue } from "@niivue/niivue";

type BrainViewerProps = {
  atlasImageUrl: string;
  atlasSegmentationUrl: string;
  mode: "slices" | "volume";
};

export function BrainViewer({ atlasImageUrl, atlasSegmentationUrl, mode }: BrainViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    const nv = new Niivue({
      isResizeCanvas: true,
      show3Dcrosshair: mode === "volume",
      dragMode: 2,
    });
    nv.attachToCanvas(canvasRef.current);
    nv.loadVolumes([
      { url: atlasImageUrl, opacity: 1, colormap: "gray" },
      { url: atlasSegmentationUrl, opacity: 0.45, colormap: "warm" },
    ]);
    nv.setSliceType(mode === "volume" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
    return () => {
      nv.dispose();
    };
  }, [atlasImageUrl, atlasSegmentationUrl, mode]);

  return (
    <section className="panel brain-panel">
      <div className="panel-header">
        <h3>Atlas Viewer</h3>
        <p>Axial, coronal, sagittal slices by default, with optional 3D volume rendering.</p>
      </div>
      <canvas ref={canvasRef} className="brain-canvas" />
    </section>
  );
}
