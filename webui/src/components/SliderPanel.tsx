type SliderPanelProps = {
  processLatents: number[];
  onChange: (values: number[]) => void;
};

export function SliderPanel({ processLatents, onChange }: SliderPanelProps) {
  return (
    <section className="panel slider-panel">
      <div className="panel-header">
        <h3>Process Controls</h3>
        <p>Slider count is inferred from the selected checkpoint.</p>
      </div>
      <div className="slider-grid">
        {processLatents.map((value, index) => (
          <label className="slider-card" key={`slider-${index}`}>
            <span>r{index + 1}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={value}
              onChange={(event) => {
                const next = [...processLatents];
                next[index] = Number(event.target.value);
                onChange(next);
              }}
            />
            <strong>{value.toFixed(2)}</strong>
          </label>
        ))}
      </div>
    </section>
  );
}
