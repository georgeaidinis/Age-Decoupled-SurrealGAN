type SliderPanelProps = {
  ageYears: number;
  processLatents: number[];
  disabled?: boolean;
  onAgeChange: (value: number) => void;
  onProcessChange: (values: number[]) => void;
  onGenerate: () => void;
};

export function SliderPanel({
  ageYears,
  processLatents,
  disabled = false,
  onAgeChange,
  onProcessChange,
  onGenerate,
}: SliderPanelProps) {
  return (
    <section className="panel slider-panel">
      <div className="panel-header">
        <h3>Latent Controls</h3>
        <p>Adjust age and process sliders, then click Generate to show the anchored edit relative to the selected subject's inferred current state.</p>
      </div>

      <div className="slider-grid">
        <label className="slider-card slider-card-wide">
          <span>Age</span>
          <input
            type="range"
            min={20}
            max={100}
            step={1}
            value={ageYears}
            disabled={disabled}
            onChange={(event) => onAgeChange(Number(event.target.value))}
          />
          <strong>{ageYears.toFixed(0)} years</strong>
        </label>

        {processLatents.map((value, index) => (
          <label className="slider-card" key={`slider-${index}`}>
            <span>r{index + 1}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={value}
              disabled={disabled}
              onChange={(event) => {
                const next = [...processLatents];
                next[index] = Number(event.target.value);
                onProcessChange(next);
              }}
            />
            <strong>{value.toFixed(2)}</strong>
          </label>
        ))}
      </div>

      <div className="generate-row">
        <button className="generate-button" disabled={disabled} onClick={onGenerate}>
          Generate
        </button>
      </div>
    </section>
  );
}
