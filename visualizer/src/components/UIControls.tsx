import type { ReactNode } from 'react';
import type { CSSProperties } from 'react';

type Tone = 'blue' | 'green' | 'purple' | 'amber' | 'red';

const TONE_COLORS: Record<Tone, string> = {
  blue: 'var(--blue)',
  green: 'var(--green)',
  purple: 'var(--purple)',
  amber: 'var(--amber)',
  red: 'var(--red)',
};

interface ControlSectionProps {
  title: string;
  description?: string;
  children: ReactNode;
}

export function ControlSection({ title, description, children }: ControlSectionProps) {
  return (
    <section className="control-section">
      <div className="section-heading">{title}</div>
      {description && <p className="control-description">{description}</p>}
      {children}
    </section>
  );
}

interface SliderFieldProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  minLabel?: string;
  maxLabel?: string;
  valueLabel?: string;
  tone?: Tone;
}

export function SliderField({
  label,
  min,
  max,
  step,
  value,
  onChange,
  minLabel,
  maxLabel,
  valueLabel,
  tone = 'blue',
}: SliderFieldProps) {
  const pct = ((value - min) / (max - min)) * 100;
  const style = {
    '--pct': `${Math.max(0, Math.min(100, pct))}%`,
    '--slider-color': TONE_COLORS[tone],
  } as CSSProperties;

  return (
    <label className="slider-field">
      <div className="slider-field-head">
        <span>{label}</span>
        <strong>{valueLabel ?? value.toLocaleString()}</strong>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={style}
        onChange={event => onChange(parseFloat(event.target.value))}
      />
      {(minLabel || maxLabel) && (
        <div className="slider-meta">
          <span>{minLabel ?? min}</span>
          <span>{maxLabel ?? max}</span>
        </div>
      )}
    </label>
  );
}

interface ModeButtonProps {
  active: boolean;
  onClick: () => void;
  title: string;
  subtitle?: string;
  tone?: Tone;
  className?: string;
}

export function ModeButton({
  active,
  onClick,
  title,
  subtitle,
  tone = 'blue',
  className = '',
}: ModeButtonProps) {
  return (
    <button
      type="button"
      className={`mode-button tone-${tone} ${active ? 'active' : ''} ${className}`.trim()}
      aria-pressed={active}
      onClick={onClick}
    >
      <strong>{title}</strong>
      {subtitle && <span>{subtitle}</span>}
    </button>
  );
}

interface StatusBannerProps {
  tone?: Tone;
  children: ReactNode;
}

export function StatusBanner({ tone = 'blue', children }: StatusBannerProps) {
  return <div className={`status-banner tone-${tone}`}>{children}</div>;
}

