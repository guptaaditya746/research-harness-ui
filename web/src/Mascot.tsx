import { useId } from 'react';

export type MascotState = 'idle' | 'researching' | 'answered' | 'failed';

const LABEL: Record<MascotState, string> = {
  idle: 'Research Harness, ready',
  researching: 'Research Harness, researching',
  answered: 'Research Harness, answer backed by evidence',
  failed: 'Research Harness, run incomplete',
};

/**
 * The harness mark: three drops of process ink (cyan, magenta, yellow) that behave like one liquid body.
 * A blur-and-threshold filter joins the drops wherever they come close, so the shape flows as they move.
 * Idle: the drops drift together and the body breathes. Researching: they spread and circle, stretching
 * the body. Answered: they merge into one round drop and a spark appears. Failed: the body slumps and
 * greys. The motion is CSS in styles.css (`.mascot[data-state=...]`) and stops under reduced motion.
 */
export function Mascot({ state, size = 34 }: { state: MascotState; size?: number }) {
  const filter = `ink-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  return <svg className="mascot" data-state={state} width={size} height={size} viewBox="0 0 64 64" role="img" aria-label={LABEL[state]}>
    <defs>
      <filter id={filter} x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="3.4" result="blur" />
        <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -9" />
      </filter>
    </defs>
    <g className="mascot-body">
      <g className="mascot-ink" filter={`url(#${filter})`}>
        <g className="mascot-orbit">
          <g transform="rotate(0 32 32)"><circle className="mascot-drop drop-c" cx="32" cy="32" r="12.5" /></g>
          <g transform="rotate(120 32 32)"><circle className="mascot-drop drop-m" cx="32" cy="32" r="12.5" /></g>
          <g transform="rotate(240 32 32)"><circle className="mascot-drop drop-y" cx="32" cy="32" r="12.5" /></g>
        </g>
      </g>
      <g className="mascot-eyes">
        <ellipse className="eye-open" cx="26.5" cy="31" rx="2.3" ry="2.9" />
        <ellipse className="eye-open" cx="37.5" cy="31" rx="2.3" ry="2.9" />
        <path className="eye-happy" d="M23.8 32 q2.7 -4.2 5.4 0 M34.8 32 q2.7 -4.2 5.4 0" />
        <path className="eye-flat" d="M24 32 h5 M35 32 h5" />
      </g>
    </g>
    <path className="mascot-spark" d="M50 9 l1.6 4.4 l4.4 1.6 l-4.4 1.6 l-1.6 4.4 l-1.6 -4.4 l-4.4 -1.6 l4.4 -1.6 z" />
  </svg>;
}
