export type MascotState = 'idle' | 'researching' | 'answered' | 'failed';

const LABEL: Record<MascotState, string> = {
  idle: 'Research Harness, ready',
  researching: 'Research Harness, researching',
  answered: 'Research Harness, answer backed by evidence',
  failed: 'Research Harness, run incomplete',
};

// Three neighbours on a circle of radius 40 around the node at (60, 60). When the answer holds they
// dock onto the corners of the triangle below (styles.css moves each one there).
const SATELLITES = [
  { name: 's3', x: 60, y: 20 },
  { name: 's2', x: 95, y: 80 },
  { name: 's1', x: 25, y: 80 },
];
const EYE = { x: 60, y: 57, gap: 9, r: 3.6 };

/**
 * The harness mark, "Node": a graph node with a face and three neighbours (magenta, yellow, ink).
 * Idle: the node breathes and blinks. Researching: the neighbours orbit, the links flicker and the
 * eyes scan. Answered: the links give way to a triangle the neighbours dock into, a small connected
 * graph. Failed: the node greys and the neighbours drop away. The motion is CSS in styles.css
 * (`.mascot[data-state=...]`) and stops under reduced motion.
 */
export function Mascot({ state, size = 60 }: { state: MascotState; size?: number }) {
  const { x, y, gap, r } = EYE;
  return <svg className="mascot" data-state={state} width={size} height={size} viewBox="6 6 108 108" role="img" aria-label={LABEL[state]}>
    <g className="mascot-orbit">
      {SATELLITES.map(sat => <line key={sat.name} className="mascot-spoke" x1="60" y1="60" x2={sat.x} y2={sat.y} />)}
      {SATELLITES.map(sat => <circle key={sat.name} className={`mascot-sat ${sat.name}`} cx={sat.x} cy={sat.y} r="9" />)}
    </g>
    <polygon className="mascot-tri" points="60,18 101,88 19,88" />
    <circle className="mascot-node" cx="60" cy="60" r="28" />
    <g className="mascot-eyes">
      <ellipse className="eye-open" cx={x - gap} cy={y} rx={r} ry={r * 1.25} />
      <ellipse className="eye-open" cx={x + gap} cy={y} rx={r} ry={r * 1.25} />
      <path className="eye-happy" d={`M${x - gap - r * 1.3} ${y + 1} q${r * 1.3} ${-r * 2.2} ${r * 2.6} 0 M${x + gap - r * 1.3} ${y + 1} q${r * 1.3} ${-r * 2.2} ${r * 2.6} 0`} />
      <path className="eye-flat" d={`M${x - gap - r * 1.2} ${y} h${r * 2.4} M${x + gap - r * 1.2} ${y} h${r * 2.4}`} />
    </g>
  </svg>;
}
