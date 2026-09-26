import type { Run } from './generated/run';

type Lane = 'Planning' | 'Agent' | 'Schema' | 'Extraction' | 'Review' | 'Answer' | 'Other';
const LANES: Lane[] = ['Planning', 'Agent', 'Schema', 'Extraction', 'Review', 'Answer', 'Other'];
// Pipeline steps shown as markers on their lane: the trace rows that say what a stage produced.
const MILESTONES: Record<string, Lane> = {
  'Understanding question': 'Planning', 'Routing the reading': 'Planning', 'Bounding extraction scope': 'Extraction',
  'Inferring schema': 'Schema', 'Extracting triples': 'Extraction', 'Adding DBLP backbone': 'Answer',
  'Querying graph and composing answer': 'Answer', 'Answer unsupported': 'Answer',
};

type Call = { lane: Lane; start: number; end: number; label: string; failed: boolean };
type Mark = { lane: Lane; at: number; label: string };

function laneOf(purpose: string): Lane {
  const text = purpose.toLowerCase();
  if (/agent|reader|supervisor|compaction|summary|tool loop/.test(text)) return 'Agent';
  if (/question understanding|concept|planning|reading|grounded/.test(text)) return 'Planning';
  if (/schema/.test(text)) return 'Schema';
  if (/evidence review/.test(text)) return 'Review';
  if (/answer|verification|verifier/.test(text)) return 'Answer';
  if (/extract|edc|definition|canonical|mapping|ontology|claim/.test(text)) return 'Extraction';
  return 'Other';
}

/** Model calls as bars, tool calls as ticks and stage results as markers, all from ``run.trace``. */
export function parseTimeline(run: Run) {
  const calls: Call[] = [];
  const ticks: number[] = [];
  const marks: Mark[] = [];
  for (const row of run.trace) {
    const parts = row.returned.split(' · ');
    if (row.action === 'Model call finished' || row.action === 'Model call failed') {
      // Finished: "alias · purpose · 12.3s · …"; failed: "alias · purpose · error · 4.1s" or
      // "timed out after 30s (alias · purpose)" (model_client.py).
      const timeout = /timed out after (\d+(?:\.\d+)?)s \((.+?) · (.+?)\)/.exec(row.returned);
      const durations = [...row.returned.matchAll(/(?:^|[\s(])(\d+(?:\.\d+)?)s(?=$|[\s·)])/g)];
      const seconds = timeout ? Number(timeout[1]) : durations.length ? Number(durations[durations.length - 1][1]) : 0;
      const alias = timeout ? timeout[2] : parts[0];
      const purpose = timeout ? timeout[3] : parts[1] ?? '';
      const failed = row.action === 'Model call failed';
      const detail = failed ? ` · ${timeout ? 'timed out' : parts.slice(2).filter(part => !/^\d+(\.\d+)?s$/.test(part.trim())).join(' · ')}` : '';
      calls.push({ lane: laneOf(purpose), start: Math.max(0, row.atS - seconds), end: row.atS,
        label: `${alias} · ${purpose} · ${seconds.toFixed(1)} s${detail}`, failed });
    } else if (row.purpose && / ms$/.test(row.returned) && / items · /.test(row.returned)) {
      ticks.push(row.atS);
    } else if (MILESTONES[row.action]) {
      marks.push({ lane: MILESTONES[row.action], at: row.atS, label: `${row.action}: ${row.returned}` });
    }
  }
  return { calls, ticks, marks };
}

function niceStep(total: number) {
  const raw = total / 6;
  const base = 10 ** Math.floor(Math.log10(Math.max(raw, 0.1)));
  return [1, 2, 5, 10].map(factor => factor * base).find(step => step >= raw) ?? raw;
}

export function RunTimeline({ run }: { run: Run }) {
  const { calls, ticks, marks } = parseTimeline(run);
  if (!calls.length && !ticks.length) return null;
  const total = Math.max(run.durationS || 0, ...calls.map(call => call.end), ...ticks, ...marks.map(mark => mark.at), 1);
  const used = LANES.filter(lane => calls.some(call => call.lane === lane) || marks.some(mark => mark.lane === lane)
    || (lane === 'Agent' && ticks.length));
  const width = 760, left = 86, right = 12, laneHeight = 24, top = 8;
  const height = top + used.length * laneHeight + 26;
  const x = (seconds: number) => left + (seconds / total) * (width - left - right);
  const step = niceStep(total);
  const axis = Array.from({ length: Math.floor(total / step) + 1 }, (_, index) => index * step);
  const modelSeconds = calls.reduce((sum, call) => sum + (call.end - call.start), 0);
  const byLane = used.map(lane => ({ lane, seconds: calls.filter(call => call.lane === lane).reduce((sum, call) => sum + call.end - call.start, 0) }));
  const longest = byLane.filter(item => item.seconds > 0).sort((a, b) => b.seconds - a.seconds)[0];
  return <section className="section" aria-label="Run timeline" style={{ marginTop: 34 }}>
    <div className="section-top"><span className="section-title">Run timeline</span>
      <span className="muted micro">{total.toFixed(0)} s · {calls.length} model calls ({modelSeconds.toFixed(0)} s of model time){ticks.length ? ` · ${ticks.length} tool calls` : ''}{longest ? ` · most time in ${longest.lane.toLowerCase()} (${longest.seconds.toFixed(0)} s)` : ''}</span></div>
    <div className="timeline-scroll">
      <svg className="timeline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Timeline of ${calls.length} model calls and ${ticks.length} tool calls over ${total.toFixed(0)} seconds`}>
        {axis.map(value => <g key={value}>
          <line x1={x(value)} x2={x(value)} y1={top} y2={height - 20} className="timeline-grid" />
          <text x={x(value)} y={height - 6} className="timeline-axis" textAnchor="middle">{value}s</text>
        </g>)}
        {used.map((lane, index) => {
          const y = top + index * laneHeight;
          return <g key={lane}>
            <text x={0} y={y + laneHeight / 2 + 4} className="timeline-lane">{lane}</text>
            {calls.filter(call => call.lane === lane).map((call, i) =>
              <rect key={i} x={x(call.start)} y={y + 5} width={Math.max(2, x(call.end) - x(call.start))} height={laneHeight - 10}
                rx={2} className={`timeline-bar lane-${lane.toLowerCase()}${call.failed ? ' failed' : ''}`}><title>{call.label}</title></rect>)}
            {lane === 'Agent' && ticks.map((at, i) => <line key={`t${i}`} x1={x(at)} x2={x(at)} y1={y + 2} y2={y + laneHeight - 2} className="timeline-tick"><title>tool call at {at.toFixed(1)} s</title></line>)}
            {marks.filter(mark => mark.lane === lane).map((mark, i) =>
              <path key={`m${i}`} d={`M${x(mark.at)} ${y + 4} l4 ${laneHeight / 2 - 4} l-4 ${laneHeight / 2 - 4} l-4 ${-(laneHeight / 2 - 4)} z`} className="timeline-mark"><title>{mark.label}</title></path>)}
          </g>;
        })}
      </svg>
    </div>
    <p className="muted micro">Bars are model calls (hover for model, purpose and duration; magenta = failed), ticks are the agent's tool calls, diamonds are stage results. Fast and Adaptive runs follow a fixed order of stages; Agentic runs start with the agent's own loop.</p>
  </section>;
}
