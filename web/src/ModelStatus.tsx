import { useEffect, useRef, useState } from 'react';

type ModelEntry = {
  alias: string; roles: string[]; state: string; running?: number | null; waiting?: number | null;
  kvCache?: number | null; queueP95Seconds?: number | null; latencyP95Seconds?: number | null; preemptions?: number | null;
};
type Status = {
  reachable: boolean; detail?: string | null; overall: string; updatedAt?: string | null; models: ModelEntry[];
  runs: { active: number; queued: number; capacity: number };
};

// States as thesis-llm-setup classifies them (llm_setup/status.py): load and queue decide busy vs saturated.
const STATE_LABEL: Record<string, string> = {
  healthy: 'Ready', busy: 'Busy', saturated: 'Queueing', unavailable: 'Down', recovering: 'Recovering', unknown: 'Unknown',
};
const shortName = (alias: string) => alias.replace(/-model$/, '').replace(/^qwen-/, '');
const percent = (value?: number | null) => value == null ? '–' : `${Math.round(value * 100)}%`;
const seconds = (value?: number | null) => value == null ? '–' : `${value.toFixed(1)} s`;

/** Model load, queue and availability in the header, polled every 5 s from the harness API (`GET /models/status`). */
export function ModelStatus({ api }: { api: string }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      if (!document.hidden) {
        try {
          const response = await fetch(`${api}/models/status`);
          if (!response.ok) throw new Error(String(response.status));
          const body = await response.json() as Status;
          if (!stopped) { setStatus(body); setFailed(false); }
        } catch {
          if (!stopped) setFailed(true);
        }
      }
      if (!stopped) timer = setTimeout(poll, 5000);
    };
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [api]);

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent ? event.key === 'Escape' : !box.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('keydown', close);
    window.addEventListener('mousedown', close);
    return () => { window.removeEventListener('keydown', close); window.removeEventListener('mousedown', close); };
  }, [open]);

  const models = status?.models ?? [];
  const reachable = !!status?.reachable && !failed;
  const overall = reachable ? status?.overall ?? 'unknown' : 'unknown';
  const queued = status?.runs.queued ?? 0;
  const summary = failed ? 'Harness API unreachable' : !status ? 'Checking models…' : !reachable ? 'Model status unknown'
    : `Models ${STATE_LABEL[overall]?.toLowerCase() ?? overall}${queued ? ` · ${queued} run${queued === 1 ? '' : 's'} queued` : ''}`;

  return <div className="model-status" ref={box}>
    <button type="button" className="model-status-button" aria-expanded={open} aria-controls="model-status-panel" onClick={() => setOpen(value => !value)}>
      <span className="model-dots" aria-hidden="true">{(models.length ? models : [{ alias: '?', state: 'unknown' } as ModelEntry]).map(model =>
        <span key={model.alias} className={`model-dot state-${reachable ? model.state : 'unknown'}`} />)}</span>
      <span className="model-status-text">{summary}</span>
    </button>
    {open && <div className="model-status-panel" id="model-status-panel" role="dialog" aria-label="Model status">
      {!reachable && <p className="muted micro">{failed ? 'The harness API did not answer.' : status?.detail ?? 'No status yet.'} Load and queue come from the thesis-llm-setup status API (LLM_STATUS_URL).</p>}
      <table className="model-table">
        <thead><tr><th>Model</th><th>State</th><th>Running</th><th>Waiting</th><th>KV cache</th><th>Queue p95</th></tr></thead>
        <tbody>{models.map(model => <tr key={model.alias}>
          <td><strong>{shortName(model.alias)}</strong><div className="muted micro">{model.roles.join(', ') || 'not used by the harness'}</div></td>
          <td><span className={`model-dot state-${reachable ? model.state : 'unknown'}`} aria-hidden="true" /> {STATE_LABEL[reachable ? model.state : 'unknown'] ?? model.state}</td>
          <td>{model.running ?? '–'}</td><td>{model.waiting ?? '–'}</td><td>{percent(model.kvCache)}</td><td>{seconds(model.queueP95Seconds)}</td>
        </tr>)}</tbody>
      </table>
      {status && <p className="micro">Harness runs: {status.runs.active} researching, {status.runs.queued} waiting for a slot, {status.runs.capacity} slots.</p>}
      {status?.updatedAt && reachable && <p className="muted micro">Updated {new Date(status.updatedAt).toLocaleTimeString()}. Busy: 2+ requests waiting or KV cache 80%+. Queueing: 5+ waiting or KV cache 92%+.</p>}
    </div>}
  </div>;
}
