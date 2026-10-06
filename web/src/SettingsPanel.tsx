// The harness settings the Settings page may change (GET/PUT /config, harness_config.ui): grouped controls,
// one save for all changes, a reset per changed setting. Changes apply to runs started afterwards.
import { useEffect, useMemo, useState } from 'react';
import { api, getJson } from './api';

export type ConfigRow = {
  name: string; group: string; label: string; help: string; kind: 'choice' | 'bool' | 'int' | 'float' | 'text';
  options: string[] | null; min: number | null; max: number | null; default: string; value: string;
};
export type Config = { settings: ConfigRow[]; overrides: Record<string, string>; file: string; active_runs: number };

const errorText = (error: unknown) => {
  const text = String(error instanceof Error ? error.message : error);
  const detail = text.match(/"detail":"([^"]+)"/);
  return detail ? detail[1] : text;
};

function Control({ row, value, onChange }: { row: ConfigRow; value: string; onChange: (value: string) => void }) {
  const id = `setting-${row.name}`;
  if (row.kind === 'choice') {
    return <select id={id} className="input" value={value} onChange={event => onChange(event.target.value)}>
      {(row.options || []).map(option => <option key={option} value={option}>{option}</option>)}
    </select>;
  }
  if (row.kind === 'bool') {
    return <input id={id} type="checkbox" checked={value === 'true'} onChange={event => onChange(event.target.checked ? 'true' : 'false')} />;
  }
  if (row.kind === 'int' || row.kind === 'float') {
    return <input id={id} className="input" type="number" style={{ width: '12ch' }} value={value}
      step={row.kind === 'int' ? 1 : 'any'} min={row.min ?? undefined} max={row.max ?? undefined}
      onChange={event => onChange(event.target.value)} />;
  }
  return <input id={id} className="input" type="text" style={{ width: '100%' }} value={value} placeholder={row.default || '(empty)'}
    onChange={event => onChange(event.target.value)} />;
}

export function SettingsPanel({ onSaved }: { onSaved?: (config: Config) => void }) {
  const [config, setConfig] = useState<Config | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const load = () => getJson<Config>(`${api}/config`).then(value => { setConfig(value); onSaved?.(value); })
    .catch(error => setMessage(`Settings could not be loaded: ${errorText(error)}`));
  useEffect(() => { load(); }, []);

  const groups = useMemo(() => {
    const out: [string, ConfigRow[]][] = [];
    for (const row of config?.settings || []) {
      const group = out.find(([name]) => name === row.group);
      if (group) group[1].push(row); else out.push([row.group, [row]]);
    }
    return out;
  }, [config]);

  const send = async (values: Record<string, string | null>) => {
    setMessage('');
    try {
      const saved = await getJson<Config>(`${api}/config`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) });
      setConfig(saved); setDrafts({}); onSaved?.(saved);
      setMessage('Saved. Runs started from now on use these settings and record them.');
    } catch (error) { setMessage(errorText(error)); }
  };

  if (!config) return <p className="muted">{message || 'Loading settings…'}</p>;
  const changed = Object.keys(drafts).length;
  return <section aria-label="Harness settings" style={{ display: 'grid', gap: 30 }}>
    {config.active_runs > 0 && <p className="notice">{config.active_runs} run(s) active: settings can be changed when they have finished.</p>}
    {groups.map(([group, rows]) => <fieldset key={group} style={{ border: 0, padding: 0, margin: 0, display: 'grid', gap: 14 }}>
      <legend className="section-title" style={{ marginBottom: 10 }}>{group}</legend>
      {rows.map(row => {
        const value = drafts[row.name] ?? row.value;
        const overridden = row.name in config.overrides;
        return <div key={row.name} className="setting-row">
          <label htmlFor={`setting-${row.name}`} style={{ fontSize: 14 }}>{row.label}
            <div className="muted micro"><code>{row.name}</code></div>
          </label>
          <div style={{ display: 'grid', gap: 4 }}>
            <Control row={row} value={value} onChange={next => setDrafts(previous => {
              const copy = { ...previous };
              if (next === row.value) delete copy[row.name]; else copy[row.name] = next;
              return copy;
            })} />
            <span className="muted micro">{row.help} Default: {row.default || '(empty)'}.
              {overridden && <> Changed here. <button className="text-button micro" disabled={config.active_runs > 0}
                onClick={() => send({ [row.name]: null })}>Reset</button></>}
            </span>
          </div>
        </div>;
      })}
    </fieldset>)}
    <div className="actions" style={{ alignItems: 'center' }}>
      <button className="btn btn-primary" disabled={!changed || config.active_runs > 0} onClick={() => send(drafts)}>
        {changed ? `Save ${changed} change${changed === 1 ? '' : 's'}` : 'No changes'}</button>
      {changed > 0 && <button className="text-button micro" onClick={() => setDrafts({})}>Discard</button>}
      <button className="text-button micro muted" onClick={load}>Reload</button>
    </div>
    {message && <p className="muted micro" role="status">{message}</p>}
    <p className="muted micro">Kept in <code>{config.file}</code> and applied over <code>.env</code> and the profile when the API starts. Settings not listed here are changed in <code>.env</code>; the evidence documents themselves are built in labelling (<code>labelling evidence-docs</code>, its <code>HARNESS_DOC_*</code> settings).</p>
  </section>;
}
