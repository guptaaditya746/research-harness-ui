// The screens other than Ask: Corpus, Settings, Runs, Gold, Benchmark; and the Ask screen's running and
// readings states. State lives in App; each screen gets what it shows and the actions it offers.
import type { Run } from './generated/run';
import { pathName, when } from './format';
import { Progress } from './Result';
import { type Config, SettingsPanel } from './SettingsPanel';
import type { BenchmarkQuestion, Budget, GoldCandidate, PaperLabel, RunPath } from './types';

const intro = { maxWidth: '56ch', marginBottom: 40 } as const;

export function CorpusScreen({ papers }: { papers: PaperLabel[] }) {
  return <>
    <h2>Corpus</h2>
    <p className="muted" style={intro}>Papers are parsed once, chunked into passages and indexed. Every path reads the same index, so runs stay comparable. New PDFs are added with <code>make ingest</code>.</p>
    <table className="table" style={{ marginTop: 34 }}>
      <thead><tr><th>Paper</th><th>Identifier</th></tr></thead>
      <tbody>{papers.map(paper => <tr key={paper.paper_id}>
        <td>{paper.title}</td>
        <td className="muted micro">{paper.dblp_iri ? 'DBLP' : 'local'} · {paper.paper_id}</td>
      </tr>)}</tbody>
    </table>
    <p className="muted micro" style={{ marginTop: 16 }}>{papers.length} papers indexed</p>
  </>;
}

function BudgetChoice({ title, note, value, options, onChange }: {
  title: string; note: string; value: Budget; options: [Budget, string][]; onChange: (value: Budget) => void;
}) {
  return <div>
    <div className="section-title">{title}</div>
    <p className="muted" style={{ fontSize: 13.5 }}>{note}</p>
    <div className="seg" role="radiogroup" aria-label={title}>
      {options.map(([option, label]) => <label key={option} className="seg-opt">
        <input type="radio" checked={value === option} onChange={() => onChange(option)} />{label}
      </label>)}
    </div>
  </div>;
}

export function SettingsScreen({ baseMode, agMode, save, onConfig }: {
  baseMode: Budget; agMode: Budget; save: (change: { baseline?: Budget; agentic?: Budget }) => void;
  onConfig?: (config: Config) => void;
}) {
  return <>
    <h2>Settings</h2>
    <p className="muted" style={intro}>Execution budgets live here so the ask screen stays one question and two runs. A change applies to every run that follows and is recorded with it.</p>
    <div style={{ display: 'grid', gap: 34 }}>
      <BudgetChoice title="Fast baseline budget" note="Slow adds an evidence-verification pass over the extracted triples."
        value={baseMode} options={[['fast', 'Fast'], ['slow', 'Slow · evidence review']]} onChange={value => save({ baseline: value })} />
      <BudgetChoice title="Agentic research budget" note="Fast caps planning depth and skips re-checks; the agent still chooses its own actions."
        value={agMode} options={[['fast', 'Fast'], ['slow', 'Slow · deeper research']]} onChange={value => save({ agentic: value })} />
    </div>
    <h3 style={{ marginTop: 56 }}>Harness settings</h3>
    <p className="muted" style={intro}>What runs read (the corpus, or the question's evidence document D), the models, time limits, how much evidence each path reads, extraction and the agent. Each run records the values it used.</p>
    <SettingsPanel onSaved={onConfig} />
  </>;
}

export function RunsScreen({ runs, openArtifact, setOpenArtifact, artifactBody, openRun }: {
  runs: Run[]; openArtifact: string; setOpenArtifact: (value: string) => void; artifactBody: string; openRun: (run: Run) => void;
}) {
  return <>
    <h2>Runs</h2>
    <p className="muted" style={intro}>Every run keeps its artifacts: the concept plans it considered, the reading it settled on, the subschema and query it ran, the triples and the passages behind them.</p>
    {runs.map(run => <details className="run-row" data-run-id={run.id} key={run.id}>
      <summary>
        <span className="muted micro">{when(run.startedAt)}</span>
        <span style={{ flex: '1 1 240px', fontSize: 15.5 }}>{run.question}</span>
        <span className="muted micro">{pathName(run.path)} · {run.budget}{run.runInput === "evidence_doc" ? " · input: evidence document" : ""} · {run.durationS} s</span>
      </summary>
      <div style={{ display: 'grid', gap: 10, marginTop: 14 }}>
        <div>{run.status}{run.reason ? ` — ${run.reason}` : ''}</div>
        <div><span className="muted">Reading&nbsp;&nbsp;</span>{run.concept?.label || run.readings[run.chosenReading ?? 0]?.label || 'No reading chosen'}</div>
        <div style={{ display: 'flex', gap: '10px 22px', flexWrap: 'wrap' }}>
          {run.artifacts.map(name => {
            const key = `${run.id}/${name}`;
            return <button key={name} className="text-button micro" style={{ color: openArtifact === key ? 'var(--color-text)' : undefined }}
              onClick={() => setOpenArtifact(openArtifact === key ? '' : key)}>{name}</button>;
          })}
        </div>
        {openArtifact.startsWith(`${run.id}/`) && <pre className="code">{artifactBody}</pre>}
        <div>
          <button className="text-button" style={{ fontSize: 13 }} onClick={() => openRun(run)}>Open run</button>
          <span className="muted micro" style={{ marginLeft: 24 }}>{run.rating ? `Rated ${run.rating}` : 'Not rated'} · {run.id}</span>
        </div>
      </div>
    </details>)}
  </>;
}

export function GoldScreen({ candidates, drafts, setDraft, errors, save }: {
  candidates: GoldCandidate[]; drafts: Record<string, string>; setDraft: (key: string, value: string) => void;
  errors: Record<string, string>; save: (candidate: GoldCandidate) => void;
}) {
  return <>
    <h2>Gold candidates</h2>
    <p className="muted">Review claim subgraphs and their provenance before approving a dataset row. Triples remain diagnostic. JSON is checked when you save.</p>
    {candidates.length === 0 && <p className="muted">No candidates yet. Rate a run, then promote it from the result sidebar.</p>}
    {candidates.map(candidate => {
      const claimsKey = `${candidate.id}:claims`;
      const triplesKey = `${candidate.id}:triples`;
      return <details className="run-row" key={candidate.id}>
        <summary><span>{candidate.question}</span><span className="muted micro">{candidate.status}</span></summary>
        <div className="muted micro">Gold claims</div>
        <textarea className="input code" aria-label={`Claims for ${candidate.question}`}
          value={drafts[claimsKey] ?? JSON.stringify(candidate.gold_claims || [], null, 2)}
          onChange={event => setDraft(claimsKey, event.target.value)} />
        <div className="muted micro">Diagnostic triples</div>
        <textarea className="input code" aria-label={`Triples for ${candidate.question}`}
          value={drafts[triplesKey] ?? JSON.stringify(candidate.gold_triples || [], null, 2)}
          onChange={event => setDraft(triplesKey, event.target.value)} />
        <button className="btn btn-secondary" onClick={() => save(candidate)}>Save and approve</button>
        {errors[candidate.id] && <p className="muted micro" role="status">{errors[candidate.id]}</p>}
      </details>;
    })}
  </>;
}

const fixed = (value: number | null | undefined, digits: number) => value?.toFixed(digits) ?? '—';

export function BenchmarkScreen({ rows, externalOn, refresh, run, runInput = 'corpus' }: {
  rows: BenchmarkQuestion[]; externalOn: boolean; refresh: () => void; run: (row: BenchmarkQuestion) => void; runInput?: string;
}) {
  const columns = externalOn ? ['baseline', 'adaptive-baseline', 'agentic', 'external'] : ['baseline', 'adaptive-baseline', 'agentic'];
  return <>
    <h2>Benchmark</h2>
    <p className="muted">Triple precision, recall and F1 against reviewed gold, averaged over attempts. In evaluation mode an Agentic attempt researches every proposed reading (one run each) and its runs are scored together as one prediction against all accepted readings.</p>
    <p className="muted micro">Runs that read {runInput === 'evidence_doc' ? 'the evidence document D' : 'the corpus'} (the run input in Settings; the two are scored separately).</p>
    <button className="btn btn-secondary" onClick={refresh}>Refresh</button>
    {rows.length === 0 && <p className="muted" style={{ marginTop: 20 }}>No approved gold questions yet. Approve a candidate on the Gold page first.</p>}
    <table className="table">
      <thead><tr><th>Question</th><th>Theme</th><th>Difficulty</th><th>Fast</th><th>Adaptive</th><th>Agentic</th>{externalOn && <th>External</th>}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.id}>
        <td><details>
          <summary>{row.question}</summary>
          <div className="muted micro">Gold {row.gold_version}</div>
          <button className="text-button micro" onClick={() => run(row)}>Run benchmark · 3× per path</button>
          {row.paths.map(path => <div key={path.path} className="micro">{path.path}: {path.completed}/{path.completed + path.failed} attempts · triple F1 {path.f1.mean?.toFixed(2) || '—'} · model calls {path.model_calls ?? '—'} · tool calls {path.tool_calls ?? '—'} · wall {fixed(path.wall_clock_s, 1)} s · {path.repetitions.join(', ')}</div>)}
        </details></td>
        <td>{row.theme ?? row.category}</td>
        <td>{row.difficulty ?? '—'}</td>
        {columns.map(column => {
          const path = row.paths.find(item => item.path === column || (column === 'adaptive-baseline' && item.path === 'baseline-enhanced'));
          return <td key={column}>{path ? path.completed ? <>
            <div>P {fixed(path.precision.mean, 2)} · R {fixed(path.recall.mean, 2)} · F1 {fixed(path.f1.mean, 2)}</div>
            <div className="muted micro">{path.completed}/{path.completed + path.failed} runs · {fixed(path.wall_clock_s, 1)} s</div>
          </> : `${path.failed} failed` : '—'}</td>;
        })}
      </tr>)}</tbody>
    </table>
  </>;
}

const PATH_TITLE: Record<RunPath, [string, string]> = {
  agentic: ['Agentic research', 'Concept · shared evidence graph · MCP verification'],
  'baseline-enhanced': ['Adaptive baseline', 'Iterative retrieval · batched extraction · evidence review'],
  external: ['External harness', 'mini-SWE-agent · bash commands over the corpus · none of our pipeline'],
  baseline: ['Fast baseline', 'Fixed pipeline · no verification pass'],
};

export function RunningScreen({ path, trace, waitingOn, waitedS, phase, onStop }: {
  path: RunPath; trace: Run['trace']; waitingOn: string | null; waitedS: number; phase: string; onStop: () => void;
}) {
  const [title, note] = PATH_TITLE[path] ?? PATH_TITLE.baseline;
  const lastRow = trace[trace.length - 1];
  return <section style={{ marginTop: 60 }}>
    <div className="section-top" style={{ marginBottom: 18 }}><span className="section-title">{title}</span><span className="muted micro">{note}</span></div>
    <Progress rows={trace} waitingOn={waitingOn} waitedS={waitedS} phase={phase} />
    {trace.map((step, i) => <div className={`log-row${step.action.startsWith('Model call') ? ' log-model' : ''}${step.action === 'Model call failed' ? ' log-failed' : ''}`} key={i}>
      <span className="log-time muted">{step.atS}s</span><span>{step.action}</span><span className="muted micro">{step.returned}</span>
    </div>)}
    {waitingOn && <div className="log-row log-waiting" role="status">
      <span className="log-time muted">…</span><span>Waiting {waitedS}s</span><span className="muted micro">{waitingOn}</span>
    </div>}
    {!waitingOn && lastRow && waitedS >= 10 && <div className="log-row log-waiting" role="status">
      <span className="log-time muted">…</span><span>Working {waitedS}s</span><span className="muted micro">since “{lastRow.action}”</span>
    </div>}
    <button className="text-button" style={{ marginTop: 24, fontSize: 13 }} onClick={onStop}>Stop run</button>
  </section>;
}

export function ReadingsScreen({ run, picking, choose, onNone }: {
  run: Run; picking: boolean; choose: (index: number) => void; onNone: () => void;
}) {
  return <section style={{ marginTop: 60 }}>
    <p style={{ maxWidth: '56ch' }}>The agent proposed {run.readings.length} interpretations of this question. Choose one to research.</p>
    <p className="muted" style={{ fontSize: 13.5, marginBottom: 34 }}>Each interpretation scopes retrieval and extraction differently; the question itself stays unchanged.</p>
    {run.readings.map((r, i) => {
      const concept = run.concepts[i];
      const hits = concept?.evidenceHits != null ? `${concept.evidenceHits} supporting passages` : '';
      return <button key={i} className="reading-row" disabled={picking} onClick={() => choose(i)}>
        <span className="muted micro">{i + 1}</span>
        <span style={{ display: 'grid', gap: 4 }}>
          <strong style={{ fontSize: 16.5 }}>{r.label}</strong>
          <span className="muted" style={{ fontSize: 13.5 }}>{r.gloss}</span>
          <span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{[r.classes.join(' · '), r.properties.join(' · '), hits].filter(Boolean).join('  |  ')}</span>
        </span>
      </button>;
    })}
    <button className="text-button" style={{ marginTop: 22, fontSize: 13 }} onClick={onNone}>None of these — rephrase the question</button>
  </section>;
}
