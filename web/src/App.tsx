import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import type { Check, ClaimSubgraph, Reading, Run, Triple } from './generated/run';

// The graph library is large; load it only when a graph is shown.
const AnswerGraphView = lazy(() => import('./AnswerGraph').then(module => ({ default: module.AnswerGraphView })));

type Scene = 'idle' | 'running' | 'readings' | 'agentic_result' | 'baseline_result' | 'failed' | 'runs' | 'corpus' | 'settings' | 'gold' | 'benchmark';
type TripleView = 'statements' | 'graph' | 'turtle';
type GoldCandidate = { id: string; run_id: string; question: string; gold_triples: Triple[]; gold_claims?: ClaimSubgraph[]; notes: string; status: string };
type PaperLabel = { paper_id: string; dblp_iri?: string | null; title: string };
type LibraryQuestion = { question: string; id?: string | null; category: string; type?: string | null; expected?: string | null; source: string };
type RunPath = 'agentic' | 'baseline' | 'baseline-enhanced';
type Aggregate = { mean: number | null; min: number | null; max: number | null };
type BenchmarkPath = { path: string; completed: number; failed: number; precision: Aggregate; recall: Aggregate; f1: Aggregate; claim_precision: Aggregate; claim_recall: Aggregate; claim_f1: Aggregate; repetitions: string[] };
type BenchmarkQuestion = { id: string; question: string; category: string; gold_version: string; paths: BenchmarkPath[] };
const screenRun: Record<string, string> = { readings: 'r-0413', agentic_result: 'r-0412', baseline_result: 'r-0411', failed: 'r-0410' };
const api = '/api';

function metric(pair?: [number, number] | null) {
  if (!pair) return '—';
  return `${pair[1] ? (pair[0] / pair[1]).toFixed(2) : '—'}  (${pair[0]} / ${pair[1]})`;
}

function turtle(reading: Reading) {
  const iri = (text: string) => `hrn:${text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')}`;
  return ['@prefix hrn: <https://harness.local/schema#> .', '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .', '',
    ...reading.triples.map(t => `${iri(t.s)} hrn:${t.p} ${/^\d+$/.test(t.o) ? `"${t.o}"^^xsd:integer` : iri(t.o)} .`)].join('\n');
}

function ontology(reading: Reading) {
  return ['@prefix hrn: <https://harness.local/schema#> .', '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
    '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .', '',
    ...reading.classes.map(c => `hrn:${c} a rdfs:Class .`), '',
    ...reading.axioms.map(([p, d, r]) => `hrn:${p} a rdf:Property ;\n  rdfs:domain hrn:${d} ;\n  rdfs:range ${r.includes(':') ? r : `hrn:${r}`} .`)].join('\n');
}

function Sidebar({ run, reading, onRate, onPromote }: { run: Run; reading?: Reading; onRate: (rating: string) => void; onPromote: () => void }) {
  const [plansOpen, setPlansOpen] = useState(false);
  return <aside className="sidebar" aria-label="Evaluation">
    <div><div className="section-title" style={{ marginBottom: 10 }}>Interpretation</div>
      <p style={{ fontSize: 13.5, margin: 0 }}>{run.status === 'corpus_absent' ? 'No match on the declared corpus evidence surface.' : run.status === 'no_evidence_found' ? 'No evidence found on the declared indexed full-text search surface.' : run.chosenReading != null ? `Selected: ${reading?.label || 'recorded reading'}` : 'Choose a reading before research continues.'}</p></div>
    <div><div className="section-title" style={{ marginBottom: 14 }}>Metrics</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        {[
          ['Concept disagreement', reading?.metrics.disagreement == null ? 'n/a — single pass' : reading.metrics.disagreement.toFixed(2), '1 − mean pairwise Jaccard over 3 independent concept plans'],
          ['Triple precision', metric(reading?.metrics.precision), 'returned triples that are correct against the gold set'],
          ['Triple recall', metric(reading?.metrics.recall), 'gold triples for this reading that were returned'],
        ].map(([label, value, note]) => <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span className="muted" style={{ fontSize: 13 }}>{label}</span><strong style={{ fontSize: 20, lineHeight: 1.2 }}>{value}</strong><span className="muted" style={{ fontSize: 12 }}>{note}</span>
        </div>)}
      </div><button className="text-button micro" style={{ marginTop: 14 }} onClick={() => setPlansOpen(!plansOpen)}>{plansOpen ? 'Hide plans' : 'Compare plans'}</button>
      {plansOpen && <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>{(run.plans.length ? run.plans : ['Single pass — no alternative plans generated.']).map((plan, i) => <div className="micro muted" key={i}>{plan}</div>)}</div>}
    </div>
    <div><div className="section-title" style={{ marginBottom: 10 }}>Your rating</div><div role="radiogroup" aria-label="Answer usefulness" style={{ display: 'grid', gap: 7 }}>
      {([['useful', 'Useful'], ['partial', 'Partially useful'], ['not_useful', 'Not useful']] as const).map(([value, label]) => <label className="radio" key={value}><input type="radio" name="rating" checked={run.rating === value} onChange={() => onRate(value)} /><span className="dot" />{label}</label>)}
    </div>{run.rating && <button className="btn btn-secondary" style={{ marginTop: 14 }} onClick={onPromote}>Promote to gold candidate</button>}</div>
  </aside>;
}

function Triples({ reading, labels }: { reading: Reading; labels: Record<string, string> }) {
  const [view, setView] = useState<TripleView>('graph');
  const label = (value: string) => labels[value] ? `${labels[value]} (${value})` : value;
  return <section className="section" aria-label="Triples"><div className="section-top"><span className="section-title">Answer graph</span><div className="section-links">
    {(['graph', 'statements', 'turtle'] as const).map(v => <button key={v} className="text-button micro" style={{ color: view === v ? 'var(--color-text)' : undefined }} onClick={() => setView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>)}
  </div></div>
    {view === 'statements' && reading.triples.map((t: Triple, i: number) => <details className="row-details" key={i}><summary><span style={{ fontSize: 15.5 }}>{label(t.s)}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{t.p}</span><span style={{ fontSize: 15.5 }}>{label(t.o)}</span><span className="muted" style={{ marginLeft: 'auto', fontSize: 12 }}>{t.support}</span></summary><p style={{ margin: '12px 0 0', fontStyle: 'italic', fontSize: 14 }}>{t.evidence}</p><div className="muted micro">{t.source}</div></details>)}
    {view === 'graph' && <Suspense fallback={<p className="muted micro">Loading graph…</p>}><AnswerGraphView reading={reading} labels={labels} /></Suspense>}
    {view === 'turtle' && <pre className="code">{turtle(reading)}</pre>}
  </section>;
}

function Claims({ reading, labels }: { reading: Reading; labels: Record<string, string> }) {
  const label = (value: string) => labels[value] ? `${labels[value]} (${value})` : value;
  return <section className="section" aria-label="Connected claims"><div className="section-top"><span className="section-title">Connected claims</span><span className="muted micro">{reading.claims.length} source-backed claims</span></div>
    {reading.claims.map(claim => <details className="row-details" key={claim.id}><summary><strong>{claim.kind}</strong><span>{label(claim.paperId)}</span><span className="muted micro" style={{ marginLeft: 'auto' }}>{claim.status}</span></summary><div style={{ display: 'grid', gap: 7, marginTop: 12 }}>{Object.entries(claim.fields).map(([name, field]) => <div key={name}><strong>{name}</strong>: {label(field.value)} <span className="muted micro">[{field.sourceIds.join(', ')}]</span></div>)}</div><p className="muted" style={{ marginTop: 10 }}>{Object.entries(claim.evidence).map(([source, excerpt]) => `${source}: ${excerpt}`).join(' · ')}</p>{claim.evidenceAnchors.length > 0 && <div className="muted micro">Evidence anchors: {claim.evidenceAnchors.map(anchor => `${anchor.kind}:${anchor.sourceId}`).join(' · ')}</div>}</details>)}
    {reading.claimRelations.map((relation, i) => <div className="muted micro" key={i}>{relation.sourceClaim} {relation.relation} {relation.targetClaim}: {relation.reason}</div>)}
  </section>;
}

const checkMark: Record<Check['status'], string> = { pass: '✓', warn: '!', fail: '✕', skipped: '–', info: 'i' };

function Checks({ checks }: { checks: Check[] }) {
  if (!checks.length) return null;
  return <section className="section" aria-label="Checks"><div className="section-top"><span className="section-title">Checks</span><span className="muted micro">each stage's outcome</span></div>
    <ul className="checks">{checks.map(check => <li key={check.name} className={`check check-${check.status}`}><span className="check-mark" aria-label={check.status}>{checkMark[check.status]}</span><span className="check-name">{check.name}</span><span className="muted micro">{check.detail}</span></li>)}</ul>
  </section>;
}

type TraceRows = Run['trace'];
const MODEL_ROWS = new Set(['Model call started', 'Model call finished', 'Model call failed', 'Tool call']);

/** Live progress, derived only from streamed trace rows: what is running now, not whether it will succeed. */
function Progress({ rows, waitingOn, waitedS, phase }: { rows: TraceRows; waitingOn: string | null; waitedS: number; phase: string }) {
  const last = rows[rows.length - 1];
  const stageRow = [...rows].reverse().find(row => !MODEL_ROWS.has(row.action));
  const [model, purpose] = (waitingOn || '').split(' · ');
  const tools = rows.filter(row => row.action === 'Tool call');
  const models = rows.filter(row => row.action === 'Model call finished').length;
  let passages = '';
  for (const row of rows) {
    const found = row.returned.match(/(\d+) passages? from (\d+) papers?/);
    if (found) passages = `${found[1]} passages from ${found[2]} papers`;
    const facts = row.action === 'Retrieving verified corpus facts' && row.returned.match(/(\d+) facts?/);
    if (facts) passages = `${facts[1]} verified facts`;
  }
  const elapsed = Math.round((last?.atS ?? 0) + waitedS);
  const items: [string, string][] = [
    ['Phase', phase],
    ['Stage', purpose || stageRow?.action || 'starting'],
    ['Model in flight', waitingOn ? `${model} · ${waitedS}s` : 'none'],
    ['Model calls done', String(models)],
    ['Corpus tools', tools.length ? `${tools.length} · last: ${tools[tools.length - 1].returned}` : 'none yet'],
    ['Examined', passages || '—'],
    ['Elapsed', `${elapsed}s`],
  ];
  return <dl className="progress" aria-label="Run progress">{items.map(([name, value]) => <div key={name}><dt className="muted micro">{name}</dt><dd>{value}</dd></div>)}</dl>;
}

/** The outcome in words, taken from the stage checks: which stage ended the run and why. */
function outcome(run: Run): string | null {
  const failed = run.checks.find(check => check.status === 'fail');
  if (!failed) return null;
  if (failed.name === 'retrieval') return `Corpus search found no usable passages: ${failed.detail}`;
  if (failed.name === 'extraction pass' || failed.name === 'extraction') return `Passages were found, but extraction produced no supported triples: ${failed.detail}`;
  if (failed.name.startsWith('agent')) return `Agent validation failed (${failed.name}): ${failed.detail}`;
  return `Stopped at ${failed.name}: ${failed.detail}`;
}

function Result({ run, pick, setPick, onDeeper, labels }: { run: Run; pick: number; setPick: (index: number) => void; onDeeper: () => void; labels: Record<string, string> }) {
  const [showOntology, setShowOntology] = useState(false);
  const [showQuery, setShowQuery] = useState(false);
  const reading = run.readings[pick] || run.readings[0];
  const failed = run.status === 'incomplete';
  const absent = run.status === 'corpus_absent';
  const noEvidence = run.status === 'no_evidence_found';
  const pathLabel = run.path === 'baseline-enhanced' ? 'adaptive baseline' : run.path === 'baseline' ? 'fast baseline' : 'agentic research';
  const reason = run.reason || '';
  const modelFailure = [...run.trace].reverse().find(row => row.action === 'Model call failed')?.returned;
  const failure = reason === 'provider_error'
    ? `A model call failed: ${modelFailure || 'the provider did not return a usable reply'}. The trace below the question shows every model call with its duration.`
    : reason === 'no_results'
      ? `The ${pathLabel} could not find enough supported evidence in the corpus to answer this question.`
      : reason === 'deadline_exceeded'
        ? `The ${pathLabel} exceeded its time limit and was stopped. The model service may be overloaded; try again or use the fast budget.`
        : reason.startsWith('concept_proposal_failed')
          ? `The agent could not propose interpretations of this question (${reason.replace('concept_proposal_failed: ', '')}). Check that the agent model is reachable, then try again.`
          : `The ${pathLabel} ended before it produced an answer (${reason || 'unknown failure'}).`;
  const stageOutcome = outcome(run);
  return <div style={{ marginTop: 64 }}>
    <p className="notice">{failed ? `Incomplete — ${pathLabel} stopped before answer composition.` : absent ? `No in-corpus match found. ${run.evidenceSurface || 'Corpus evidence'} was checked${run.searchCoverage?.inspected != null ? ` (${run.searchCoverage.inspected} records)` : ''}.` : noEvidence ? `No evidence found on ${run.evidenceSurface || 'the indexed full-text surface'} after the declared search procedure; this is not proof of corpus absence.` : run.executionStatus === 'degraded' ? 'Agentic execution degraded; showing a grounded fallback excluded from agentic scores.' : `Interpreted as: ${reading?.label}.${run.coverage != null && run.coverage < 1 ? ` Evaluated ${Math.round(run.coverage * 100)}% of obligations; ranking is provisional.` : ''}`}</p>
    {run.concept && <p className="muted micro" style={{ marginTop: 12 }}>Researched as: <strong>{run.concept.label}</strong> — {run.concept.gloss}{run.concept.autoSelected ? ' (selected automatically for evaluation)' : ''}</p>}
    {reading?.measure && !failed && !absent && !noEvidence && <p className="measure">Answered using: <strong>{reading.measure}</strong></p>}
    {stageOutcome && <p className="muted" style={{ marginTop: 12 }}>{stageOutcome}</p>}
    {failed && <button className="btn btn-secondary" style={{ marginTop: 24 }} onClick={onDeeper}>Try agentic research</button>}
    {!failed && run.path === 'agentic' && run.readings.length > 1 && <div style={{ marginTop: 20 }}><div className="muted micro" style={{ marginBottom: 8 }}>Re-answer under another reading</div><div style={{ display: 'flex', gap: '12px 24px', flexWrap: 'wrap' }}>{run.readings.map((r, i) => i !== pick && <button key={i} className="text-button" style={{ fontSize: 13 }} onClick={() => setPick(i)}>{r.label}</button>)}</div></div>}
    {(absent || noEvidence) && <section className="section" aria-label="Research result"><div className="section-top"><span className="section-title">Research result</span></div><p className="answer">{run.reason}</p><pre className="code">{JSON.stringify(run.searchCoverage, null, 2)}</pre></section>}
    {!failed && !absent && !noEvidence && reading && <><section className="section" aria-label="Subschema"><div className="section-top"><span className="section-title">Subschema</span><div className="section-links"><button className="text-button micro" onClick={() => setShowOntology(!showOntology)}>{showOntology ? 'Hide ontology' : 'Ontology (Turtle)'}</button><button className="text-button micro" onClick={() => setShowQuery(!showQuery)}>{showQuery ? 'Hide query' : 'SPARQL'}</button></div></div>
      <p className="muted" style={{ fontSize: 14, margin: '10px 0 16px' }}>{run.path === 'baseline' ? 'Unstated — inferred by the extractor. ' : ''}The answer is computed over {reading.classes.map(c => c.toLowerCase() + 's').join(', ')} only. Nothing outside these is read.</p>
      <div style={{ display: 'grid', gap: 12 }}>{reading.axioms.map(([property, domain, range]) => <div key={property}><div style={{ fontSize: 14.5 }}>A {domain.toLowerCase()} has {property.replaceAll('_', ' ')} relating it to {range.replace('xsd:', '')}.</div><div className="muted" style={{ font: '11.5px ui-monospace, Menlo, monospace' }}>{`hrn:${property}  rdfs:domain hrn:${domain}  rdfs:range ${range.includes(':') ? range : `hrn:${range}`}`}</div></div>)}</div>
      {showOntology && <pre className="code">{ontology(reading)}</pre>}{showQuery && <pre className="code">{reading.sparql}</pre>}
    </section><Claims reading={reading} labels={labels} /><Triples key={pick} reading={reading} labels={labels} /><section className="section" aria-label="Natural-language answer" style={{ marginTop: 34 }}><div className="section-top"><span className="section-title">Natural-language answer</span><span className="muted micro">Downstream of accepted claims</span></div><p className="answer">{reading.answer}</p></section></>}
    {failed && <section className="section" aria-label="Run result" style={{ marginTop: 34 }}><div className="section-top"><span className="section-title">Natural-language answer</span></div><p className="answer">{failure}</p></section>}
    <Checks checks={run.checks} />
  </div>;
}

const categoryOrder = ['metadata', 'citation', 'aggregation', 'synthesis', 'ambiguity', 'gold', 'recent'];
const categoryLabel: Record<string, string> = { metadata: 'Metadata', citation: 'Citations', aggregation: 'Aggregation', synthesis: 'Synthesis over full text', ambiguity: 'Ambiguous — agent interprets', gold: 'Gold candidates', recent: 'Recent runs' };

function QuestionLibrary({ items, filter, setFilter, onPick, onRun, onCompare }: { items: LibraryQuestion[]; filter: string; setFilter: (value: string) => void; onPick: (text: string) => void; onRun: (kind: RunPath, text: string) => void; onCompare: (text: string) => void }) {
  const needle = filter.trim().toLowerCase();
  const visible = items.filter(item => !needle || item.question.toLowerCase().includes(needle) || (item.type || '').toLowerCase().includes(needle));
  const groups = [...new Set(visible.map(item => item.category))].sort((a, b) => (categoryOrder.indexOf(a) + 1 || 99) - (categoryOrder.indexOf(b) + 1 || 99));
  return <section id="question-library" className="library" aria-label="Question library">
    <input className="input library-filter" autoFocus value={filter} onChange={event => setFilter(event.target.value)} placeholder="Filter questions" aria-label="Filter questions" />
    {visible.length === 0 && <p className="muted micro">No questions match.</p>}
    {groups.map(group => <div key={group} className="library-group"><div className="section-title">{categoryLabel[group] || group}</div>
      {visible.filter(item => item.category === group).map(item => <div key={`${item.source}-${item.id ?? item.question}`} className="library-row">
        <button className="text-button library-text" title="Put this question in the box to edit it" onClick={() => onPick(item.question)}>{item.question}</button>
        <span className="library-meta muted micro">{item.type?.replaceAll('_', ' ')}</span>
        <span className="library-run" aria-label={`Run: ${item.question}`}>
          <button className="text-button micro" onClick={() => onRun('baseline', item.question)}>Fast</button>
          <button className="text-button micro" onClick={() => onRun('baseline-enhanced', item.question)}>Adaptive</button>
          <button className="text-button micro" onClick={() => onRun('agentic', item.question)}>Agentic</button>
          <button className="text-button micro" onClick={() => onCompare(item.question)}>Compare</button>
        </span>
      </div>)}</div>)}
  </section>;
}

export default function App() {
  const initial = new URLSearchParams(location.search).get('screen') as Scene | null;
  const [scene, setScene] = useState<Scene>(initial || 'idle');
  const [runs, setRuns] = useState<Run[]>([]);
  const [question, setQuestion] = useState('');
  const [activeId, setActiveId] = useState(screenRun[initial || ''] || '');
  const [pick, setPick] = useState(0);
  const [path, setPath] = useState<'agentic' | 'baseline' | 'baseline-enhanced'>('agentic');
  const [baseMode, setBaseMode] = useState<'fast' | 'slow'>('fast');
  const [agMode, setAgMode] = useState<'fast' | 'slow'>('slow');
  const [openArtifact, setOpenArtifact] = useState('');
  const [artifactBody, setArtifactBody] = useState('');
  const [liveTrace, setLiveTrace] = useState<Run['trace']>([]);
  const [candidates, setCandidates] = useState<GoldCandidate[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [promotion, setPromotion] = useState('');
  const [benchmarkRows, setBenchmarkRows] = useState<BenchmarkQuestion[]>([]);
  const [papers, setPapers] = useState<PaperLabel[]>([]);
  const [notice, setNotice] = useState('');
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [goldErrors, setGoldErrors] = useState<Record<string, string>>({});
  const [picking, setPicking] = useState(false);
  const [lastRowAt, setLastRowAt] = useState(0);
  const [phase, setPhase] = useState('');
  const [, setTick] = useState(0);
  const [library, setLibrary] = useState<LibraryQuestion[]>([]);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryFilter, setLibraryFilter] = useState('');
  // Exactly one set of live streams belongs to the screen; older ones are closed.
  const streams = useRef<EventSource[]>([]);
  const watched = useRef<string>('');
  const compareIds = useRef<string[]>([]);
  const closeStreams = () => { streams.current.forEach(stream => stream.close()); streams.current = []; };
  useEffect(() => closeStreams, []);
  const active = useMemo(() => runs.find(r => r.id === activeId), [runs, activeId]);
  const reading = active?.readings[pick] || active?.readings[0];

  const getJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
    const response = await fetch(url, init);
    if (!response.ok) throw new Error(`${response.status} ${await response.text().catch(() => '')}`.trim());
    return response.json() as Promise<T>;
  };
  const loadLibrary = () => getJson<{ questions: LibraryQuestion[] }>(`${api}/questions`).then(data => setLibrary(data.questions || [])).catch(() => undefined);
  useEffect(() => { loadLibrary(); }, []);
  useEffect(() => { if (!libraryOpen) return; const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setLibraryOpen(false); }; window.addEventListener('keydown', close); return () => window.removeEventListener('keydown', close); }, [libraryOpen]);
  useEffect(() => { getJson<{ baseline: 'fast' | 'slow'; agentic: 'fast' | 'slow' }>(`${api}/settings`).then(value => { setBaseMode(value.baseline); setAgMode(value.agentic); }).catch(() => undefined); }, []);
  const saveSettings = (change: { baseline?: 'fast' | 'slow'; agentic?: 'fast' | 'slow' }) => {
    if (change.baseline) setBaseMode(change.baseline);
    if (change.agentic) setAgMode(change.agentic);
    fetch(`${api}/settings`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change) }).catch(() => undefined);
  };
  useEffect(() => { getJson<PaperLabel[]>(`${api}/corpus/papers`).then(items => { setPapers(items); setLabels(Object.fromEntries(items.flatMap(item => [[item.paper_id, item.title], ...(item.dblp_iri ? [[item.dblp_iri, item.title]] : [])]))); }).catch(() => setNotice('Corpus is not available: check CORPUS_ROOT on the API.')); getJson<Run[]>(`${api}/runs`).then((items: Run[]) => {
    setRuns(items);
    const id = screenRun[initial || ''];
    if (id) setQuestion(items.find(run => run.id === id)?.question || '');
    if (initial === 'running') setQuestion(items.find(run => run.id === 'r-0412')?.question || '');
  }).catch(() => setNotice('The API is not reachable. Start it and reload the page.')); }, []);
  useEffect(() => { if (!openArtifact) return; const [id, name] = openArtifact.split('/');
    setArtifactBody('Loading…');
    getJson<{ content: string }>(`${api}/runs/${id}/artifacts/${name}`).then(data => setArtifactBody(data.content)).catch(error => setArtifactBody(`Could not load ${name}: ${error.message}`));
  }, [openArtifact]);
  useEffect(() => { if (scene !== 'gold') return; getJson<GoldCandidate[]>(`${api}/datasets/candidates`).then(rows => { setCandidates(rows); setDrafts({}); setGoldErrors({}); }).catch(error => setNotice(`Gold candidates could not be loaded: ${error.message}`)); }, [scene]);
  const loadBenchmark = () => getJson<{ questions: BenchmarkQuestion[] }>(`${api}/benchmark`).then(data => setBenchmarkRows(data.questions || [])).catch(error => setNotice(`Benchmark could not be loaded: ${error.message}`));
  useEffect(() => { if (scene === 'benchmark') loadBenchmark(); }, [scene]);
  const runBenchmark = async (row: BenchmarkQuestion) => {
    try {
      const started = await Promise.all((['baseline', 'baseline-enhanced', 'agentic'] as const).map(kind => getJson<{ ids: string[] }>(`${api}/benchmark/execute`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: row.question, path: kind, budget: kind === 'agentic' ? agMode : baseMode, repetitions: 3 }) })));
      setNotice(`Started ${started.reduce((total, item) => total + item.ids.length, 0)} evaluation runs for "${row.question}". Runs queue for model capacity; press Refresh to update scores.`);
    } catch (error) { setNotice(`Benchmark runs could not be started: ${(error as Error).message}`); }
  };

  const openRun = (run: Run) => { setActiveId(run.id); setQuestion(run.question); setPick(run.chosenReading ?? 0); setPromotion(''); setNotice(''); setScene(run.status === 'awaiting_reading' ? 'readings' : run.status === 'incomplete' ? 'failed' : run.status === 'stopped' ? 'idle' : run.path === 'agentic' ? 'agentic_result' : 'baseline_result'); };
  const remember = (run: Run) => { setRuns(items => [run, ...items.filter(item => item.id !== run.id)]); loadLibrary(); };
  // Poll until the run is persisted: used when the event stream drops, so a
  // network blip never turns a still-running run into a "failed" screen.
  const settle = async (id: string): Promise<Run | null> => {
    // The API persists a terminal state for every run (restarts included), so
    // a run that stays unknown for a minute is reported rather than awaited.
    for (let attempt = 0; attempt < 40; attempt += 1) {
      const response = await fetch(`${api}/runs/${id}`).catch(() => null);
      if (response?.ok) return response.json() as Promise<Run>;
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
    return null;
  };
  const watch = (id: string, onDone: (run: Run) => void) => {
    const events = new EventSource(`${api}/runs/${id}/events`);
    streams.current.push(events);
    let finished = false;
    const finish = async () => {
      if (finished) return; finished = true; events.close();
      const run = await settle(id);
      if (run) onDone(run);
      else if (watched.current === id || watched.current === '*') { setNotice(`Lost contact with run ${id}. Check that the API is running, then open it from Runs.`); setScene('failed'); }
    };
    events.addEventListener('trace', event => { if (watched.current === id || watched.current === '*') { setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)]); setLastRowAt(Date.now()); } });
    events.addEventListener('done', finish);
    events.onerror = () => { if (events.readyState === EventSource.CLOSED) finish(); };
  };
  const follow = (id: string) => {
    watched.current = id;
    watch(id, run => { remember(run); if (watched.current === id) openRun(run); });
  };
  const runFixture = async (kind: RunPath, text: string = question) => {
    closeStreams(); setQuestion(text); setLibraryOpen(false); setPath(kind); setPhase(kind === 'agentic' ? 'proposing interpretations — you choose one before research' : 'research'); setScene('running'); setLiveTrace([]); setNotice(''); setActiveId('');
    try {
      const { id } = await getJson<{ id: string }>(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, path: kind, budget: kind === 'agentic' ? agMode : baseMode }) });
      setActiveId(id); follow(id);
    } catch (error) { setNotice(`The run could not be started: ${(error as Error).message}`); setScene('failed'); }
  };
  const runCompare = async (text: string = question) => {
    closeStreams(); setQuestion(text); setLibraryOpen(false); setPhase('fast + adaptive research; agentic proposes interpretations'); setScene('running'); setLiveTrace([]); setNotice(''); watched.current = '*';
    const paths = ['baseline', 'baseline-enhanced', 'agentic'] as const;
    let started: { id: string; kind: typeof paths[number] }[];
    try {
      started = await Promise.all(paths.map(async kind => ({ kind, ...(await getJson<{ id: string }>(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, path: kind, budget: kind === 'agentic' ? agMode : baseMode }) })) })));
    } catch (error) { setNotice(`The comparison could not be started: ${(error as Error).message}`); setScene('failed'); return; }
    compareIds.current = started.map(item => item.id);
    const finished: Run[] = [];
    started.forEach(({ id }) => watch(id, run => {
      remember(run); finished.push(run);
      if (finished.length < started.length || watched.current !== '*') return;
      // The agentic leg waits for the user to choose a concept.
      const agentic = finished.find(item => item.path === 'agentic' && item.status === 'awaiting_reading');
      if (agentic) { setNotice('Fast and adaptive runs finished. Choose an interpretation to run the agentic leg.'); openRun(agentic); }
      else setScene('runs');
    }));
  };
  const currentNav = ['idle', 'running', 'readings', 'agentic_result', 'baseline_result', 'failed'].includes(scene) ? 'Ask' : scene === 'gold' ? 'Gold' : scene[0].toUpperCase() + scene.slice(1);
  const showSidebar = !!active && ['agentic_result', 'baseline_result'].includes(scene);
  const busy = scene === 'running';
  useEffect(() => { if (!busy) return; const timer = window.setInterval(() => setTick(value => value + 1), 1000); return () => window.clearInterval(timer); }, [busy]);
  const lastRow = liveTrace[liveTrace.length - 1];
  const waitingOn = busy && lastRow?.action === 'Model call started' ? lastRow.returned : null;
  const waitedS = lastRowAt ? Math.max(0, Math.round((Date.now() - lastRowAt) / 1000)) : 0;
  const changeRating = async (rating: string) => { try { const updated = await getJson<Run>(`${api}/runs/${activeId}/rating`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rating }) }); setRuns(previous => previous.map(run => run.id === activeId ? updated : run)); } catch (error) { setPromotion(`Rating failed: ${(error as Error).message}`); } };
  const promote = async () => { const response = await fetch(`${api}/runs/${activeId}/promote`, { method: 'POST' }); setPromotion(response.ok ? 'Gold candidate created.' : 'Rate this run before promotion.'); };
  const stopRun = async () => { const ids = watched.current === '*' ? compareIds.current : [activeId]; closeStreams(); watched.current = ''; setScene('idle'); await Promise.all(ids.filter(Boolean).map(id => fetch(`${api}/runs/${id}/stop`, { method: 'POST' }).catch(() => undefined))); };
  const chooseReading = async (index: number) => {
    if (!activeId || picking) return;
    setPicking(true);
    try {
      const selected = await getJson<Run>(`${api}/runs/${activeId}/reading`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pick: index }) });
      closeStreams(); setPath(selected.path); setPhase('agentic research on the chosen interpretation'); setActiveId(selected.id); setPick(0); setLiveTrace([]); setScene('running'); follow(selected.id);
    } catch (error) { setNotice(`Could not start research for that interpretation: ${(error as Error).message}`); }
    finally { setPicking(false); }
  };
  const saveGold = async (candidate: GoldCandidate) => {
    const parse = (key: string, fallback: unknown) => { const text = drafts[`${candidate.id}:${key}`]; return text === undefined ? fallback : JSON.parse(text); };
    let body: { gold_claims: unknown; gold_triples: unknown; status: string };
    try { body = { gold_claims: parse('claims', candidate.gold_claims || []), gold_triples: parse('triples', candidate.gold_triples || []), status: 'approved' }; }
    catch (error) { setGoldErrors(errors => ({ ...errors, [candidate.id]: `Invalid JSON: ${(error as Error).message}` })); return; }
    const response = await fetch(`${api}/datasets/candidates/${candidate.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) { setGoldErrors(errors => ({ ...errors, [candidate.id]: String(data.detail || response.statusText) })); return; }
    setCandidates(rows => rows.map(row => row.id === candidate.id ? data as GoldCandidate : row));
    setDrafts(previous => Object.fromEntries(Object.entries(previous).filter(([key]) => !key.startsWith(`${candidate.id}:`))));
    setGoldErrors(errors => ({ ...errors, [candidate.id]: 'Saved and approved.' }));
  };
  const when = (value: string) => new Date(value).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  const pathName = (value: Run['path']) => value === 'agentic' ? 'Agentic' : value === 'baseline-enhanced' ? 'Adaptive' : 'Fast';

  return <div className="page"><header className="nav header"><span className="nav-brand">Research Harness</span>{['Ask', 'Corpus', 'Runs', 'Benchmark', 'Gold', 'Settings'].map(item => <a key={item} href="#" aria-current={currentNav === item ? 'page' : undefined} onClick={e => { e.preventDefault(); setScene(item === 'Ask' ? 'idle' : item.toLowerCase() as Scene); }}>{item}</a>)}</header>
    <main className="main"><div className="content">
      {scene === 'corpus' && <><h2>Corpus</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Papers are parsed once, chunked into passages and indexed. Every path reads the same index, so runs stay comparable. New PDFs are added with <code>make ingest</code>.</p><table className="table" style={{ marginTop: 34 }}><thead><tr><th>Paper</th><th>Identifier</th></tr></thead><tbody>{papers.map(paper => <tr key={paper.paper_id}><td>{paper.title}</td><td className="muted micro">{paper.dblp_iri ? 'DBLP' : 'local'} · {paper.paper_id}</td></tr>)}</tbody></table><p className="muted micro" style={{ marginTop: 16 }}>{papers.length} papers indexed</p></>}
      {scene === 'settings' && <><h2>Settings</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Execution budgets live here so the ask screen stays one question and two runs. A change applies to every run that follows and is recorded with it.</p><div style={{ display: 'grid', gap: 34 }}>
        <div><div className="section-title">Fast baseline budget</div><p className="muted" style={{ fontSize: 13.5 }}>Slow adds an evidence-verification pass over the extracted triples.</p><div className="seg" role="radiogroup" aria-label="Fast baseline budget"><label className="seg-opt"><input type="radio" checked={baseMode === 'fast'} onChange={() => saveSettings({ baseline: 'fast' })} />Fast</label><label className="seg-opt"><input type="radio" checked={baseMode === 'slow'} onChange={() => saveSettings({ baseline: 'slow' })} />Slow · evidence review</label></div></div>
        <div><div className="section-title">Agentic research budget</div><p className="muted" style={{ fontSize: 13.5 }}>Fast caps planning depth and skips re-checks; the agent still chooses its own actions.</p><div className="seg" role="radiogroup" aria-label="Agentic research budget"><label className="seg-opt"><input type="radio" checked={agMode === 'fast'} onChange={() => saveSettings({ agentic: 'fast' })} />Fast</label><label className="seg-opt"><input type="radio" checked={agMode === 'slow'} onChange={() => saveSettings({ agentic: 'slow' })} />Slow · deeper research</label></div></div>
      </div></>}
      {scene === 'runs' && <><h2>Runs</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Every run keeps its artifacts: the concept plans it considered, the reading it settled on, the subschema and query it ran, the triples and the passages behind them.</p>{runs.map(run => <details className="run-row" data-run-id={run.id} key={run.id}><summary><span className="muted micro">{when(run.startedAt)}</span><span style={{ flex: '1 1 240px', fontSize: 15.5 }}>{run.question}</span><span className="muted micro">{pathName(run.path)} · {run.budget} · {run.durationS} s</span></summary><div style={{ display: 'grid', gap: 10, marginTop: 14 }}><div>{run.status}{run.reason ? ` — ${run.reason}` : ''}</div><div><span className="muted">Reading&nbsp;&nbsp;</span>{run.concept?.label || run.readings[run.chosenReading ?? 0]?.label || 'No reading chosen'}</div><div style={{ display: 'flex', gap: '10px 22px', flexWrap: 'wrap' }}>{run.artifacts.map(name => <button key={name} className="text-button micro" style={{ color: openArtifact === `${run.id}/${name}` ? 'var(--color-text)' : undefined }} onClick={() => setOpenArtifact(openArtifact === `${run.id}/${name}` ? '' : `${run.id}/${name}`)}>{name}</button>)}</div>{openArtifact.startsWith(`${run.id}/`) && <pre className="code">{artifactBody}</pre>}<div><button className="text-button" style={{ fontSize: 13 }} onClick={() => openRun(run)}>Open run</button><span className="muted micro" style={{ marginLeft: 24 }}>{run.rating ? `Rated ${run.rating}` : 'Not rated'} · {run.id}</span></div></div></details>)}</>}
      {scene === 'gold' && <><h2>Gold candidates</h2><p className="muted">Review claim subgraphs and their provenance before approving a dataset row. Triples remain diagnostic. JSON is checked when you save.</p>{candidates.length === 0 && <p className="muted">No candidates yet. Rate a run, then promote it from the result sidebar.</p>}{candidates.map(candidate => { const claimsKey = `${candidate.id}:claims`; const triplesKey = `${candidate.id}:triples`; return <details className="run-row" key={candidate.id}><summary><span>{candidate.question}</span><span className="muted micro">{candidate.status}</span></summary><div className="muted micro">Gold claims</div><textarea className="input code" aria-label={`Claims for ${candidate.question}`} value={drafts[claimsKey] ?? JSON.stringify(candidate.gold_claims || [], null, 2)} onChange={event => { const value = event.target.value; setDrafts(previous => ({ ...previous, [claimsKey]: value })); }} /><div className="muted micro">Diagnostic triples</div><textarea className="input code" aria-label={`Triples for ${candidate.question}`} value={drafts[triplesKey] ?? JSON.stringify(candidate.gold_triples || [], null, 2)} onChange={event => { const value = event.target.value; setDrafts(previous => ({ ...previous, [triplesKey]: value })); }} /><button className="btn btn-secondary" onClick={() => saveGold(candidate)}>Save and approve</button>{goldErrors[candidate.id] && <p className="muted micro" role="status">{goldErrors[candidate.id]}</p>}</details>; })}</>}
      {scene === 'benchmark' && <><h2>Benchmark</h2><p className="muted">Approved claim subgraphs aggregated across evaluation repetitions. Edge scores remain diagnostic. Agentic repetitions research the agent's top-ranked concept.</p><button className="btn btn-secondary" onClick={loadBenchmark}>Refresh</button>{benchmarkRows.length === 0 && <p className="muted" style={{ marginTop: 20 }}>No approved gold questions yet. Approve a candidate on the Gold page first.</p>}<table className="table"><thead><tr><th>Question</th><th>Category</th><th>Fast</th><th>Adaptive</th><th>Agentic</th></tr></thead><tbody>{benchmarkRows.map(row => <tr key={row.id}><td><details><summary>{row.question}</summary><div className="muted micro">Gold {row.gold_version}</div><button className="text-button micro" onClick={() => runBenchmark(row)}>Run benchmark · 3× per path</button>{row.paths.map(path => <div key={path.path} className="micro">{path.path}: {path.completed}/{path.completed + path.failed} runs · edge F1 {path.f1.mean?.toFixed(2) || '—'} · repetitions {path.repetitions.join(', ')}</div>)}</details></td><td>{row.category}</td>{['baseline', 'adaptive-baseline', 'agentic'].map(pathName => { const path = row.paths.find(item => item.path === pathName || (pathName === 'adaptive-baseline' && item.path === 'baseline-enhanced')); return <td key={pathName}>{path ? path.completed ? `Claim F1 ${path.claim_f1.mean?.toFixed(2) ?? '—'} · P ${path.claim_precision.mean?.toFixed(2) ?? '—'} · R ${path.claim_recall.mean?.toFixed(2) ?? '—'}` : `${path.failed} failed` : '—'}</td>; })}</tr>)}</tbody></table></>}
      {notice && <p className="notice" role="alert">{notice}</p>}
      {!['corpus', 'settings', 'runs', 'gold', 'benchmark'].includes(scene) && <><div className="question-wrap"><textarea className="input question" rows={Math.max(1, Math.ceil(question.length / 38))} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about the corpus" aria-label="Question" /></div><div className="actions"><button className="btn btn-secondary" disabled={busy || !question.trim()} onClick={() => runFixture('baseline')}>Fast</button><button className="btn btn-secondary" disabled={busy || !question.trim()} onClick={() => runFixture('baseline-enhanced')}>Adaptive</button><button className="btn btn-primary" disabled={busy || !question.trim()} onClick={() => runFixture('agentic')}>Agentic</button><button className="btn btn-secondary" disabled={busy || !question.trim()} onClick={() => runCompare()}>Compare all</button><button className="text-button micro" aria-expanded={libraryOpen} aria-controls="question-library" disabled={busy} onClick={() => setLibraryOpen(open => !open)}>{libraryOpen ? 'Close question library' : `Question library (${library.length})`}</button><button className="text-button muted micro" onClick={() => setScene('settings')}>Budgets: baseline {baseMode}, agentic {agMode} · change</button></div>
        {libraryOpen && !busy && <QuestionLibrary items={library} filter={libraryFilter} setFilter={setLibraryFilter} onPick={text => { setQuestion(text); setLibraryOpen(false); }} onRun={(kind, text) => runFixture(kind, text)} onCompare={text => runCompare(text)} />}
        {scene === 'idle' && <section style={{ marginTop: 72, display: 'grid', gap: 16 }}><span className="section-title">Recent</span>{runs.filter(r => r.status === 'answered' || r.status === 'corpus_absent' || r.status === 'no_evidence_found' || r.status === 'incomplete').filter((run, index, all) => all.findIndex(other => other.question === run.question) === index).slice(0, 2).map(run => <button key={run.id} className="text-button" style={{ color: 'var(--color-text)' }} onClick={() => openRun(run)}>{run.question}</button>)}</section>}
        {scene === 'running' && <section style={{ marginTop: 60 }}><div className="section-top" style={{ marginBottom: 18 }}><span className="section-title">{path === 'agentic' ? 'Agentic research' : path === 'baseline-enhanced' ? 'Adaptive baseline' : 'Fast baseline'}</span><span className="muted micro">{path === 'agentic' ? 'Concept · shared evidence graph · MCP verification' : path === 'baseline-enhanced' ? 'Iterative retrieval · batched extraction · evidence review' : 'Fixed pipeline · no verification pass'}</span></div><Progress rows={liveTrace} waitingOn={waitingOn} waitedS={waitedS} phase={phase} />{liveTrace.map((step, i) => <div className={`log-row${step.action.startsWith('Model call') ? ' log-model' : ''}${step.action === 'Model call failed' ? ' log-failed' : ''}`} key={i}><span className="log-time muted">{step.atS}s</span><span>{step.action}</span><span className="muted micro">{step.returned}</span></div>)}{waitingOn && <div className="log-row log-waiting" role="status"><span className="log-time muted">…</span><span>Waiting {waitedS}s</span><span className="muted micro">{waitingOn}</span></div>}{busy && !waitingOn && lastRow && waitedS >= 10 && <div className="log-row log-waiting" role="status"><span className="log-time muted">…</span><span>Working {waitedS}s</span><span className="muted micro">since “{lastRow.action}”</span></div>}<button className="text-button" style={{ marginTop: 24, fontSize: 13 }} onClick={stopRun}>Stop run</button></section>}
        {scene === 'readings' && active && <section style={{ marginTop: 60 }}><p style={{ maxWidth: '56ch' }}>The agent proposed {active.readings.length} interpretations of this question. Choose one to research.</p><p className="muted" style={{ fontSize: 13.5, marginBottom: 34 }}>Each interpretation scopes retrieval and extraction differently; the question itself stays unchanged.</p>{active.readings.map((r, i) => { const concept = active.concepts[i]; return <button key={i} className="reading-row" disabled={picking} onClick={() => chooseReading(i)}><span className="muted micro">{i + 1}</span><span style={{ display: 'grid', gap: 4 }}><strong style={{ fontSize: 16.5 }}>{r.label}</strong><span className="muted" style={{ fontSize: 13.5 }}>{r.gloss}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{[r.classes.join(' · '), r.properties.join(' · '), concept?.evidenceHits != null ? `${concept.evidenceHits} matching corpus passages` : ''].filter(Boolean).join('  |  ')}</span></span></button>; })}<button className="text-button" style={{ marginTop: 22, fontSize: 13 }} onClick={() => { setScene('idle'); }}>None of these — rephrase the question</button></section>}
        {scene === 'failed' && !active && !notice && <p className="notice">The run failed before it produced a result.</p>}
        {['agentic_result', 'baseline_result', 'failed'].includes(scene) && active && <Result key={`${active.id}-${pick}`} run={active} pick={pick} setPick={setPick} labels={labels} onDeeper={() => runFixture('agentic')} />}
      </>}
    </div>{showSidebar && active && <><Sidebar run={active} reading={reading} onRate={changeRating} onPromote={promote} />{promotion && <span className="muted micro">{promotion}</span>}</>}</main></div>;
}
