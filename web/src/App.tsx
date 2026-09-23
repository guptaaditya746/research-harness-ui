import { useEffect, useMemo, useState } from 'react';
import type { ClaimSubgraph, Reading, Run, Triple } from './generated/run';

type Scene = 'idle' | 'running' | 'readings' | 'agentic_result' | 'baseline_result' | 'failed' | 'runs' | 'corpus' | 'settings' | 'gold' | 'benchmark';
type TripleView = 'statements' | 'graph' | 'turtle';
type GoldCandidate = { id: string; run_id: string; question: string; gold_triples: Triple[]; gold_claims?: ClaimSubgraph[]; notes: string; status: string };
type PaperLabel = { paper_id: string; dblp_iri?: string | null; title: string };
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
  const [view, setView] = useState<TripleView>('statements');
  const label = (value: string) => labels[value] ? `${labels[value]} (${value})` : value;
  return <section className="section" aria-label="Triples"><div className="section-top"><span className="section-title">Triples</span><div className="section-links">
    {(['statements', 'graph', 'turtle'] as const).map(v => <button key={v} className="text-button micro" style={{ color: view === v ? 'var(--color-text)' : undefined }} onClick={() => setView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>)}
  </div></div>
    {view === 'statements' && reading.triples.map((t: Triple, i: number) => <details className="row-details" key={i}><summary><span style={{ fontSize: 15.5 }}>{label(t.s)}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{t.p}</span><span style={{ fontSize: 15.5 }}>{label(t.o)}</span><span className="muted" style={{ marginLeft: 'auto', fontSize: 12 }}>{t.support}</span></summary><p style={{ margin: '12px 0 0', fontStyle: 'italic', fontSize: 14 }}>{t.evidence}</p><div className="muted micro">{t.source}</div></details>)}
    {view === 'graph' && reading.triples.map((t: Triple, i: number) => <div className="graph-row" key={i}><span>{label(t.s)}</span><span className="graph-edge">{t.p}</span><span style={{ textAlign: 'right' }}>{label(t.o)}</span></div>)}
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

function Result({ run, pick, setPick, onDeeper, labels }: { run: Run; pick: number; setPick: (index: number) => void; onDeeper: () => void; labels: Record<string, string> }) {
  const [showOntology, setShowOntology] = useState(false);
  const [showQuery, setShowQuery] = useState(false);
  const reading = run.readings[pick] || run.readings[0];
  const failed = run.status === 'incomplete';
  const absent = run.status === 'corpus_absent';
  const noEvidence = run.status === 'no_evidence_found';
  const pathLabel = run.path === 'baseline-enhanced' ? 'adaptive baseline' : run.path === 'baseline' ? 'fast baseline' : 'agentic research';
  const failure = run.reason === 'provider_error'
    ? `The ${pathLabel} retrieved passages, but the model gateway failed while inferring the schema. Start or reconnect the local model service at LITELLM_BASE_URL, then run again.`
    : run.reason === 'no_results'
      ? `The ${pathLabel} could not find enough supported evidence in the corpus to answer this question.`
      : `The ${pathLabel} ended before it produced an answer (${run.reason || 'unknown failure'}).`;
  return <div style={{ marginTop: 64 }}>
    <p className="notice">{failed ? `Incomplete — ${pathLabel} stopped before answer composition.` : absent ? `No in-corpus match found. ${run.evidenceSurface || 'Corpus evidence'} was checked${run.searchCoverage?.inspected != null ? ` (${run.searchCoverage.inspected} records)` : ''}.` : noEvidence ? `No evidence found on ${run.evidenceSurface || 'the indexed full-text surface'} after the declared search procedure; this is not proof of corpus absence.` : run.executionStatus === 'degraded' ? 'Agentic execution degraded; showing a grounded fallback excluded from agentic scores.' : `Interpreted as: ${reading?.label}.${run.coverage != null && run.coverage < 1 ? ` Evaluated ${Math.round(run.coverage * 100)}% of obligations; ranking is provisional.` : ''}`}</p>
    {failed && <button className="btn btn-secondary" style={{ marginTop: 24 }} onClick={onDeeper}>Try deeper research</button>}
    {!failed && run.path === 'agentic' && run.readings.length > 1 && <div style={{ marginTop: 20 }}><div className="muted micro" style={{ marginBottom: 8 }}>Re-answer under another reading</div><div style={{ display: 'flex', gap: '12px 24px', flexWrap: 'wrap' }}>{run.readings.map((r, i) => i !== pick && <button key={i} className="text-button" style={{ fontSize: 13 }} onClick={() => setPick(i)}>{r.label}</button>)}</div></div>}
    {(absent || noEvidence) && <section className="section" aria-label="Research result"><div className="section-top"><span className="section-title">Research result</span></div><p className="answer">{run.reason}</p><pre className="code">{JSON.stringify(run.searchCoverage, null, 2)}</pre></section>}
    {!failed && !absent && !noEvidence && reading && <><section className="section" aria-label="Subschema"><div className="section-top"><span className="section-title">Subschema</span><div className="section-links"><button className="text-button micro" onClick={() => setShowOntology(!showOntology)}>{showOntology ? 'Hide ontology' : 'Ontology (Turtle)'}</button><button className="text-button micro" onClick={() => setShowQuery(!showQuery)}>{showQuery ? 'Hide query' : 'SPARQL'}</button></div></div>
      <p className="muted" style={{ fontSize: 14, margin: '10px 0 16px' }}>{run.path === 'baseline' ? 'Unstated — inferred by the extractor. ' : ''}The answer is computed over {reading.classes.map(c => c.toLowerCase() + 's').join(', ')} only. Nothing outside these is read.</p>
      <div style={{ display: 'grid', gap: 12 }}>{reading.axioms.map(([property, domain, range]) => <div key={property}><div style={{ fontSize: 14.5 }}>A {domain.toLowerCase()} has {property.replaceAll('_', ' ')} relating it to {range.replace('xsd:', '')}.</div><div className="muted" style={{ font: '11.5px ui-monospace, Menlo, monospace' }}>{`hrn:${property}  rdfs:domain hrn:${domain}  rdfs:range ${range.includes(':') ? range : `hrn:${range}`}`}</div></div>)}</div>
      {showOntology && <pre className="code">{ontology(reading)}</pre>}{showQuery && <pre className="code">{reading.sparql}</pre>}
    </section><Claims reading={reading} labels={labels} /><Triples key={pick} reading={reading} labels={labels} /><section className="section" aria-label="Natural-language answer" style={{ marginTop: 34 }}><div className="section-top"><span className="section-title">Natural-language answer</span><span className="muted micro">Downstream of accepted claims</span></div><p className="answer">{reading.answer}</p></section></>}
    {failed && <section className="section" aria-label="Run result" style={{ marginTop: 34 }}><div className="section-top"><span className="section-title">Natural-language answer</span></div><p className="answer">{failure}</p></section>}
  </div>;
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
  const active = useMemo(() => runs.find(r => r.id === activeId), [runs, activeId]);
  const reading = active?.readings[pick] || active?.readings[0];

  useEffect(() => { fetch(`${api}/corpus/papers`).then(r => r.json()).then((items: PaperLabel[]) => setLabels(Object.fromEntries(items.flatMap(item => [[item.paper_id, item.title], ...(item.dblp_iri ? [[item.dblp_iri, item.title]] : [])])))); fetch(`${api}/runs`).then(r => r.json()).then((items: Run[]) => {
    setRuns(items);
    const id = screenRun[initial || ''];
    if (id) setQuestion(items.find(run => run.id === id)?.question || '');
    if (initial === 'running') setQuestion(items.find(run => run.id === 'r-0412')?.question || '');
  }); }, []);
  useEffect(() => { if (!openArtifact) return; const [id, name] = openArtifact.split('/');
    fetch(`${api}/runs/${id}/artifacts/${name}`).then(r => r.json()).then(data => setArtifactBody(data.content));
  }, [openArtifact]);
  useEffect(() => { if (scene !== 'gold') return; fetch(`${api}/datasets/candidates`).then(r => r.json()).then(setCandidates); }, [scene]);
  useEffect(() => { if (scene !== 'benchmark') return; fetch(`${api}/benchmark`).then(r => r.json()).then(data => setBenchmarkRows(data.questions || [])); }, [scene]);

  const openRun = (run: Run) => { setActiveId(run.id); setQuestion(run.question); setPick(run.chosenReading ?? 0); setScene(run.status === 'awaiting_reading' ? 'readings' : run.status === 'incomplete' ? 'failed' : run.status === 'stopped' ? 'idle' : run.path === 'agentic' ? 'agentic_result' : 'baseline_result'); };
  const runFixture = async (kind: 'agentic' | 'baseline' | 'baseline-enhanced') => {
    setPath(kind); setScene('running'); setLiveTrace([]);
    const response = await fetch(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, path: kind, budget: kind === 'agentic' ? agMode : baseMode }) });
    if (!response.ok) { setScene('failed'); return; }
    const { id } = await response.json(); setActiveId(id);
    const events = new EventSource(`${api}/runs/${id}/events`);
    events.addEventListener('trace', event => setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)]));
    events.addEventListener('done', async () => { events.close(); const run = await fetch(`${api}/runs/${id}`).then(r => r.json()) as Run; setRuns(items => [run, ...items.filter(item => item.id !== id)]); openRun(run); });
    events.onerror = () => { events.close(); setScene('failed'); };
  };
  const runCompare = async () => {
    setScene('running'); setLiveTrace([]);
    const paths = ['baseline', 'baseline-enhanced', 'agentic'] as const;
    const responses = await Promise.all(paths.map(kind => fetch(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, path: kind, budget: kind === 'agentic' ? agMode : baseMode }) })));
    if (responses.some(response => !response.ok)) { setScene('failed'); return; }
    const started = await Promise.all(responses.map(response => response.json() as Promise<{ id: string }>));
    let remaining = started.length;
    started.forEach(({ id }) => {
      const events = new EventSource(`${api}/runs/${id}/events`);
      events.addEventListener('trace', event => setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)]));
      events.addEventListener('done', async () => { events.close(); const run = await fetch(`${api}/runs/${id}`).then(response => response.json()) as Run; setRuns(items => [run, ...items.filter(item => item.id !== id)]); remaining -= 1; if (remaining === 0) setScene('runs'); });
      events.onerror = () => { events.close(); remaining -= 1; if (remaining === 0) setScene('runs'); };
    });
  };
  const currentNav = ['idle', 'running', 'readings', 'agentic_result', 'baseline_result', 'failed'].includes(scene) ? 'Ask' : scene === 'gold' ? 'Gold' : scene[0].toUpperCase() + scene.slice(1);
  const showSidebar = !!active && ['agentic_result', 'baseline_result'].includes(scene);
  const changeRating = async (rating: string) => { const updated = await fetch(`${api}/runs/${activeId}/rating`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rating }) }).then(r => r.json()) as Run; setRuns(previous => previous.map(run => run.id === activeId ? updated : run)); };
  const promote = async () => { const response = await fetch(`${api}/runs/${activeId}/promote`, { method: 'POST' }); setPromotion(response.ok ? 'Gold candidate created.' : 'Rate this run before promotion.'); };
  const stopRun = async () => { if (!activeId) return; await fetch(`${api}/runs/${activeId}/stop`, { method: 'POST' }); setScene('idle'); setQuestion(''); };
  const chooseReading = async (index: number) => { if (!activeId) return; const response = await fetch(`${api}/runs/${activeId}/reading`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pick: index }) }); if (!response.ok) return; const selected = await response.json() as Run; setActiveId(selected.id); setPick(index); setLiveTrace([]); setScene('running'); const events = new EventSource(`${api}/runs/${selected.id}/events`); events.addEventListener('trace', event => setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)])); events.addEventListener('done', async () => { events.close(); const completed = await fetch(`${api}/runs/${selected.id}`).then(r => r.json()) as Run; setRuns(items => [completed, ...items.filter(item => item.id !== completed.id)]); openRun(completed); }); events.onerror = () => { events.close(); setScene('failed'); }; };

  return <div className="page"><header className="nav header"><span className="nav-brand">Research Harness</span>{['Ask', 'Corpus', 'Runs', 'Benchmark', 'Gold', 'Settings'].map(item => <a key={item} href="#" aria-current={currentNav === item ? 'page' : undefined} onClick={e => { e.preventDefault(); setScene(item === 'Ask' ? 'idle' : item.toLowerCase() as Scene); }}>{item}</a>)}</header>
    <main className="main"><div className="content">
      {scene === 'corpus' && <><h2>Corpus</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Papers are parsed once, chunked into passages and indexed. Both paths read the same index, so runs stay comparable.</p><button className="btn btn-secondary">Add PDFs</button><table className="table" style={{ marginTop: 34 }}><thead><tr><th>Paper</th><th>Year</th><th>Status</th></tr></thead><tbody>{[
        ['Break It Down: A Question Understanding Benchmark', '2020', 'Indexed'], ['AmbigQA: Answering Ambiguous Open-domain Questions', '2020', 'Indexed'], ['OneKE: A Dockerized Schema-Guided Extraction System', '2024', 'Indexed'], ['Dense Passage Retrieval for Open-Domain QA', '2020', 'Indexed'], ['Measuring Ambiguity in Open-ended Research Questions', '2023', 'Parsing'],
      ].map(([title, year, status]) => <tr key={title}><td>{title}</td><td>{year}</td><td className="muted">{status}</td></tr>)}</tbody></table><p className="muted micro" style={{ marginTop: 16 }}>5 of 42 shown</p></>}
      {scene === 'settings' && <><h2>Settings</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Execution budgets live here so the ask screen stays one question and two runs. A change applies to every run that follows and is recorded with it.</p><div style={{ display: 'grid', gap: 34 }}>
        <div><div className="section-title">Fast baseline budget</div><p className="muted" style={{ fontSize: 13.5 }}>Slow adds an evidence-verification pass over the extracted triples.</p><div className="seg" role="radiogroup" aria-label="Fast baseline budget"><label className="seg-opt"><input type="radio" checked={baseMode === 'fast'} onChange={() => setBaseMode('fast')} />Fast · 4–8 s</label><label className="seg-opt"><input type="radio" checked={baseMode === 'slow'} onChange={() => setBaseMode('slow')} />Slow · 10–15 s</label></div></div>
        <div><div className="section-title">Agentic research budget</div><p className="muted" style={{ fontSize: 13.5 }}>Fast caps planning depth and skips re-checks; the agent still chooses its own actions.</p><div className="seg" role="radiogroup" aria-label="Agentic research budget"><label className="seg-opt"><input type="radio" checked={agMode === 'fast'} onChange={() => setAgMode('fast')} />Fast · 45 s</label><label className="seg-opt"><input type="radio" checked={agMode === 'slow'} onChange={() => setAgMode('slow')} />Slow · 120 s</label></div></div>
      </div></>}
      {scene === 'runs' && <><h2>Runs</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Every run keeps its artifacts: the concept plans it considered, the reading it settled on, the subschema and query it ran, the triples and the passages behind them.</p>{runs.map(run => <details className="run-row" data-run-id={run.id} key={run.id}><summary><span className="muted micro">11 Sep, 16:20</span><span style={{ flex: '1 1 240px', fontSize: 15.5 }}>{run.question}</span><span className="muted micro">{run.path === 'agentic' ? 'Agentic' : 'Fast'} · {run.budget} · {run.durationS} s</span></summary><div style={{ display: 'grid', gap: 10, marginTop: 14 }}><div>{run.status}{run.reason ? ` — ${run.reason}` : ''}</div><div><span className="muted">Reading&nbsp;&nbsp;</span>{run.readings[run.chosenReading ?? 0]?.label || 'No reading chosen'}</div><div style={{ display: 'flex', gap: '10px 22px', flexWrap: 'wrap' }}>{run.artifacts.map(name => <button key={name} className="text-button micro" style={{ color: openArtifact === `${run.id}/${name}` ? 'var(--color-text)' : undefined }} onClick={() => setOpenArtifact(openArtifact === `${run.id}/${name}` ? '' : `${run.id}/${name}`)}>{name}</button>)}</div>{openArtifact.startsWith(`${run.id}/`) && <pre className="code">{artifactBody}</pre>}<div><button className="text-button" style={{ fontSize: 13 }} onClick={() => openRun(run)}>Open run</button><span className="muted micro" style={{ marginLeft: 24 }}>{run.rating ? `Rated ${run.rating}` : 'Not rated'} · {run.id}</span></div></div></details>)}</>}
      {scene === 'gold' && <><h2>Gold candidates</h2><p className="muted">Review claim subgraphs and their provenance before approving a dataset row. Triples remain diagnostic.</p>{candidates.map((candidate, index) => <details className="run-row" key={candidate.id}><summary><span>{candidate.question}</span><span className="muted micro">{candidate.status}</span></summary><div className="muted micro">Gold claims</div><textarea className="input code" aria-label={`Claims for ${candidate.question}`} value={JSON.stringify(candidate.gold_claims || [], null, 2)} onChange={event => { try { const claims = JSON.parse(event.target.value); setCandidates(rows => rows.map((row, i) => i === index ? { ...row, gold_claims: claims } : row)); } catch { /* keep the last valid JSON */ } }} /><div className="muted micro">Diagnostic triples</div><textarea className="input code" aria-label={`Triples for ${candidate.question}`} value={JSON.stringify(candidate.gold_triples, null, 2)} onChange={event => { try { const triples = JSON.parse(event.target.value); setCandidates(rows => rows.map((row, i) => i === index ? { ...row, gold_triples: triples } : row)); } catch { /* keep the last valid JSON */ } }} /><button className="btn btn-secondary" onClick={async () => { const updated = await fetch(`${api}/datasets/candidates/${candidate.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gold_claims: candidate.gold_claims || [], gold_triples: candidate.gold_triples, status: 'approved' }) }).then(r => r.json()); setCandidates(rows => rows.map(row => row.id === candidate.id ? updated : row)); }}>Save and approve</button></details>)}</>}
      {scene === 'benchmark' && <><h2>Benchmark</h2><p className="muted">Approved claim subgraphs aggregated across evaluation repetitions. Edge scores remain diagnostic.</p><table className="table"><thead><tr><th>Question</th><th>Category</th><th>Fast</th><th>Adaptive</th><th>Agentic</th></tr></thead><tbody>{benchmarkRows.map(row => <tr key={row.id}><td><details><summary>{row.question}</summary><div className="muted micro">Gold {row.gold_version}</div>{row.paths.map(path => <div key={path.path} className="micro">{path.path}: {path.completed}/{path.completed + path.failed} runs · edge F1 {path.f1.mean?.toFixed(2) || '—'} · repetitions {path.repetitions.join(', ')}</div>)}</details></td><td>{row.category}</td>{['baseline', 'adaptive-baseline', 'agentic'].map(pathName => { const path = row.paths.find(item => item.path === pathName || (pathName === 'adaptive-baseline' && item.path === 'baseline-enhanced')); return <td key={pathName}>{path ? path.completed ? `Claim F1 ${path.claim_f1.mean?.toFixed(2) ?? '—'} · P ${path.claim_precision.mean?.toFixed(2) ?? '—'} · R ${path.claim_recall.mean?.toFixed(2) ?? '—'}` : `${path.failed} failed` : '—'}</td>; })}</tr>)}</tbody></table></>}
      {!['corpus', 'settings', 'runs', 'gold'].includes(scene) && <><div className="question-wrap"><textarea className="input question" rows={Math.max(1, Math.ceil(question.length / 38))} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about the corpus" aria-label="Question" /></div><div className="actions"><button className="btn btn-secondary" disabled={!question.trim()} onClick={() => runFixture('baseline')}>Fast</button><button className="btn btn-secondary" disabled={!question.trim()} onClick={() => runFixture('baseline-enhanced')}>Adaptive</button><button className="btn btn-primary" disabled={!question.trim()} onClick={() => runFixture('agentic')}>Agentic</button><button className="btn btn-secondary" disabled={!question.trim()} onClick={runCompare}>Compare all</button><button className="text-button muted micro" onClick={() => setScene('settings')}>Budgets: baseline {baseMode}, agentic {agMode} · change</button></div>
        {scene === 'idle' && <section style={{ marginTop: 72, display: 'grid', gap: 16 }}><span className="section-title">Recent</span>{runs.filter(r => r.status === 'answered' || r.status === 'corpus_absent' || r.status === 'no_evidence_found' || r.status === 'incomplete').slice(0, 2).map(run => <button key={run.id} className="text-button" style={{ color: 'var(--color-text)' }} onClick={() => openRun(run)}>{run.question}</button>)}</section>}
        {scene === 'running' && <section style={{ marginTop: 60 }}><div className="section-top" style={{ marginBottom: 18 }}><span className="section-title">{path === 'agentic' ? 'Agentic research' : path === 'baseline-enhanced' ? 'Adaptive baseline' : 'Fast baseline'}</span><span className="muted micro">{path === 'agentic' ? 'Planner lenses · shared evidence graph · verification' : path === 'baseline-enhanced' ? 'Iterative retrieval · batched extraction · clarification' : 'Fixed pipeline · no verification pass'}</span></div>{liveTrace.map((step, i) => <div className="log-row" key={i}><span className="log-time muted">{step.atS}s</span><span>{step.action}</span><span className="muted micro">{step.returned}</span></div>)}<button className="text-button" style={{ marginTop: 24, fontSize: 13 }} onClick={stopRun}>Stop run</button></section>}
        {scene === 'readings' && active && <section style={{ marginTop: 60 }}><p style={{ maxWidth: '56ch' }}>The agent produced multiple supported readings. Choose one to continue.</p><p className="muted" style={{ fontSize: 13.5, marginBottom: 34 }}>Each binds a different subschema and produces a different query.</p>{active.readings.map((r, i) => <button key={i} className="reading-row" onClick={() => chooseReading(i)}><span className="muted micro">{i + 1}</span><span style={{ display: 'grid', gap: 4 }}><strong style={{ fontSize: 16.5 }}>{r.label}</strong><span className="muted" style={{ fontSize: 13.5 }}>{r.gloss}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{r.classes.join(' · ')} &nbsp; | &nbsp; {r.properties.join(' · ')}</span></span></button>)}<button className="text-button" style={{ marginTop: 22, fontSize: 13 }} onClick={() => { setScene('idle'); setQuestion(''); }}>None of these — rephrase the question</button></section>}
        {['agentic_result', 'baseline_result', 'failed'].includes(scene) && active && <Result key={`${active.id}-${pick}`} run={active} pick={pick} setPick={setPick} labels={labels} onDeeper={() => { setActiveId('r-0413'); setScene('readings'); }} />}
      </>}
    </div>{showSidebar && active && <><Sidebar run={active} reading={reading} onRate={changeRating} onPromote={promote} />{promotion && <span className="muted micro">{promotion}</span>}</>}</main></div>;
}
