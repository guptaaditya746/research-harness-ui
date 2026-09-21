import { useEffect, useMemo, useState } from 'react';
import type { Reading, Run, Triple } from './generated/run';

type Scene = 'idle' | 'running' | 'readings' | 'agentic_result' | 'baseline_result' | 'failed' | 'runs' | 'corpus' | 'settings' | 'gold';
type TripleView = 'statements' | 'graph' | 'turtle';
type GoldCandidate = { id: string; run_id: string; question: string; gold_triples: Triple[]; notes: string; status: string };
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
      <p style={{ fontSize: 13.5, margin: 0 }}>{run.path === 'baseline' ? 'Answered under an unstated criterion. The baseline does not detect the ambiguity.' : 'Ambiguous — the success criterion is under-specified.'}</p></div>
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

function Triples({ reading }: { reading: Reading }) {
  const [view, setView] = useState<TripleView>('statements');
  return <section className="section" aria-label="Triples"><div className="section-top"><span className="section-title">Triples</span><div className="section-links">
    {(['statements', 'graph', 'turtle'] as const).map(v => <button key={v} className="text-button micro" style={{ color: view === v ? 'var(--color-text)' : undefined }} onClick={() => setView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>)}
  </div></div>
    {view === 'statements' && reading.triples.map((t: Triple, i: number) => <details className="row-details" key={i}><summary><span style={{ fontSize: 15.5 }}>{t.s}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{t.p}</span><span style={{ fontSize: 15.5 }}>{t.o}</span><span className="muted" style={{ marginLeft: 'auto', fontSize: 12 }}>{t.support}</span></summary><p style={{ margin: '12px 0 0', fontStyle: 'italic', fontSize: 14 }}>{t.evidence}</p><div className="muted micro">{t.source}</div></details>)}
    {view === 'graph' && reading.triples.map((t: Triple, i: number) => <div className="graph-row" key={i}><span>{t.s}</span><span className="graph-edge">{t.p}</span><span style={{ textAlign: 'right' }}>{t.o}</span></div>)}
    {view === 'turtle' && <pre className="code">{turtle(reading)}</pre>}
  </section>;
}

function Result({ run, pick, setPick, onDeeper }: { run: Run; pick: number; setPick: (index: number) => void; onDeeper: () => void }) {
  const [showOntology, setShowOntology] = useState(false);
  const [showQuery, setShowQuery] = useState(false);
  const reading = run.readings[pick] || run.readings[0];
  const failed = run.status === 'incomplete';
  return <div style={{ marginTop: 64 }}>
    <p className="notice">{failed ? 'Incomplete — the fast path stopped after retrieval.' : run.path === 'baseline' ? 'No criterion was stated and none was requested — the baseline chose one silently.' : `Interpreted as: ${reading?.label}.${run.coverage != null && run.coverage < 1 ? ` Evaluated ${Math.round(run.coverage * 100)}% of obligations; ranking is provisional.` : ''}`}</p>
    <p className="answer">{failed ? '“Hardest subset” could not be resolved from the corpus. Retrieval returned passages on benchmark difficulty, but the papers define no shared hard subset.' : reading?.answer}</p>
    {failed && <button className="btn btn-secondary" style={{ marginTop: 24 }} onClick={onDeeper}>Try deeper research</button>}
    {!failed && run.path === 'agentic' && run.readings.length > 1 && <div style={{ marginTop: 20 }}><div className="muted micro" style={{ marginBottom: 8 }}>Re-answer under another reading</div><div style={{ display: 'flex', gap: '12px 24px', flexWrap: 'wrap' }}>{run.readings.map((r, i) => i !== pick && <button key={i} className="text-button" style={{ fontSize: 13 }} onClick={() => setPick(i)}>{r.label}</button>)}</div></div>}
    {!failed && run.path === 'baseline' && <div style={{ marginTop: 20 }}><div className="muted micro">Re-answer under a stated reading (agentic)</div><button className="text-button" style={{ fontSize: 13, marginTop: 8 }} onClick={onDeeper}>Citation count within the corpus</button></div>}
    {!failed && reading && <><section className="section" aria-label="Subschema"><div className="section-top"><span className="section-title">Subschema</span><div className="section-links"><button className="text-button micro" onClick={() => setShowOntology(!showOntology)}>{showOntology ? 'Hide ontology' : 'Ontology (Turtle)'}</button><button className="text-button micro" onClick={() => setShowQuery(!showQuery)}>{showQuery ? 'Hide query' : 'SPARQL'}</button></div></div>
      <p className="muted" style={{ fontSize: 14, margin: '10px 0 16px' }}>{run.path === 'baseline' ? 'Unstated — inferred by the extractor. ' : ''}The answer is computed over {reading.classes.map(c => c.toLowerCase() + 's').join(', ')} only. Nothing outside these is read.</p>
      <div style={{ display: 'grid', gap: 12 }}>{reading.axioms.map(([property, domain, range]) => <div key={property}><div style={{ fontSize: 14.5 }}>A {domain.toLowerCase()} has {property.replaceAll('_', ' ')} relating it to {range.replace('xsd:', '')}.</div><div className="muted" style={{ font: '11.5px ui-monospace, Menlo, monospace' }}>{`hrn:${property}  rdfs:domain hrn:${domain}  rdfs:range ${range.includes(':') ? range : `hrn:${range}`}`}</div></div>)}</div>
      {showOntology && <pre className="code">{ontology(reading)}</pre>}{showQuery && <pre className="code">{reading.sparql}</pre>}
    </section><Triples key={pick} reading={reading} /></>}
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
  const [promotion, setPromotion] = useState('');
  const active = useMemo(() => runs.find(r => r.id === activeId), [runs, activeId]);
  const reading = active?.readings[pick] || active?.readings[0];

  useEffect(() => { fetch(`${api}/runs`).then(r => r.json()).then((items: Run[]) => {
    setRuns(items);
    const id = screenRun[initial || ''];
    if (id) setQuestion(items.find(run => run.id === id)?.question || '');
    if (initial === 'running') setQuestion(items.find(run => run.id === 'r-0412')?.question || '');
  }); }, []);
  useEffect(() => { if (!openArtifact) return; const [id, name] = openArtifact.split('/');
    fetch(`${api}/runs/${id}/artifacts/${name}`).then(r => r.json()).then(data => setArtifactBody(data.content));
  }, [openArtifact]);
  useEffect(() => { if (scene !== 'gold') return; fetch(`${api}/datasets/candidates`).then(r => r.json()).then(setCandidates); }, [scene]);

  const openRun = (run: Run) => { setActiveId(run.id); setQuestion(run.question); setPick(run.chosenReading ?? 0); setScene(run.status === 'awaiting_reading' ? 'readings' : run.status === 'incomplete' ? 'failed' : run.path === 'agentic' ? 'agentic_result' : 'baseline_result'); };
  const runFixture = async (kind: 'agentic' | 'baseline' | 'baseline-enhanced') => {
    setPath(kind); setScene('running'); setLiveTrace([]);
    if (kind === 'agentic') { setActiveId('r-0413'); return; }
    const response = await fetch(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, path: kind, budget: baseMode }) });
    if (!response.ok) { setScene('failed'); return; }
    const { id } = await response.json(); setActiveId(id);
    const events = new EventSource(`${api}/runs/${id}/events`);
    events.addEventListener('trace', event => setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)]));
    events.addEventListener('done', async () => { events.close(); const run = await fetch(`${api}/runs/${id}`).then(r => r.json()) as Run; setRuns(items => [run, ...items.filter(item => item.id !== id)]); openRun(run); });
    events.onerror = () => { events.close(); setScene('failed'); };
  };
  const currentNav = ['idle', 'running', 'readings', 'agentic_result', 'baseline_result', 'failed'].includes(scene) ? 'Ask' : scene === 'gold' ? 'Gold' : scene[0].toUpperCase() + scene.slice(1);
  const showSidebar = !!active && ['agentic_result', 'baseline_result'].includes(scene);
  const changeRating = async (rating: string) => { const updated = await fetch(`${api}/runs/${activeId}/rating`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rating }) }).then(r => r.json()) as Run; setRuns(previous => previous.map(run => run.id === activeId ? updated : run)); };
  const promote = async () => { const response = await fetch(`${api}/runs/${activeId}/promote`, { method: 'POST' }); setPromotion(response.ok ? 'Gold candidate created.' : 'Rate this run before promotion.'); };

  return <div className="page"><header className="nav header"><span className="nav-brand">Research Harness</span>{['Ask', 'Corpus', 'Runs', 'Gold', 'Settings'].map(item => <a key={item} href="#" aria-current={currentNav === item ? 'page' : undefined} onClick={e => { e.preventDefault(); setScene(item === 'Ask' ? 'idle' : item.toLowerCase() as Scene); }}>{item}</a>)}</header>
    <main className="main"><div className="content">
      {scene === 'corpus' && <><h2>Corpus</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Papers are parsed once, chunked into passages and indexed. Both paths read the same index, so runs stay comparable.</p><button className="btn btn-secondary">Add PDFs</button><table className="table" style={{ marginTop: 34 }}><thead><tr><th>Paper</th><th>Year</th><th>Status</th></tr></thead><tbody>{[
        ['Break It Down: A Question Understanding Benchmark', '2020', 'Indexed'], ['AmbigQA: Answering Ambiguous Open-domain Questions', '2020', 'Indexed'], ['OneKE: A Dockerized Schema-Guided Extraction System', '2024', 'Indexed'], ['Dense Passage Retrieval for Open-Domain QA', '2020', 'Indexed'], ['Measuring Ambiguity in Open-ended Research Questions', '2023', 'Parsing'],
      ].map(([title, year, status]) => <tr key={title}><td>{title}</td><td>{year}</td><td className="muted">{status}</td></tr>)}</tbody></table><p className="muted micro" style={{ marginTop: 16 }}>5 of 42 shown</p></>}
      {scene === 'settings' && <><h2>Settings</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Execution budgets live here so the ask screen stays one question and two runs. A change applies to every run that follows and is recorded with it.</p><div style={{ display: 'grid', gap: 34 }}>
        <div><div className="section-title">Fast baseline budget</div><p className="muted" style={{ fontSize: 13.5 }}>Slow adds an evidence-verification pass over the extracted triples.</p><div className="seg" role="radiogroup" aria-label="Fast baseline budget"><label className="seg-opt"><input type="radio" checked={baseMode === 'fast'} onChange={() => setBaseMode('fast')} />Fast · 4–8 s</label><label className="seg-opt"><input type="radio" checked={baseMode === 'slow'} onChange={() => setBaseMode('slow')} />Slow · 10–15 s</label></div></div>
        <div><div className="section-title">Agentic research budget</div><p className="muted" style={{ fontSize: 13.5 }}>Fast caps planning depth and skips re-checks; the agent still chooses its own actions.</p><div className="seg" role="radiogroup" aria-label="Agentic research budget"><label className="seg-opt"><input type="radio" checked={agMode === 'fast'} onChange={() => setAgMode('fast')} />Fast · 45 s</label><label className="seg-opt"><input type="radio" checked={agMode === 'slow'} onChange={() => setAgMode('slow')} />Slow · 120 s</label></div></div>
      </div></>}
      {scene === 'runs' && <><h2>Runs</h2><p className="muted" style={{ maxWidth: '56ch', marginBottom: 40 }}>Every run keeps its artifacts: the concept plans it considered, the reading it settled on, the subschema and query it ran, the triples and the passages behind them.</p>{runs.map(run => <details className="run-row" data-run-id={run.id} key={run.id}><summary><span className="muted micro">11 Sep, 16:20</span><span style={{ flex: '1 1 240px', fontSize: 15.5 }}>{run.question}</span><span className="muted micro">{run.path === 'agentic' ? 'Agentic' : 'Fast'} · {run.budget} · {run.durationS} s</span></summary><div style={{ display: 'grid', gap: 10, marginTop: 14 }}><div>{run.status}{run.reason ? ` — ${run.reason}` : ''}</div><div><span className="muted">Reading&nbsp;&nbsp;</span>{run.readings[run.chosenReading ?? 0]?.label || 'No reading chosen'}</div><div style={{ display: 'flex', gap: '10px 22px', flexWrap: 'wrap' }}>{run.artifacts.map(name => <button key={name} className="text-button micro" style={{ color: openArtifact === `${run.id}/${name}` ? 'var(--color-text)' : undefined }} onClick={() => setOpenArtifact(openArtifact === `${run.id}/${name}` ? '' : `${run.id}/${name}`)}>{name}</button>)}</div>{openArtifact.startsWith(`${run.id}/`) && <pre className="code">{artifactBody}</pre>}<div><button className="text-button" style={{ fontSize: 13 }} onClick={() => openRun(run)}>Open run</button><span className="muted micro" style={{ marginLeft: 24 }}>{run.rating ? `Rated ${run.rating}` : 'Not rated'} · {run.id}</span></div></div></details>)}</>}
      {scene === 'gold' && <><h2>Gold candidates</h2><p className="muted">Review promoted triples before approving a dataset row.</p>{candidates.map((candidate, index) => <details className="run-row" key={candidate.id}><summary><span>{candidate.question}</span><span className="muted micro">{candidate.status}</span></summary><textarea className="input code" aria-label={`Triples for ${candidate.question}`} value={JSON.stringify(candidate.gold_triples, null, 2)} onChange={event => { try { const triples = JSON.parse(event.target.value); setCandidates(rows => rows.map((row, i) => i === index ? { ...row, gold_triples: triples } : row)); } catch { /* keep the last valid JSON */ } }} /><button className="btn btn-secondary" onClick={async () => { const updated = await fetch(`${api}/datasets/candidates/${candidate.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gold_triples: candidate.gold_triples, status: 'approved' }) }).then(r => r.json()); setCandidates(rows => rows.map(row => row.id === candidate.id ? updated : row)); }}>Save and approve</button></details>)}</>}
      {!['corpus', 'settings', 'runs', 'gold'].includes(scene) && <><div className="question-wrap"><textarea className="input question" rows={Math.max(1, Math.ceil(question.length / 38))} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about the corpus" aria-label="Question" /></div><div className="actions"><button className="btn btn-secondary" disabled={!question.trim()} onClick={() => runFixture('baseline')}>Fast baseline</button><button className="btn btn-secondary" disabled={!question.trim()} onClick={() => runFixture('baseline-enhanced')}>Enhanced baseline</button><button className="btn btn-primary" disabled={!question.trim()} onClick={() => runFixture('agentic')}>Agentic research</button><button className="text-button muted micro" onClick={() => setScene('settings')}>Budgets: baseline {baseMode}, agentic {agMode} · change</button></div>
        {scene === 'idle' && <section style={{ marginTop: 72, display: 'grid', gap: 16 }}><span className="section-title">Recent</span>{runs.filter(r => r.status === 'answered' || r.status === 'incomplete').slice(0, 2).map(run => <button key={run.id} className="text-button" style={{ color: 'var(--color-text)' }} onClick={() => openRun(run)}>{run.question}</button>)}</section>}
        {scene === 'running' && <section style={{ marginTop: 60 }}><div className="section-top" style={{ marginBottom: 18 }}><span className="section-title">{path === 'agentic' ? 'Agentic research' : path === 'baseline-enhanced' ? 'Enhanced baseline' : 'Fast baseline'}</span><span className="muted micro">{path === 'agentic' ? 'No fixed pipeline — each action is chosen from what the last one returned' : path === 'baseline-enhanced' ? 'Iterative retrieval · batched extraction' : 'Fixed pipeline · no verification pass'}</span></div>{(path === 'agentic' ? runs.find(r => r.id === 'r-0412')?.trace || [] : liveTrace).map((step, i) => <div className="log-row" key={i}><span className="log-time muted">{step.atS}s</span><span>{step.action}</span><span className="muted micro">{step.returned}</span></div>)}<button className="text-button" style={{ marginTop: 24, fontSize: 13 }} onClick={() => { setScene('idle'); setQuestion(''); }}>Stop run</button>{path === 'agentic' && <button className="text-button" style={{ marginLeft: 24, fontSize: 13 }} onClick={() => openRun(runs.find(r => r.id === 'r-0413')!)}>Open fixture result</button>}</section>}
        {scene === 'readings' && active && <section style={{ marginTop: 60 }}><p style={{ maxWidth: '56ch' }}>Stopped after {active.durationS} s: no definition of <em>successful</em> exists in the corpus, and three independent plans disagreed on what to substitute (0.42).</p><p className="muted" style={{ fontSize: 13.5, marginBottom: 34 }}>Four readings are supported by the indexed papers. Each binds a different subschema and produces a different query.</p>{active.readings.map((r, i) => <button key={i} className="reading-row" onClick={() => { setActiveId('r-0412'); setPick(i); setScene('agentic_result'); }}><span className="muted micro">{i + 1}</span><span style={{ display: 'grid', gap: 4 }}><strong style={{ fontSize: 16.5 }}>{r.label}</strong><span className="muted" style={{ fontSize: 13.5 }}>{r.gloss}</span><span style={{ color: 'var(--color-accent-700)', fontSize: 12 }}>{r.classes.join(' · ')} &nbsp; | &nbsp; {r.properties.join(' · ')}</span></span></button>)}<button className="text-button" style={{ marginTop: 22, fontSize: 13 }} onClick={() => { setScene('idle'); setQuestion(''); }}>None of these — rephrase the question</button></section>}
        {['agentic_result', 'baseline_result', 'failed'].includes(scene) && active && <Result key={`${active.id}-${pick}`} run={active} pick={pick} setPick={setPick} onDeeper={() => { setActiveId('r-0413'); setScene('readings'); }} />}
      </>}
    </div>{showSidebar && active && <><Sidebar run={active} reading={reading} onRate={changeRating} onPromote={promote} />{promotion && <span className="muted micro">{promotion}</span>}</>}</main></div>;
}
