import { useEffect, useMemo, useRef, useState } from 'react';
import type { Run } from './generated/run';
import { Mascot, type MascotState } from './Mascot';
import { ModelStatus } from './ModelStatus';
import { api, getJson } from './api';
import { QuestionLibrary } from './QuestionLibrary';
import { Result, Sidebar } from './Result';
import { BenchmarkScreen, CorpusScreen, GoldScreen, ReadingsScreen, RunningScreen, RunsScreen, SettingsScreen } from './screens';
import type { BenchmarkQuestion, GoldCandidate, LibraryQuestion, PaperLabel, RunPath, Scene } from './types';

// The run each fixture screen (?screen=...) opens; used by the screenshot tests.
const screenRun: Record<string, string> = { readings: 'r-0413', agentic_result: 'r-0412', baseline_result: 'r-0411', failed: 'r-0410' };

export default function App() {
  const initial = new URLSearchParams(location.search).get('screen') as Scene | null;
  const [scene, setScene] = useState<Scene>(initial || 'idle');
  const [runs, setRuns] = useState<Run[]>([]);
  const [question, setQuestion] = useState('');
  const [activeId, setActiveId] = useState(screenRun[initial || ''] || '');
  const [pick, setPick] = useState(0);
  const [path, setPath] = useState<RunPath>('agentic');
  const [externalOn, setExternalOn] = useState(false);
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
  // Exactly one set of live streams and status probes belongs to the screen.
  const streamDisposers = useRef<(() => void)[]>([]);
  const watched = useRef<string>('');
  const compareIds = useRef<string[]>([]);
  const closeStreams = () => { streamDisposers.current.forEach(dispose => dispose()); streamDisposers.current = []; };
  useEffect(() => closeStreams, []);
  const active = useMemo(() => runs.find(r => r.id === activeId), [runs, activeId]);
  const reading = active?.readings[pick] || active?.readings[0];

  const loadLibrary = () => getJson<{ questions: LibraryQuestion[] }>(`${api}/questions`).then(data => setLibrary(data.questions || [])).catch(() => undefined);
  useEffect(() => { loadLibrary(); }, []);
  useEffect(() => { if (!libraryOpen) return; const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setLibraryOpen(false); }; window.addEventListener('keydown', close); return () => window.removeEventListener('keydown', close); }, [libraryOpen]);
  const [runInput, setRunInput] = useState('corpus');
  const noteConfig = (config: { settings: { name: string; value: string }[] }) =>
    setRunInput(config.settings.find(row => row.name === 'HARNESS_RUN_INPUT')?.value || 'corpus');
  useEffect(() => { getJson<{ settings: { name: string; value: string }[] }>(`${api}/config`).then(noteConfig).catch(() => undefined); }, []);
  useEffect(() => { getJson<{ baseline: 'fast' | 'slow'; agentic: 'fast' | 'slow'; external?: 'on' | 'off' }>(`${api}/settings`).then(value => { setBaseMode(value.baseline); setAgMode(value.agentic); setExternalOn(value.external === 'on'); }).catch(() => undefined); }, []);
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
  const loadBenchmark = () => getJson<{ questions: BenchmarkQuestion[] }>(`${api}/benchmark?run_input=${runInput}`).then(data => setBenchmarkRows(data.questions || [])).catch(error => setNotice(`Benchmark could not be loaded: ${error.message}`));
  useEffect(() => { if (scene === 'benchmark') loadBenchmark(); }, [scene]);
  const runBenchmark = async (row: BenchmarkQuestion) => {
    try {
      const benchmarkPaths: RunPath[] = externalOn ? ['baseline', 'baseline-enhanced', 'agentic', 'external'] : ['baseline', 'baseline-enhanced', 'agentic'];
      const started = await Promise.all(benchmarkPaths.map(kind => getJson<{ ids: string[] }>(`${api}/benchmark/execute`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: row.question, path: kind, budget: kind === 'agentic' ? agMode : baseMode, repetitions: 3 }) })));
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
    let finished = false;
    let disposed = false;
    let probing = false;
    let unreachableSince = 0;
    let poller = 0;
    const dispose = () => { disposed = true; events.close(); window.clearInterval(poller); };
    streamDisposers.current.push(dispose);
    const complete = (run: Run) => {
      if (finished || disposed) return;
      finished = true; dispose(); onDone(run);
    };
    const lost = () => {
      if (finished || disposed) return;
      finished = true; dispose();
      if (watched.current === id || watched.current === '*') {
        setNotice(`Lost contact with run ${id}. Check the API, then reopen it from Runs.`);
        setScene('failed');
      }
    };
    const probe = async () => {
      if (finished || disposed || probing) return;
      probing = true;
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 4000);
      const response = await fetch(`${api}/runs/${id}`, { signal: controller.signal }).catch(() => null);
      window.clearTimeout(timeout);
      probing = false;
      if (finished || disposed) return;
      if (response?.ok) { complete(await response.json() as Run); return; }
      // A pending run is 404 until persisted. Proxy errors mean the API is
      // unavailable, so they must not reset the connection-loss timer.
      if (response?.status === 404) { unreachableSince = 0; return; }
      if (!unreachableSince) unreachableSince = Date.now();
      if (Date.now() - unreachableSince >= 30_000) lost();
    };
    poller = window.setInterval(() => { void probe(); }, 5000);
    const finish = async () => {
      if (finished || disposed) return;
      const run = await settle(id);
      if (run) complete(run); else lost();
    };
    events.addEventListener('trace', event => { if (!disposed && (watched.current === id || watched.current === '*')) { setLiveTrace(rows => [...rows, JSON.parse((event as MessageEvent).data)]); setLastRowAt(Date.now()); } });
    events.addEventListener('done', () => { void finish(); });
    events.onerror = () => { void probe(); };
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
  const resultScene = ['agentic_result', 'baseline_result'].includes(scene);
  const mascotState: MascotState = busy ? 'researching'
    : scene === 'failed' || (resultScene && active?.status === 'incomplete') ? 'failed'
    : resultScene && active?.status === 'answered' ? 'answered' : 'idle';
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

  const askScene = !['corpus', 'settings', 'runs', 'gold', 'benchmark'].includes(scene);
  const recent = runs.filter(r => r.status === 'answered' || r.status === 'corpus_absent' || r.status === 'no_evidence_found' || r.status === 'incomplete')
    .filter((run, index, all) => all.findIndex(other => other.question === run.question) === index).slice(0, 2);
  const noQuestion = busy || !question.trim();

  return <div className="page">
    <header className="nav header">
      <span className="nav-mark"><Mascot state={mascotState} /><span className="nav-brand">Research Harness</span></span>
      {['Ask', 'Corpus', 'Runs', 'Benchmark', 'Gold', 'Settings'].map(item => <a key={item} href="#" aria-current={currentNav === item ? 'page' : undefined}
        onClick={e => { e.preventDefault(); setScene(item === 'Ask' ? 'idle' : item.toLowerCase() as Scene); }}>{item}</a>)}
      <ModelStatus api={api} />
    </header>
    <main className="main"><div className="content">
      {scene === 'corpus' && <CorpusScreen papers={papers} />}
      {scene === 'settings' && <SettingsScreen baseMode={baseMode} agMode={agMode} save={saveSettings} onConfig={noteConfig} />}
      {scene === 'runs' && <RunsScreen runs={runs} openArtifact={openArtifact} setOpenArtifact={setOpenArtifact} artifactBody={artifactBody} openRun={openRun} />}
      {scene === 'gold' && <GoldScreen candidates={candidates} drafts={drafts} errors={goldErrors} save={saveGold}
        setDraft={(key, value) => setDrafts(previous => ({ ...previous, [key]: value }))} />}
      {scene === 'benchmark' && <BenchmarkScreen rows={benchmarkRows} externalOn={externalOn} refresh={loadBenchmark} run={runBenchmark} runInput={runInput} />}
      {notice && <p className="notice" role="alert">{notice}</p>}
      {askScene && <>
        <div className="question-wrap"><textarea className="input question" rows={Math.max(1, Math.ceil(question.length / 38))} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about the corpus" aria-label="Question" /></div>
        <div className="actions">
          <button className="btn btn-secondary" disabled={noQuestion} onClick={() => runFixture('baseline')}>Fast</button>
          <button className="btn btn-secondary" disabled={noQuestion} onClick={() => runFixture('baseline-enhanced')}>Adaptive</button>
          <button className="btn btn-primary" disabled={noQuestion} onClick={() => runFixture('agentic')}>Agentic</button>
          <button className="btn btn-secondary" disabled={noQuestion || !externalOn} onClick={() => runFixture('external')}
            title={externalOn ? 'An off-the-shelf agent (mini-SWE-agent) with bash commands over the same corpus and ontology, none of our pipeline' : 'The External path is off: set HARNESS_EXTERNAL_COMMAND and HARNESS_EXTERNAL_CONFIG'}>External</button>
          <button className="btn btn-secondary" disabled={noQuestion} onClick={() => runCompare()}>Compare all</button>
          <button className="text-button micro" aria-expanded={libraryOpen} aria-controls="question-library" disabled={busy} onClick={() => setLibraryOpen(open => !open)}>{libraryOpen ? 'Close question library' : `Question library (${library.length})`}</button>
          <button className="text-button muted micro" onClick={() => setScene('settings')}>Input: {runInput === 'evidence_doc' ? 'evidence document D' : 'corpus'} · budgets: baseline {baseMode}, agentic {agMode} · change</button>
        </div>
        {libraryOpen && !busy && <QuestionLibrary items={library} filter={libraryFilter} setFilter={setLibraryFilter}
          onPick={text => { setQuestion(text); setLibraryOpen(false); }} onRun={(kind, text) => runFixture(kind, text)} onCompare={text => runCompare(text)} />}
        {scene === 'idle' && !libraryOpen && <section className="starter-section" aria-label="Evaluation questions">
          <div className="starter-heading">
            <span className="section-title">Evaluation questions</span>
            <span className="muted micro">{library.length ? `The ${library.length} questions of the evaluation set. Run one directly, or pick it to edit.` : 'No evaluation questions loaded: check HARNESS_QUESTION_SETS and that the API is running. You can still type a question above.'}</span>
          </div>
          {library.length > 0 && <QuestionLibrary items={library} filter={libraryFilter} setFilter={setLibraryFilter}
            onPick={text => setQuestion(text)} onRun={(kind, text) => runFixture(kind, text)} onCompare={text => runCompare(text)} autoFocus={false} />}
        </section>}
        {scene === 'idle' && <section style={{ marginTop: 72, display: 'grid', gap: 16 }}>
          <span className="section-title">Recent</span>
          {recent.map(run => <button key={run.id} className="text-button" style={{ color: 'var(--color-text)' }} onClick={() => openRun(run)}>{run.question}</button>)}
        </section>}
        {scene === 'running' && <RunningScreen path={path} trace={liveTrace} waitingOn={waitingOn} waitedS={waitedS} phase={phase} onStop={stopRun} />}
        {scene === 'readings' && active && <ReadingsScreen run={active} picking={picking} choose={chooseReading} onNone={() => { setScene('idle'); }} />}
        {scene === 'failed' && !active && !notice && <p className="notice">The run failed before it produced a result.</p>}
        {['agentic_result', 'baseline_result', 'failed'].includes(scene) && active && <Result key={`${active.id}-${pick}`} run={active} pick={pick} setPick={setPick} labels={labels} onDeeper={() => runFixture('agentic')} />}
      </>}
    </div>{showSidebar && active && <><Sidebar run={active} reading={reading} onRate={changeRating} onPromote={promote} />{promotion && <span className="muted micro">{promotion}</span>}</>}</main>
  </div>;
}
