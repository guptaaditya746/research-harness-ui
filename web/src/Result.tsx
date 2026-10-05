// A run's result: answer, subschema, claims, answer graph, checks; the evaluation sidebar; live progress.
import { Suspense, lazy, useState } from 'react';
import type { Check, Reading, Run, SchemaTerm, Triple } from './generated/run';
import { Answer } from './Answer';
import { RunTimeline } from './RunTimeline';
import { metric, ontology, OWL_KIND, outcome, turtle } from './format';
import type { TripleView } from './types';

// The graph library is large; load it only when a graph is shown.
const AnswerGraphView = lazy(() => import('./AnswerGraph').then(module => ({ default: module.AnswerGraphView })));

const ORIGIN_BADGE: Record<SchemaTerm['origin'], { text: string; background: string; color: string; title: string }> = {
  dblp: { text: 'DBLP', background: '#cbeeff', color: '#004961', title: 'From the DBLP ontology' },
  core: { text: 'core', background: '#eae7e7', color: '#444141', title: 'Harness core vocabulary' },
  invented: { text: 'invented', background: '#fcefc0', color: '#5c4600', title: 'Invented for this question' },
};

function OriginBadge({ origin }: { origin: SchemaTerm['origin'] }) {
  const style = ORIGIN_BADGE[origin];
  return <span title={style.title} style={{ background: style.background, color: style.color, borderRadius: 3, padding: '1px 6px', fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap' }}>{style.text}</span>;
}

function SchemaTerms({ terms }: { terms: SchemaTerm[] }) {
  const order: Record<SchemaTerm['origin'], number> = { dblp: 0, core: 1, invented: 2 };
  const sorted = [...terms].sort((a, b) => order[a.origin] - order[b.origin] || a.name.localeCompare(b.name));
  return <div style={{ display: 'grid', gap: 12 }}>
    <div className="muted micro" aria-label="Terms by origin">{(['dblp', 'core', 'invented'] as const).map(origin => <span key={origin} style={{ marginRight: 14 }}><OriginBadge origin={origin} /> {terms.filter(term => term.origin === origin).length}</span>)}</div>
    {sorted.map(term => <div key={term.iri}>
      <div style={{ fontSize: 14.5, display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}><OriginBadge origin={term.origin} /><strong>{term.name}</strong><span className="muted micro">{OWL_KIND[term.kind]}</span></div>
      <div className="muted" style={{ font: '11.5px ui-monospace, Menlo, monospace' }}>{[term.parent && `${term.kind === 'class' ? 'subClassOf' : 'subPropertyOf'} ${term.parent}`, term.domain && `domain ${term.domain}`, term.range && `range ${term.range}`].filter(Boolean).join('  ·  ')}</div>
      {term.definition && <div className="muted micro">{term.definition}</div>}
    </div>)}
  </div>;
}

export function Sidebar({ run, reading, onRate, onPromote }: { run: Run; reading?: Reading; onRate: (rating: string) => void; onPromote: () => void }) {
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
export function Progress({ rows, waitingOn, waitedS, phase }: { rows: TraceRows; waitingOn: string | null; waitedS: number; phase: string }) {
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


export function Result({ run, pick, setPick, onDeeper, labels }: { run: Run; pick: number; setPick: (index: number) => void; onDeeper: () => void; labels: Record<string, string> }) {
  const [showOntology, setShowOntology] = useState(false);
  const [showQuery, setShowQuery] = useState(false);
  const reading = run.readings[pick] || run.readings[0];
  const failed = run.status === 'incomplete';
  const absent = run.status === 'corpus_absent';
  const noEvidence = run.status === 'no_evidence_found';
  const pathLabel = run.path === 'baseline-enhanced' ? 'adaptive baseline' : run.path === 'baseline' ? 'fast baseline' : run.path === 'external' ? 'external harness' : 'agentic research';
  const reason = run.reason || '';
  const modelFailure = [...run.trace].reverse().find(row => row.action === 'Model call failed')?.returned;
  const failure = reason === 'provider_error'
    ? `A model call failed: ${modelFailure || 'the provider did not return a usable reply'}. The trace below the question shows every model call with its duration.`
    : reason === 'no_results'
      ? `The ${pathLabel} could not find enough supported evidence in the corpus to answer this question.`
      : reason === 'answer_unsupported'
        ? `The ${pathLabel} found evidence, but the answer it wrote was not supported by the extracted triples. The checks below say how many claims the verifier removed.`
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
    {!failed && !absent && !noEvidence && reading && <><Answer reading={reading} labels={labels} /><section className="section" aria-label="Subschema"><div className="section-top"><span className="section-title">Subschema</span><div className="section-links"><button className="text-button micro" onClick={() => setShowOntology(!showOntology)}>{showOntology ? 'Hide ontology' : 'Ontology (Turtle)'}</button><button className="text-button micro" onClick={() => setShowQuery(!showQuery)}>{showQuery ? 'Hide query' : 'SPARQL'}</button></div></div>
      <p className="muted" style={{ fontSize: 14, margin: '10px 0 16px' }}>{run.path === 'baseline' ? 'Unstated — inferred by the extractor. ' : ''}The answer is computed over {reading.classes.map(c => c.toLowerCase() + 's').join(', ')} only. Nothing outside these is read.</p>
      {reading.schemaTerms.length > 0 ? <SchemaTerms terms={reading.schemaTerms} /> : <div style={{ display: 'grid', gap: 12 }}>{reading.axioms.map(([property, domain, range]) => <div key={property}><div style={{ fontSize: 14.5 }}>A {domain.toLowerCase()} has {property.replaceAll('_', ' ')} relating it to {range.replace('xsd:', '')}.</div><div className="muted" style={{ font: '11.5px ui-monospace, Menlo, monospace' }}>{`hrn:${property}  rdfs:domain hrn:${domain}  rdfs:range ${range.includes(':') ? range : `hrn:${range}`}`}</div></div>)}</div>}
      {showOntology && <pre className="code">{ontology(reading)}</pre>}{showQuery && <pre className="code">{reading.sparql}</pre>}
    </section><Claims reading={reading} labels={labels} /><Triples key={pick} reading={reading} labels={labels} /></>}
    {failed && <section className="section" aria-label="Run result" style={{ marginTop: 34 }}><div className="section-top"><span className="section-title">Natural-language answer</span></div><p className="answer">{failure}</p></section>}
    <RunTimeline run={run} />
    <Checks checks={run.checks} />
  </div>;
}
