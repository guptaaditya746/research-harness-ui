// Text the screens show: metric pairs, Turtle of a reading's triples and ontology, outcomes, dates.
import type { Reading, Run, SchemaTerm } from './generated/run';

export function metric(pair?: [number, number] | null) {
  if (!pair) return '—';
  return `${pair[1] ? (pair[0] / pair[1]).toFixed(2) : '—'}  (${pair[0]} / ${pair[1]})`;
}

export const PREFIXES = ['@prefix hrn: <https://harness.local/schema#> .', '@prefix dblp: <https://dblp.org/rdf/schema#> .',
  '@prefix owl: <http://www.w3.org/2002/07/owl#> .', '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
  '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .'];
// Harness terms declared equivalent to DBLP properties are written with their DBLP name (as in triples.ttl).
const PREDICATE_ALIAS: Record<string, string> = { author: 'dblp:authoredBy', authoredBy: 'dblp:authoredBy', title: 'dblp:title',
  year: 'dblp:yearOfPublication', venue: 'dblp:publishedIn', publishedInStream: 'dblp:publishedInStream' };
export const OWL_KIND: Record<SchemaTerm['kind'], string> = { class: 'owl:Class', object: 'owl:ObjectProperty', datatype: 'owl:DatatypeProperty' };

export function predicateName(reading: Reading, p: string) {
  return PREDICATE_ALIAS[p] ?? reading.schemaTerms.find(term => term.kind !== 'class' && term.name.endsWith(`:${p}`))?.name ?? `hrn:${p}`;
}

export function turtle(reading: Reading) {
  const node = (value: string) => /^https?:\/\//.test(value) ? `<${value}>` : `<https://harness.local/resource/${encodeURIComponent(value)}>`;
  const literalValued = (p: string) => ['title', 'year', 'venue'].includes(p)
    || reading.schemaTerms.find(term => term.name === predicateName(reading, p))?.kind === 'datatype';
  return [...PREFIXES, '',
    ...reading.triples.map(t => `${node(t.s)} ${predicateName(reading, t.p)} ${literalValued(t.p) ? JSON.stringify(t.o) : node(t.o)} .`)].join('\n');
}

export function ontology(reading: Reading) {
  if (!reading.schemaTerms.length) {
    // Runs from before Phase 4 carry no term list; show what they have.
    return ['@prefix hrn: <https://harness.local/schema#> .', '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .', '',
      ...reading.classes.map(c => `hrn:${c} a rdfs:Class .`), '',
      ...reading.axioms.map(([p, d, r]) => `hrn:${p} rdfs:domain hrn:${d} ;\n  rdfs:range ${r.includes(':') ? r : `hrn:${r}`} .`)].join('\n');
  }
  return [...PREFIXES, '', ...reading.schemaTerms.flatMap(term => {
    const parts = [`${term.name} a ${OWL_KIND[term.kind]}`];
    if (term.parent) parts.push(`${term.kind === 'class' ? 'rdfs:subClassOf' : 'rdfs:subPropertyOf'} ${term.parent}`);
    if (term.domain) parts.push(`rdfs:domain ${term.domain}`);
    if (term.range) parts.push(`rdfs:range ${term.range}`);
    if (term.definition) parts.push(`rdfs:comment ${JSON.stringify(term.definition)}`);
    return [`# ${term.origin}`, `${parts.join(' ;\n  ')} .`, ''];
  })].join('\n');
}

/** The outcome in words, taken from the stage checks: which stage ended the run and why. */
export function outcome(run: Run): string | null {
  const failed = run.checks.find(check => check.status === 'fail');
  if (!failed) return null;
  if (failed.name === 'retrieval') return `Corpus search found no usable passages: ${failed.detail}`;
  if (failed.name === 'extraction pass' || failed.name === 'extraction') return `Passages were found, but extraction produced no supported triples: ${failed.detail}`;
  if (failed.name.startsWith('agent')) return `Agent validation failed (${failed.name}): ${failed.detail}`;
  return `Stopped at ${failed.name}: ${failed.detail}`;
}

export const when = (value: string) => new Date(value).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
export const pathName = (value: Run['path']) => value === 'agentic' ? 'Agentic' : value === 'baseline-enhanced' ? 'Adaptive' : value === 'external' ? 'External' : 'Fast';
