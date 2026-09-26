import { type ReactNode, useState } from 'react';
import type { Reading } from './generated/run';

const CITATION = /\[([^\]]+)\]/g;

/** The paper a passage or fact id belongs to: the longest known paper id it starts with. */
function paperOf(source: string, labels: Record<string, string>) {
  const key = Object.keys(labels).filter(id => source.startsWith(id)).sort((a, b) => b.length - a.length)[0];
  return key ? labels[key] : '';
}

/**
 * The answer as short prose with numbered references (Phase 6.1): each [source] becomes a number,
 * and the references below give the paper and the evidence the triples quote. With per-sentence
 * verdicts (verified answers) the sentences the verifier removed are listed with the reason.
 */
export function Answer({ reading, labels }: { reading: Reading; labels: Record<string, string> }) {
  const [showRemoved, setShowRemoved] = useState(false);
  const sentences = reading.answerSentences ?? [];
  const kept = sentences.filter(item => item.verdict !== 'removed');
  const removed = sentences.filter(item => item.verdict === 'removed');
  const text = kept.length ? kept.map(item => item.text).join(' ') : reading.answer;
  const known = new Set(reading.triples.map(triple => triple.source));
  const numbers = new Map<string, number>();
  const parts: (string | number[])[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const ids = match[1].split(',').map(id => id.trim()).filter(Boolean);
    if (!ids.length || !ids.every(id => known.has(id) || /[-_]/.test(id))) continue;
    parts.push(text.slice(last, match.index).trimEnd());
    parts.push(ids.map(id => { if (!numbers.has(id)) numbers.set(id, numbers.size + 1); return numbers.get(id)!; }));
    last = (match.index ?? 0) + match[0].length;
  }
  parts.push(text.slice(last));
  const evidence = (id: string) => reading.triples.find(triple => triple.source === id)?.evidence ?? '';
  const fromFacts = sentences.some(item => item.verdict === 'from_facts');
  return <section className="section" aria-label="Answer">
    <div className="section-top"><span className="section-title">Answer</span>
      <span className="muted micro">{sentences.length ? fromFacts ? 'Stated from the extracted facts (no composed sentence was supported)' : `${kept.length} sentence${kept.length === 1 ? '' : 's'} checked against the triples` : 'Composed from the extracted triples'}</span></div>
    <p className="answer">{parts.map((part, index) => typeof part === 'string' ? <span key={index}>{part.replace(/\s+([.,;:])/g, '$1')}</span>
      : <sup key={index} className="ref">{part.map(n => <a key={n} href={`#ref-${n}`}>{n}</a>).reduce<ReactNode[]>((all, node, i) => i ? [...all, ',', node] : [node], [])}</sup>)}</p>
    {numbers.size > 0 && <ol className="references">{[...numbers.entries()].map(([id, n]) => <li key={id} id={`ref-${n}`}>
      {paperOf(id, labels) ? <><span className="ref-source">{paperOf(id, labels)}</span> <code className="muted micro">{id}</code></> : <span className="ref-source">{id}</span>}
      {evidence(id) && <div className="muted micro">“{evidence(id).slice(0, 220)}{evidence(id).length > 220 ? '…' : ''}”</div>}
    </li>)}</ol>}
    {removed.length > 0 && <div className="removed">
      <button type="button" className="text-button micro" aria-expanded={showRemoved} onClick={() => setShowRemoved(value => !value)}>
        {showRemoved ? 'Hide' : 'Show'} {removed.length} sentence{removed.length === 1 ? '' : 's'} the verifier removed</button>
      {showRemoved && <ul className="removed-list">{removed.map((item, index) => <li key={index}><s>{item.text.replace(CITATION, '').trim()}</s>{item.reason && <span className="muted micro"> · {item.reason}</span>}</li>)}</ul>}
    </div>}
  </section>;
}
