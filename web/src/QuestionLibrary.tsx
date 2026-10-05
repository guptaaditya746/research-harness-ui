import type { LibraryQuestion, RunPath } from './types';

const categoryOrder = ['corpus_wide', 'ambiguous', 'metadata', 'citation', 'aggregation', 'synthesis', 'ambiguity', 'gold', 'recent'];
const categoryLabel: Record<string, string> = { corpus_wide: 'Corpus-wide', ambiguous: 'Ambiguous — agent interprets', metadata: 'Metadata', citation: 'Citations', aggregation: 'Aggregation', synthesis: 'Synthesis over full text', ambiguity: 'Ambiguous — agent interprets', gold: 'Gold candidates', recent: 'Recent runs' };

export function QuestionLibrary({ items, filter, setFilter, onPick, onRun, onCompare, autoFocus = true }: { items: LibraryQuestion[]; filter: string; setFilter: (value: string) => void; onPick: (text: string) => void; onRun: (kind: RunPath, text: string) => void; onCompare: (text: string) => void; autoFocus?: boolean }) {
  const needle = filter.trim().toLowerCase();
  const visible = items.filter(item => !needle || item.question.toLowerCase().includes(needle) || (item.type || '').toLowerCase().includes(needle) || (item.theme || '').toLowerCase().includes(needle));
  const groups = [...new Set(visible.map(item => item.category))].sort((a, b) => (categoryOrder.indexOf(a) + 1 || 99) - (categoryOrder.indexOf(b) + 1 || 99));
  return <section id="question-library" className="library" aria-label="Question library">
    <input className="input library-filter" autoFocus={autoFocus} value={filter} onChange={event => setFilter(event.target.value)} placeholder="Filter questions" aria-label="Filter questions" />
    {visible.length === 0 && <p className="muted micro">No questions match.</p>}
    {groups.map(group => <div key={group} className="library-group"><div className="section-title">{categoryLabel[group] || group}</div>
      {visible.filter(item => item.category === group).map(item => <div key={`${item.source}-${item.id ?? item.question}`} className="library-row">
        <button className="text-button library-text" title="Put this question in the box to edit it" onClick={() => onPick(item.question)}>{item.question}</button>
        <span className="library-meta muted micro">{item.theme || item.type?.replaceAll('_', ' ')}</span>
        <span className="library-run" aria-label={`Run: ${item.question}`}>
          <button className="text-button micro" onClick={() => onRun('baseline', item.question)}>Fast</button>
          <button className="text-button micro" onClick={() => onRun('baseline-enhanced', item.question)}>Adaptive</button>
          <button className="text-button micro" onClick={() => onRun('agentic', item.question)}>Agentic</button>
          <button className="text-button micro" onClick={() => onCompare(item.question)}>Compare</button>
        </span>
      </div>)}</div>)}
  </section>;
}
