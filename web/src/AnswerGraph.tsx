import cytoscape from 'cytoscape';
import type { Core, ElementDefinition } from 'cytoscape';
import fcose from 'cytoscape-fcose';
import layoutUtilities from 'cytoscape-layout-utilities';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { AnswerGraph, GraphEdge, Reading } from './generated/run';

cytoscape.use(layoutUtilities);
cytoscape.use(fcose);

type Selection =
  | { type: 'edge'; edge: GraphEdge }
  | { type: 'node'; id: string; label: string; kind: string; computed?: { operation: string; derivedFrom: string[] } };

/** Older runs have no server graph: connect their triples client-side, one node per entity. */
export function graphFromTriples(reading: Reading): AnswerGraph {
  const nodes = new Map<string, AnswerGraph['nodes'][number]>();
  const node = (value: string, kind: AnswerGraph['nodes'][number]['kind']) => {
    const id = `${kind}:${value}`;
    if (!nodes.has(id)) nodes.set(id, { id, label: value, kind, iri: null });
    return id;
  };
  const edges = reading.triples.map((t, index) => {
    const predicate = t.p.toLowerCase();
    const objectKind = predicate === 'authoredby' || predicate === 'author' ? 'author'
      : ['year', 'venue', 'title'].includes(predicate) ? 'value' : 'entity';
    return {
      id: t.tripleId || `t-${index}`, source: node(t.s, 'entity'), target: node(t.o, objectKind),
      label: t.p, kind: 'fact' as const, sourceIds: [t.source], evidence: t.evidence, origin: null,
    };
  });
  return { nodes: [...nodes.values()], edges, computed: [] };
}

/** Where a fact edge's predicate comes from (Phase 4): colour, legend text and the edge detail line. */
export const ORIGIN_STYLE = {
  dblp: { color: '#006786', label: 'DBLP ontology' },
  core: { color: '#7d7979', label: 'harness core vocabulary' },
  invented: { color: '#c79a00', label: 'invented for this question' },
} as const;

function token(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function AnswerGraphView({ reading, labels }: { reading: Reading; labels: Record<string, string> }) {
  const graph = useMemo(() => reading.graph ?? graphFromTriples(reading), [reading]);
  const container = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);
  const computedById = useMemo(() => new Map(graph.computed.map(item => [item.id, item])), [graph]);

  useEffect(() => {
    if (!container.current) return;
    const accent = token('--color-accent', '#0088b0');
    const accent2 = token('--color-accent-2', '#d6006c');
    const text = token('--color-text', '#201e1d');
    const muted = token('--color-neutral-400', '#bab6b6');
    const labelFor = (value: string, fallback: string) => labels[value] || fallback;
    const elements: ElementDefinition[] = [
      ...graph.nodes.map(item => {
        const label = labelFor(item.iri || item.label, item.label);
        // Computed nodes are sized to their text; cytoscape's `width: label` breaks edge endpoints.
        return { data: { id: item.id, label, kind: item.kind, boxWidth: Math.max(72, Math.min(260, label.length * 7 + 18)) } };
      }),
      ...graph.edges.map(item => ({
        data: { id: `edge:${item.id}`, source: item.source, target: item.target, label: item.label, kind: item.kind,
          origin: item.origin ?? 'none' },
      })),
    ];
    const instance = cytoscape({
      container: container.current,
      elements,
      wheelSensitivity: 0.2,
      style: [
        { selector: 'node', style: {
          label: 'data(label)', 'font-size': 11, 'text-wrap': 'wrap', 'text-max-width': '150px',
          'text-valign': 'bottom', 'text-margin-y': 6, color: text, 'background-color': accent,
          width: 18, height: 18,
        } },
        { selector: 'node[kind = "paper"]', style: { shape: 'round-rectangle', width: 26, height: 18 } },
        { selector: 'node[kind = "author"]', style: { shape: 'ellipse', 'background-color': accent2 } },
        { selector: 'node[kind = "value"], node[kind = "entity"]', style: { shape: 'diamond', 'background-color': muted } },
        { selector: 'node[kind = "claim"]', style: { shape: 'hexagon', 'background-color': accent } },
        { selector: 'node[kind = "passage"]', style: { shape: 'tag', 'background-color': muted, width: 14, height: 14 } },
        // Computed conclusions look different from source facts on purpose.
        { selector: 'node[kind = "computed"]', style: {
          shape: 'round-rectangle', 'background-color': token('--color-bg', '#f3f2f2'), 'border-width': 2,
          'border-style': 'dashed', 'border-color': accent2, width: 'data(boxWidth)', height: 28,
          'text-valign': 'center', 'text-margin-y': 0, 'font-weight': 'bold', 'text-max-width': '250px',
        } },
        { selector: 'edge', style: {
          label: 'data(label)', 'font-size': 9, color: text, 'curve-style': 'bezier', width: 1.4,
          'line-color': muted, 'target-arrow-color': muted, 'target-arrow-shape': 'triangle',
          'text-rotation': 'autorotate', 'text-background-color': token('--color-bg', '#f3f2f2'),
          'text-background-opacity': 1, 'text-background-padding': '2px',
        } },
        ...(Object.entries(ORIGIN_STYLE).map(([origin, style]) => ({
          selector: `edge[origin = "${origin}"]`,
          style: { 'line-color': style.color, 'target-arrow-color': style.color, width: 2 },
        }))),
        { selector: 'edge[kind = "evidence"]', style: { 'line-style': 'dotted' } },
        { selector: 'edge[kind = "derived"]', style: { 'line-style': 'dashed', 'line-color': accent2, 'target-arrow-color': accent2 } },
        { selector: '.highlight', style: { 'line-color': accent2, 'target-arrow-color': accent2, width: 3 } },
        { selector: ':selected', style: { 'border-width': 3, 'border-color': text } },
      ],
      layout: {
        name: 'fcose', animate: false, quality: 'proof', nodeRepulsion: 6000, idealEdgeLength: 90, padding: 24,
        // Disconnected parts of the answer (two ranked authors, say) are packed together instead of spread apart.
        packComponents: true, componentSpacing: 40, nodeDimensionsIncludeLabels: true,
      } as cytoscape.LayoutOptions,
    });
    instance.on('tap', 'edge', event => {
      const id = String(event.target.id()).replace(/^edge:/, '');
      const edge = graph.edges.find(item => item.id === id);
      instance.elements().removeClass('highlight');
      if (edge) setSelected({ type: 'edge', edge });
    });
    instance.on('tap', 'node', event => {
      const data = event.target.data();
      const computed = computedById.get(String(data.id));
      instance.elements().removeClass('highlight');
      if (computed) computed.derivedFrom.forEach(id => instance.$id(`edge:${id}`).addClass('highlight'));
      setSelected({ type: 'node', id: String(data.id), label: String(data.label), kind: String(data.kind),
        computed: computed ? { operation: computed.operation, derivedFrom: computed.derivedFrom } : undefined });
    });
    instance.on('tap', event => { if (event.target === instance) { instance.elements().removeClass('highlight'); setSelected(null); } });
    cy.current = instance;
    // Exposed for tests: the canvas cannot be inspected through the DOM.
    (container.current as HTMLDivElement & { __cy?: Core }).__cy = instance;
    return () => { instance.destroy(); cy.current = null; };
  }, [graph, labels, computedById]);

  const counts = `${graph.nodes.length} nodes · ${graph.edges.length} edges` + (graph.computed.length ? ` · ${graph.computed.length} computed` : '');
  return <div className="answer-graph">
    <div className="muted micro" style={{ display: 'flex', gap: 18, flexWrap: 'wrap', marginBottom: 8 }}>
      <span>{counts}</span>
      <span>solid: source fact · dotted: evidence passage · <span style={{ color: 'var(--color-accent-2)' }}>dashed: computed by the harness</span></span>
      {graph.edges.some(edge => edge.origin) && <span aria-label="Edge colours">{Object.values(ORIGIN_STYLE).map(style => <span key={style.label} style={{ marginRight: 10 }}><span style={{ display: 'inline-block', width: 14, height: 3, background: style.color, verticalAlign: 'middle', marginRight: 4 }} />{style.label}</span>)}</span>}
      <button className="text-button micro" onClick={() => cy.current?.fit(undefined, 24)}>Fit</button>
    </div>
    <div ref={container} className="answer-graph-canvas" role="img" aria-label="Answer graph" />
    <div className="answer-graph-detail" aria-live="polite">
      {!selected && <span className="muted micro">Click an edge to see its source, or a computed result to see what it was computed from.</span>}
      {selected?.type === 'edge' && <div>
        <div><strong>{selected.edge.label}</strong> <span className="muted micro">{selected.edge.kind === 'derived' ? 'computed step' : selected.edge.kind}{selected.edge.origin ? ` · ${ORIGIN_STYLE[selected.edge.origin].label}` : ''}</span></div>
        {selected.edge.sourceIds.length > 0 && <div className="micro">Source: {selected.edge.sourceIds.join(', ')}</div>}
        {selected.edge.evidence && <p style={{ fontStyle: 'italic', margin: '6px 0 0' }}>{selected.edge.evidence}</p>}
      </div>}
      {selected?.type === 'node' && <div>
        <div><strong>{selected.label}</strong> <span className="muted micro">{selected.kind}</span></div>
        {selected.computed && <div className="micro">Computed: {selected.computed.operation} · from {selected.computed.derivedFrom.length} source edges (highlighted)</div>}
      </div>}
    </div>
  </div>;
}
