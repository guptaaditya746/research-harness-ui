import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import fs from 'fs';
cytoscape.use(fcose);
const run = JSON.parse(fs.readFileSync('fixtures/r-0412.json'));
const g = run.readings[0].graph;
const cy = cytoscape({ headless: true, styleEnabled: true, elements: [
  ...g.nodes.map(n => ({ data: { id: n.id, label: n.label, kind: n.kind } })),
  ...g.edges.map(e => ({ data: { id: `edge:${e.id}`, source: e.source, target: e.target, label: e.label, kind: e.kind } })),
]});
console.log('nodes', cy.nodes().length, 'edges', cy.edges().length);
cy.layout({ name: 'fcose', animate: false }).run();
cy.nodes().forEach(n => console.log(n.id(), JSON.stringify(n.position())));
