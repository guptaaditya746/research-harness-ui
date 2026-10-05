// Shapes of API responses and UI state shared by the screens (runs themselves: ./generated/run).
import type { ClaimSubgraph, Triple } from './generated/run';

export type Scene = 'idle' | 'running' | 'readings' | 'agentic_result' | 'baseline_result' | 'failed' | 'runs' | 'corpus' | 'settings' | 'gold' | 'benchmark';
export type TripleView = 'statements' | 'graph' | 'turtle';
export type GoldCandidate = { id: string; run_id: string; question: string; gold_triples: Triple[]; gold_claims?: ClaimSubgraph[]; notes: string; status: string };
export type PaperLabel = { paper_id: string; dblp_iri?: string | null; title: string };
export type LibraryQuestion = {
  question: string; id?: string | null; category: string; type?: string | null; theme?: string | null; expected?: string | null; source: string;
};
export type RunPath = 'agentic' | 'baseline' | 'baseline-enhanced' | 'external';
export type Budget = 'fast' | 'slow';
export type Aggregate = { mean: number | null; min: number | null; max: number | null };
export type BenchmarkPath = {
  path: string; completed: number; failed: number; precision: Aggregate; recall: Aggregate; f1: Aggregate;
  claim_precision: Aggregate; claim_recall: Aggregate; claim_f1: Aggregate;
  model_calls?: number; tool_calls?: number; wall_clock_s?: number; repetitions: string[];
};
export type BenchmarkQuestion = {
  id: string; question: string; category: string; theme?: string; difficulty?: number | null; expected_status?: string;
  gold_version: string; paths: BenchmarkPath[];
};
