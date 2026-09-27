"""Strict / entailed / soft scoring, reproduced on the nlp27 worked example.

The paper facts are real (ACL Anthology 2025.acl-long.230 and 2025.emnlp-main.730); the DBLP record
ids and the run output are made up so every scoring case appears once. The expected numbers are the
ones in labelling/docs/EVALUATION.md.
"""

from __future__ import annotations

import json

import pytest
from harness_evaluation import canonical_triple_key, evaluate_gold_row, load_relations, score_levels
from harness_evaluation.judge import CachedJudge, parse_verdict
from harness_evaluation.relations import build_relations
from harness_evaluation.stats import mean_ci, paired_difference_ci
from harness_schema.models import Metrics, Reading, Run
from rdflib import OWL, RDFS, Graph, Namespace

SAFE = "https://dblp.org/rec/conf/acl/SafeRAG25"
HYDRA = "https://dblp.org/rec/conf/emnlp/HydraRAG25"
SAFE_TITLE = "SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model"
HYDRA_TITLE = "HydraRAG: Structured Cross-Source Enhanced Large Language Model Reasoning"


def t(s: str, p: str, o: str, **extra) -> dict:
    return {"s": s, "p": p, "o": o, "support": "test", "evidence": "e", "source": "x", **extra}


GOLD = [
    t(SAFE, "title", SAFE_TITLE),
    t(HYDRA, "title", HYDRA_TITLE),
    t(SAFE, "addressesProblem", "security of RAG", weight="vital",
      variants=["vulnerability of RAG to manipulated retrieved knowledge"]),
    t(SAFE, "contributionType", "Benchmark", weight="vital"),
    t(SAFE, "evaluatesAttack", "silver noise", weight="okay"),
    t(SAFE, "evaluatesAttack", "soft ad", weight="okay"),
    t(SAFE, "evaluatesAttack", "inter-context conflict", weight="okay"),
    t(SAFE, "evaluatesAttack", "white DoS", weight="okay"),
    t(HYDRA, "addressesProblem", "multi-hop, multi-entity reasoning across knowledge graphs and text",
      weight="vital"),
    t(HYDRA, "contributionType", "Method", weight="vital"),
    t(HYDRA, "addressesProblem", "verifying evidence across sources", weight="okay"),
]
RUN = [
    t(SAFE, "title", SAFE_TITLE),                                                      # S1 strict
    t(HYDRA, "title", HYDRA_TITLE),                                                    # S2 strict
    t(SAFE, "authoredBy", "Xun Liang"),                                                # S3 not scored
    t(SAFE, "evaluatesAttack", "soft ad"),                                             # S4 strict
    t(SAFE, "hasAttackTask", "silver noise"),                                          # S5 entailed
    t(SAFE, "addressesProblem", "vulnerability of RAG to manipulated retrieved knowledge"),  # S6 variant
    t(HYDRA, "addressesProblem", "reasoning across structured and unstructured sources"),    # S7 judge
    t(SAFE, "evaluatesComponents", "14 RAG components"),                               # S8 true, not in gold
    t(HYDRA, "addressesProblem", "defending RAG against prompt injection"),            # S9 wrong
]
REJECTED = [t(HYDRA, "addressesProblem", "defending RAG against prompt injection", verdict="wrong",
              relevance="answers")]


def key(item):
    return canonical_triple_key(item)


@pytest.fixture(scope="module")
def relations(tmp_path_factory):
    path = tmp_path_factory.mktemp("ontology") / "rules.ttl"
    path.write_text(
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix dblp: <https://dblp.org/rdf/schema#> .\n"
        "@prefix hrn: <https://harness.local/schema#> .\n"
        "hrn:hasAttackTask rdfs:subPropertyOf hrn:evaluatesAttack .\n"
        "hrn:firstAuthor rdfs:subPropertyOf dblp:authoredBy .\n"
        "hrn:wrote owl:inverseOf dblp:authoredBy .\n"
        "hrn:HybridRAGMethod rdfs:subClassOf hrn:Method .\n"
    )
    return load_relations(path)


def judge(gold, run):
    return gold["o"].startswith("multi-hop") and run["o"].startswith("reasoning across structured")


def test_nlp27_reproduces_the_documented_numbers(relations) -> None:
    result = score_levels(RUN, GOLD, key=key, relations=relations, judge=judge, rejected=REJECTED)
    levels = result["levels"]
    # Entailed: S5 (more specific than F3) recovers F3 but earns no precision credit.
    assert [(levels[name]["matched_predicted"], levels[name]["matched_gold"]) for name in levels] == [
        (3, 3), (3, 4), (6, 6)]
    assert {name: round(levels[name]["f1"], 2) for name in levels} == {
        "strict": 0.32, "entailed": 0.37, "soft": 0.63}
    assert levels["soft"]["predicted"] == 8 and levels["soft"]["gold"] == 11
    assert levels["strict"]["vital_recall"] == 0.0 and levels["soft"]["vital_recall"] == 0.5
    assert levels["soft"]["parts"]["dblp"]["f1"] == 1.0
    assert (levels["soft"]["parts"]["invented"]["matched_gold"], levels["soft"]["parts"]["invented"]["gold"]) == (4, 9)
    # S5 is more specific than F3: it recovers F3, but the gold does not vouch for the detail, so it is
    # unjudged rather than correct. S8 is unjudged (true, not in gold); S9 was rejected as wrong.
    assert result["judged_precision"] == {"value": 5 / 6, "correct": 5, "wrong": 1, "coverage": 6 / 8}
    assert result["excluded_dblp"] == [{"s": SAFE, "p": "authoredBy", "o": "Xun Liang"}]
    assert {(row["run"]["o"], row["level"], row["how"]) for row in result["matched"]} == {
        (SAFE_TITLE, "strict", "canonical key"), (HYDRA_TITLE, "strict", "canonical key"),
        ("soft ad", "strict", "canonical key"), ("silver noise", "entailed", "rule"),
        ("vulnerability of RAG to manipulated retrieved knowledge", "soft", "variant"),
        ("reasoning across structured and unstructured sources", "soft", "judge"),
    }
    assert sorted(row["o"] for row in result["missed"] if row["weight"] == "vital") == ["Benchmark", "Method"]
    assert {row["o"]: row["verdict"] for row in result["extra"]} == {
        "14 RAG components": "unjudged", "defending RAG against prompt injection": "wrong"}
    # The judge is only asked about same-subject pairs with no rule or variant edge.
    assert result["judge"] == {"asked": 2 * 3 + 3, "pending": 0, "matched": 1}


def test_levels_are_cumulative_and_soft_without_judge_uses_variants_only(relations) -> None:
    result = score_levels(RUN, GOLD, key=key, relations=relations)
    assert result["levels"]["soft"]["matched_gold"] == 5
    assert result["judge"]["asked"] == 0


def test_entailment_direction_follows_semantic_precision_and_recall(relations) -> None:
    general = [t(SAFE, "evaluatesAttack", "silver noise")]
    specific = [t(SAFE, "hasAttackTask", "silver noise")]
    # Run more general than gold: true (implied by the gold), but does not state the gold fact.
    run_general = score_levels(general, specific, key=key, relations=relations)["levels"]
    assert (run_general["entailed"]["matched_predicted"], run_general["entailed"]["matched_gold"]) == (1, 0)
    # Run more specific than gold: states the gold fact, but the gold does not vouch for the detail.
    run_specific = score_levels(specific, general, key=key, relations=relations)["levels"]
    assert (run_specific["entailed"]["matched_predicted"], run_specific["entailed"]["matched_gold"]) == (0, 1)
    # Soft is symmetric: both directions count as "the same fact".
    assert run_general["soft"]["f1"] == run_specific["soft"]["f1"] == 1.0


def test_one_to_one_credit() -> None:
    gold = [t(SAFE, "evaluatesAttack", "soft ad", variants=["soft advertising", "soft ads"])]
    run = [t(SAFE, "evaluatesAttack", "soft advertising"), t(SAFE, "evaluatesAttack", "soft ads")]
    soft = score_levels(run, gold, key=key)["levels"]["soft"]
    assert (soft["matched_predicted"], soft["matched_gold"], soft["predicted"]) == (1, 1, 2)


def test_inverse_class_and_same_as_rules() -> None:
    hrn = Namespace("https://harness.local/schema#")
    graph = Graph()
    graph.add((hrn.wrote, OWL.inverseOf, hrn.writtenBy))
    graph.add((hrn.HybridRAGMethod, RDFS.subClassOf, hrn.Method))
    graph.add((hrn.HydraRAG, OWL.sameAs, hrn["HydraRAG-framework"]))
    from harness_evaluation import _entity_key, _predicate_key

    rules = build_relations(graph, predicate_key=_predicate_key,
                            entity_key=lambda v: _entity_key(v, predicate="", position="object"))
    assert score_levels([t("ada", "wrote", "p9")], [t("p9", "writtenBy", "ada")], key=key,
                        relations=rules)["levels"]["entailed"]["matched_gold"] == 1
    assert score_levels([t("hydrarag", "type", "HybridRAGMethod")], [t("hydrarag", "rdf:type", "Method")],
                        key=key, relations=rules)["levels"]["entailed"]["matched_gold"] == 1
    assert score_levels([t("HydraRAG-framework", "type", "Method")], [t("HydraRAG", "type", "Method")],
                        key=key, relations=rules)["levels"]["entailed"]["matched_gold"] == 1


def test_default_rules_link_harness_and_dblp_predicates() -> None:
    rules = load_relations()
    run = [t("https://dblp.org/rec/conf/acl/X25", "https://dblp.org/rdf/schema#authoredBy", "Ada")]
    gold = [t("https://dblp.org/rec/conf/acl/X25", "author", "Ada")]
    assert score_levels(run, gold, key=key, relations=rules)["levels"]["strict"]["f1"] == 1.0


def test_cached_judge_calls_once_and_cache_only_mode_reports_pending(tmp_path) -> None:
    cache = tmp_path / "judge.jsonl"
    calls = []

    def complete(prompt):
        calls.append(prompt.ref)
        assert json.loads(prompt.user)["reference"]["object"] == "a"
        return {"same_fact": True, "reason": "same"}

    live = CachedJudge(cache, complete)
    assert live(t("s", "p", "a"), t("s", "p", "b")) is True
    assert live(t("s", "p", "a"), t("s", "p", "b")) is True
    assert len(calls) == 1 and calls[0].startswith("evaluation/triple_match@")
    offline = CachedJudge(cache)
    assert offline(t("s", "p", "a"), t("s", "p", "b")) is True
    assert offline(t("s", "p", "a"), t("s", "p", "c")) is None
    assert parse_verdict({"same_fact": "no"}) is False and parse_verdict({}) is None


def test_ambiguous_gold_uses_the_best_accepted_reading(relations) -> None:
    reading = Reading(label="one", gloss="g", classes=[], axioms=[], sparql="", answer="a",
                      triples=[t("p1", "authoredBy", "Ada")], metrics=Metrics(), status="answered")
    run = Run(id="r", startedAt="2026-01-01T00:00:00Z", durationS=1, question="Who is the best author?",
              path="agentic", budget="fast", status="answered", readings=[reading], chosenReading=0)
    row = {"gold_triples": [t("p1", "authoredBy", "Ada"), t("p2", "authoredBy", "Bo")],
           "accepted_readings": [
               {"label": "most citations", "measure": "citations", "gold_triples": [t("p2", "authoredBy", "Bo")]},
               {"label": "most papers", "measure": "papers", "gold_triples": [t("p1", "authoredBy", "Ada")]}]}
    result = evaluate_gold_row(run, row, relations=relations)
    assert result["accepted_reading"]["label"] == "most papers"
    assert result["chosen_levels"]["levels"]["strict"]["f1"] == 1.0


def test_confidence_intervals() -> None:
    assert mean_ci([]) == {"mean": None, "low": None, "high": None, "n": 0}
    assert mean_ci([0.5, 0.5]) == {"mean": 0.5, "low": 0.5, "high": 0.5, "n": 2}
    interval = mean_ci([0.1, 0.4, 0.6, 0.9, 0.5, 0.3])
    assert interval["low"] < interval["mean"] < interval["high"]
    assert paired_difference_ci([0.6, 0.7], [0.5, 0.5])["mean"] == pytest.approx(0.15)


def test_build_benchmark_groups_by_complexity_and_pairs_paths(relations) -> None:
    from harness_evaluation.benchmark import build_benchmark

    def run(run_id: str, path: str, question: str, triples: list[dict]) -> Run:
        reading = Reading(label="one", gloss="g", classes=[], axioms=[], sparql="", answer="a",
                          triples=triples, metrics=Metrics(), status="answered")
        return Run(id=run_id, startedAt="2026-01-01T00:00:00Z", durationS=1, question=question, path=path,
                   budget="fast", status="answered", readings=[reading], chosenReading=0)

    gold = [t(SAFE, "evaluatesAttack", "soft ad"), t(SAFE, "evaluatesAttack", "silver noise")]
    candidates = [
        {"id": "g-simple", "question": "Q1", "category": "synthesis", "gold_triples": gold[:1],
         "complexity": {"level": "simple", "points": 0, "features": {}}},
        {"id": "g-complex", "question": "Q2", "category": "synthesis", "gold_triples": gold,
         "complexity": {"level": "complex", "points": 3, "features": {}}},
    ]
    runs = [run("r1", "baseline", "Q1", gold[:1]), run("r2", "agentic", "Q1", gold[:1]),
            run("r3", "baseline", "Q2", gold[:1]), run("r4", "agentic", "Q2", gold)]
    result = build_benchmark(candidates, runs, relations=relations)
    rows = {row["complexity"]: row for row in result["by_complexity"]["rows"]}
    assert rows["simple"]["paths"]["agentic"]["soft"]["mean"] == 1.0
    assert rows["complex"]["paths"]["baseline"]["soft"]["mean"] == pytest.approx(2 / 3)
    assert rows["all"]["soft_f1_differences"]["agentic - baseline"]["mean"] == pytest.approx(1 / 6)
    assert rows["compound"]["questions"] == 0


PAPER = "https://dblp.org/rec/conf/acl/X25"


@pytest.mark.parametrize(("run", "precision", "recall"), [
    # the DBLP seed: authoredBy is a kind of createdBy (run more general: precision only)
    (t(PAPER, "https://dblp.org/rdf/schema#createdBy", "Ada Lovelace"), 1, 0),
    # the DBLP seed: authorOf is the inverse of authoredBy
    (t("Ada Lovelace", "https://dblp.org/rdf/schema#authorOf", PAPER), 1, 1),
    # an invented predicate linked by a rule (run more specific: recall only)
    (t(PAPER, "firstAuthor", "Ada Lovelace"), 0, 1),
    # an invented inverse
    (t("Ada Lovelace", "wrote", PAPER), 1, 1),
])
def test_rules_re_key_author_names_for_each_predicate(relations, run, precision, recall) -> None:
    gold = [t(PAPER, "authoredBy", "Ada Lovelace")]
    result = score_levels([run], gold, key=key, relations=relations)
    entailed = result["levels"]["entailed"]
    assert result["excluded_dblp"] == []
    assert (entailed["matched_predicted"], entailed["matched_gold"]) == (precision, recall)


def test_matching_prefers_vital_and_stricter_pairs_whatever_the_order(relations) -> None:
    exact = t("x", "p", "y", weight="vital")
    other = t("x", "p", "z", weight="okay", variants=["y"])
    for gold in ([exact, other], [other, exact]):
        result = score_levels([t("x", "p", "y")], gold, key=key, relations=relations)
        assert result["levels"]["strict"]["vital_recall"] == result["levels"]["soft"]["vital_recall"] == 1.0
        assert [(row["gold"]["o"], row["level"]) for row in result["matched"]] == [("y", "strict")]


def test_local_closure_follows_rules_from_any_gold_predicate(relations) -> None:
    gold = [t("ada", "wrote", PAPER)]
    result = score_levels([t(PAPER, "authoredBy", "ada")], gold, key=key, relations=relations)
    assert result["excluded_dblp"] == []
    assert result["levels"]["entailed"]["matched_gold"] == 1
    # an inverse of a child of the gold predicate still reaches it
    created = [t(PAPER, "https://dblp.org/rdf/schema#createdBy", "Ada")]
    inverse = score_levels([t("Ada", "https://dblp.org/rdf/schema#authorOf", PAPER)], created, key=key,
                           relations=relations)
    assert inverse["excluded_dblp"] == [] and inverse["levels"]["entailed"]["matched_gold"] == 1


def test_prefixed_class_names_use_the_class_rules(relations) -> None:
    result = score_levels([t("hydrarag", "rdf:type", "hrn:HybridRAGMethod")], [t("hydrarag", "type", "Method")],
                          key=key, relations=relations)
    assert result["levels"]["entailed"]["matched_gold"] == 1


def test_duplicate_gold_triples_merge_weight_and_variants(relations) -> None:
    gold = [t("x", "p", "y", variants=["why"]), t("x", "p", "y", weight="vital", variants=["wye"])]
    result = score_levels([t("x", "p", "wye")], gold, key=key, relations=relations)
    assert result["levels"]["soft"]["gold"] == 1 and result["levels"]["soft"]["vital_recall"] == 1.0


def test_a_failing_judge_leaves_pairs_pending(tmp_path) -> None:
    def broken(prompt):
        raise TimeoutError("model busy")

    judge = CachedJudge(tmp_path / "cache.jsonl", broken)
    (tmp_path / "cache.jsonl").write_text("not json\n")
    result = score_levels([t("s", "p", "a")], [t("s", "q", "b")], key=key, judge=judge)
    assert result["judge"] == {"asked": 1, "pending": 1, "matched": 0}
    assert CachedJudge(tmp_path / "cache.jsonl")(t("s", "q", "b"), t("s", "p", "a")) is None


def test_ambiguous_row_keeps_its_own_gold_when_that_scores_best(relations) -> None:
    reading = Reading(label="one", gloss="g", classes=[], axioms=[], sparql="", answer="a",
                      triples=[t("p1", "authoredBy", "Ada")], metrics=Metrics(), status="answered")
    run = Run(id="r", startedAt="2026-01-01T00:00:00Z", durationS=1, question="Who is the best author?",
              path="agentic", budget="fast", status="answered", readings=[reading], chosenReading=0)
    row = {"gold_triples": [t("p1", "authoredBy", "Ada")],
           "accepted_readings": [{"label": "other", "measure": "m", "gold_triples": [t("p2", "authoredBy", "Bo")]}]}
    result = evaluate_gold_row(run, row, relations=relations)
    assert result["accepted_reading"] is None
    assert result["chosen_levels"]["levels"]["strict"]["f1"] == result["chosen"]["f1"] == 1.0


def test_the_content_core_credits_cskg_inverses_and_evaluates_on() -> None:
    relations = load_relations()
    # CS-KG inverse: a run's "MethodZ uses ToolQ" states the gold's "ToolQ usedBy MethodZ".
    levels = score_levels([t("MethodZ", "uses", "ToolQ")], [t("ToolQ", "usedBy", "MethodZ")], key=key,
                          relations=relations)["levels"]
    assert levels["entailed"]["matched_gold"] == 1 and levels["strict"]["matched_gold"] == 0
    # evaluatesOn is a kind of usesDataset: the more specific run triple states the gold fact.
    levels = score_levels([t("p99", "evaluatesOn", "BenchY")], [t("p99", "usesDataset", "BenchY")], key=key,
                          relations=relations)["levels"]
    assert (levels["entailed"]["matched_gold"], levels["entailed"]["matched_predicted"]) == (1, 0)


def test_without_the_content_core_those_rules_are_absent(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_CONTENT_CORE", "none")
    levels = score_levels([t("MethodZ", "uses", "ToolQ")], [t("ToolQ", "usedBy", "MethodZ")], key=key,
                          relations=load_relations())["levels"]
    assert levels["entailed"]["matched_gold"] == 0
