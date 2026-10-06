"""Answer composition and verification, with one bounded repair, and the answer's SPARQL.

A mixin of ``ExtractionEngine``.

Large evidence sets (the evidence document as a run's input can leave thousands of triples) do not fit
one prompt: the triples are split into batches whose composition and verification prompts fit the
model's context window (``HARNESS_MODEL_CONTEXT_TOKENS``), each batch is answered on its own with the
same repair rule, and the answers are joined, every sentence keeping its own source ids. A batch whose
answer is unsupported is left out; only when every batch is unsupported does the answer fail.
"""

from __future__ import annotations

import json
from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_semantic import HRN

from .call_control import parallel_map
from .citations import _BRACKET, citation_ids, drop_unknown_citations
from .errors import ContextOverflow, ModelFailure, UnsupportedAnswer
from .events import emit_model_event, labelled
from .model_client import ModelClient, complete, estimate_tokens
from .schemas import AnswerResult, ExtractedTriple, SchemaInference


class AnswerStage:
    @labelled("answer composition and verification")
    def answer(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool = False,
        verifier: ModelClient | None = None,
        notes: list[str] | None = None,
        report: dict[str, Any] | None = None,
    ) -> AnswerResult:
        """``notes`` (agentic runs) go to composition only; the verifier sees the plain question.

        One bounded repair (Phase 5.1): when the verifier removes every claim of the answer, or it cites
        only ids that are no triple's source, it is written once more from the triples alone (no
        research notes), told which statements or ids were rejected. ``report["repaired"]`` records that
        it happened; ``report["citations_dropped"]`` counts sentences dropped from the returned answer
        for citing unknown ids. A second rejection raises ``UnsupportedAnswer``; the caller decides what
        to show instead. Evidence that does not fit one prompt is answered in batches (module docstring);
        ``report["batches"]`` then describes them.
        """
        batches = self._answer_batches(question, inferred, triples, verify, verifier, notes)
        if len(batches) > 1:
            return self._answer_in_batches(question, inferred, triples, batches, verify, verifier, notes, report)
        try:
            return self._answer_once(question, inferred, triples, verify, verifier, notes, report)
        except ContextOverflow:
            if len(triples) < 2:
                raise
            # The estimate was too low for this evidence: halve it and answer the halves.
            half = len(triples) // 2
            return self._answer_in_batches(question, inferred, triples, [triples[:half], triples[half:]],
                                           verify, verifier, notes, report)

    def _answer_once(
        self, question: str, inferred: SchemaInference, triples: list[ExtractedTriple], verify: bool,
        verifier: ModelClient | None, notes: list[str] | None, report: dict[str, Any] | None,
    ) -> AnswerResult:
        """One composition (and verification) over ``triples``, with the bounded repair."""
        try:
            return self._compose(question, inferred, triples, verify, verifier, notes, report=report)
        except UnsupportedAnswer as first:
            if first.stage not in {"verification", "citation"}:
                raise
            if first.stage == "citation" and not first.unknown_sources:
                raise                                      # nothing to tell the composer: same prompt again
            emit_model_event("Answer unsupported", "writing the answer once more from the triples alone")
            if report is not None:
                report["repaired"] = True
                report["notes_dropped"] = bool(notes)
            try:
                return self._compose(question, inferred, triples, verify, verifier, None,
                                     repair=first.removed or [item["text"] for item in first.sentences
                                                              if item["verdict"] == "removed"],
                                     bad_sources=first.unknown_sources, report=report)
            except UnsupportedAnswer as second:
                raise UnsupportedAnswer(f"{first}; after one repair: {second}", stage=second.stage,
                                        removed=second.removed, sentences=second.sentences,
                                        unknown_sources=second.unknown_sources) from second

    # ─── Batches ─────────────────────────────────────────────────────────────

    def _answer_batches(self, question: str, inferred: SchemaInference, triples: list[ExtractedTriple],
                        verify: bool, verifier: ModelClient | None,
                        notes: list[str] | None) -> list[list[ExtractedTriple]]:
        """The triples in consecutive batches whose prompts fit the context window: composition gets the
        batch, verification the batch plus a draft of up to one reply."""
        config = settings()
        models = [self.model, *([verifier] if verify and verifier is not None else [])]
        context = min(config.context_tokens(str(getattr(model, "model", "") or "")) for model in models)
        reply = config.max_output_tokens
        empty = render("extraction/answer_composition", {
            "question": question, "classes": inferred.classes, "properties": inferred.properties, "triples": [],
            **({"research_notes": list(notes)} if notes else {})})
        overhead = estimate_tokens(empty.system, empty.user) + 512
        budget = context - overhead - reply - (reply if verify else 0)
        budget = max(budget, 1024)
        batches: list[list[ExtractedTriple]] = [[]]
        used = 0
        for triple in triples:
            cost = estimate_tokens(json.dumps(triple.model_dump()))
            if batches[-1] and used + cost > budget:
                batches.append([])
                used = 0
            batches[-1].append(triple)
            used += cost
        return [batch for batch in batches if batch] or [list(triples)]

    def _answer_in_batches(
        self, question: str, inferred: SchemaInference, triples: list[ExtractedTriple],
        batches: list[list[ExtractedTriple]], verify: bool, verifier: ModelClient | None,
        notes: list[str] | None, report: dict[str, Any] | None,
    ) -> AnswerResult:
        config = settings()
        total = len(batches)
        if config.answer_max_batches and total > config.answer_max_batches:
            # Spread over the evidence, not the first batches only: D is ordered by paper.
            keep = config.answer_max_batches
            batches = [batches[round(index * total / keep)] for index in range(keep)]
        emit_model_event("Answer in batches", f"{sum(len(b) for b in batches)} of {len(triples)} triples in "
                                              f"{len(batches)} of {total} batches that fit the context window")

        def one(batch: list[ExtractedTriple]) -> tuple[AnswerResult | None, UnsupportedAnswer | None, dict]:
            part: dict[str, Any] = {}
            try:
                return self.answer(question, inferred, batch, verify, verifier, notes, part), None, part
            except UnsupportedAnswer as exc:
                return None, exc, part

        results = parallel_map(one, batches, config.answer_batch_concurrency)
        answered = [(result, part) for result, _, part in results if result is not None]
        failures = [error for _, error, _ in results if error is not None]
        if report is not None:
            report["batches"] = {"total": total, "composed": len(batches), "answered": len(answered),
                                 "unsupported": len(failures), "triples": len(triples),
                                 "triples_composed": sum(len(b) for b in batches)}
            report["repaired"] = any(part.get("repaired") for _, part in answered)
            dropped = [part["citations_dropped"] for _, part in answered if part.get("citations_dropped")]
            if dropped:
                report["citations_dropped"] = {"sentences": sum(item["sentences"] for item in dropped),
                                               "ids": [i for item in dropped for i in item["ids"]][:5]}
        if not answered:
            first = failures[0]
            raise UnsupportedAnswer(f"no batch of the {len(triples)} triples gave a supported answer "
                                    f"({len(batches)} batches); first: {first}", stage=first.stage,
                                    removed=first.removed, sentences=first.sentences,
                                    unknown_sources=first.unknown_sources)
        parts = [result for result, _ in answered]
        cited = list(dict.fromkeys(source for result in parts for source in result.cited_source_ids))
        return AnswerResult(answer=" ".join(result.answer.strip() for result in parts if result.answer.strip()),
                            cited_source_ids=cited, sparql=self.answer_query(triples, cited),
                            sentences=[item for result in parts for item in result.sentences])

    def _compose(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool,
        verifier: ModelClient | None,
        notes: list[str] | None,
        repair: list[str] | None = None,
        bad_sources: list[str] | None = None,
        report: dict[str, Any] | None = None,
    ) -> AnswerResult:
        result = self._draft(question, inferred, triples, notes, repair, bad_sources, report)
        return self._verify(question, triples, result, verifier, notes) if verify else result

    def _draft(self, question: str, inferred: SchemaInference, triples: list[ExtractedTriple],
               notes: list[str] | None, repair: list[str] | None, bad_sources: list[str] | None,
               report: dict[str, Any] | None) -> AnswerResult:
        """Composition: the answer from the triples, sentences citing unknown sources dropped."""
        payload: dict[str, Any] = {
            "question": question,
            "classes": inferred.classes,
            "properties": inferred.properties,
            "triples": [triple.model_dump() for triple in triples],
        }
        if notes:
            payload["research_notes"] = list(notes)
        extra: dict[str, Any] = {"repair": [str(item)[:300] for item in repair[:8]]} if repair else {}
        if bad_sources:
            extra["bad_sources"] = [str(item)[:80] for item in bad_sources[:8]]
        try:
            reply = complete(self.model, render("extraction/answer_composition", payload, **extra))
        except ModelFailure as exc:
            if "output limit" not in str(exc):
                raise
            # Cut off (a very long answer or a repetition loop): once more with explicit limits.
            emit_model_event("Answer too long", "retrying answer composition with a shorter format")
            reply = complete(self.model, render("extraction/answer_composition", payload, brief=True, **extra))
        if isinstance(reply, dict):
            reply = {key: value for key, value in reply.items() if key != "sentences"}   # set only by verification
        result = AnswerResult.model_validate(reply)
        sources = {triple.source_id for triple in triples}
        if report is not None:
            report.pop("citations_dropped", None)             # describes the attempt that is returned
        text, cited, unknown, dropped = drop_unknown_citations(result.answer, result.cited_source_ids, sources)
        if unknown and (not text or not cited):
            raise UnsupportedAnswer(
                f"no sentence of the answer cites a source of the extracted evidence (it cited {unknown[:3]}); "
                f"{len(triples)} triples available", stage="citation", unknown_sources=unknown)
        if dropped:
            # e.g. "[research_notes]" or a passage id no triple carries: those sentences have no evidence.
            emit_model_event("Answer cites unknown sources",
                             f"dropped {dropped} sentence{'s' if dropped > 1 else ''} citing {', '.join(unknown[:3])}")
            if report is not None:
                report["citations_dropped"] = {"sentences": dropped, "ids": unknown[:5]}
        if unknown:
            result = result.model_copy(update={"answer": text, "cited_source_ids": cited})
        return result.model_copy(update={"sparql": self.answer_query(triples, result.cited_source_ids)})

    def _verify(self, question: str, triples: list[ExtractedTriple], result: AnswerResult,
                verifier: ModelClient | None, notes: list[str] | None) -> AnswerResult:
        """Verification: the answer becomes the accepted sentences; none accepted raises UnsupportedAnswer."""
        sources = {triple.source_id for triple in triples}
        reviewed = complete(verifier or self.model, render(
            "extraction/answer_verification",
            {
                "question": question,
                "draft_answer": result.answer,
                "triples": [triple.model_dump() for triple in triples],
            },
            ensure_ascii=False,
        ))
        removed = reviewed.get("removed_claims")
        removed = [str(item) for item in removed] if isinstance(removed, list) else []
        sentences = self._sentence_verdicts(reviewed.get("sentences"), sources)
        accepted = [item for item in sentences if item["verdict"] == "accepted"]
        if (reviewed.get("supported") is not True or not str(reviewed.get("answer", "")).strip()
                or (sentences and not accepted)):
            removed = removed or [item["text"] for item in sentences if item["verdict"] == "removed"]
            example = f", e.g. {removed[0][:120]!r}" if removed else ""
            raise UnsupportedAnswer(
                f"answer has no fully supported factual claim (verifier removed {len(removed)} "
                f"claims{example}; {len(triples)} triples available"
                f"{', research notes used by composition' if notes else ''})",
                removed=removed, sentences=sentences)
        if accepted:
            # With per-sentence verdicts the answer is exactly the accepted sentences, in order, and
            # its sources are theirs (each already checked against the supplied triples).
            text = " ".join(item["text"] for item in accepted)
            cited = list(dict.fromkeys(source for item in accepted for source in item["sourceIds"]))
        else:
            verified_sources = reviewed.get("cited_source_ids", [])
            if not isinstance(verified_sources, list) or not verified_sources or not set(verified_sources) <= sources:
                raise UnsupportedAnswer("verified answer lacks valid triple provenance", stage="provenance")
            text, cited = str(reviewed["answer"]), [str(source) for source in verified_sources]
        return result.model_copy(update={
            "answer": text, "cited_source_ids": cited,
            "sparql": self.answer_query(triples, cited), "sentences": sentences,
        })

    @staticmethod
    def _sentence_verdicts(raw: object, sources: set[str]) -> list[dict[str, Any]]:
        """The verifier's per-sentence verdicts; a sentence counts as accepted only when it is marked
        supported and cites at least one source of the supplied triples."""
        def clean(value: object) -> str:
            return str(value).strip().strip("[]").strip()

        out: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            text = str(item["text"]).strip()
            ids = list(dict.fromkeys(clean(value) for value in item.get("source_ids") or [] if clean(value) in sources))
            accepted = item.get("supported") is True and bool(ids)
            reason = str(item.get("reason") or "")
            if item.get("supported") is True and not ids:
                reason = reason or "cites no source of the supplied triples"
            in_text = [item for group in _BRACKET.findall(text) for item in citation_ids(group, sources) or []]
            outside = [value for value in in_text if value not in sources]
            if accepted and outside:
                accepted, reason = False, f"cites a source not among the triples: {', '.join(outside[:3])}"
            if accepted and not in_text:
                # Keep the sentence traceable in the answer text: its sources in brackets before the full stop.
                body, stop = (text[:-1], text[-1]) if text[-1] in ".!?" else (text, "")
                text = f"{body.rstrip()} [{', '.join(ids)}]{stop}"
            out.append({"text": text, "sourceIds": ids, "verdict": "accepted" if accepted else "removed",
                        "reason": reason})
        return out

    def answer_query(self, triples: list[ExtractedTriple], cited: list[str]) -> str:
        """SPARQL that selects the triples behind the answer: one pattern per predicate they use.

        Built from the cited triples (all triples when none is cited), so the query always matches
        the answer and uses only subschema terms.
        """
        chosen = [triple for triple in triples if triple.source_id in set(cited)] or list(triples)
        iris: list[str] = []
        for triple in chosen:
            iri = self.property_names.get(triple.predicate)
            if iri is not None and str(iri) not in iris:
                iris.append(str(iri))
        head = f"PREFIX hrn: <{HRN}>\n"
        if not iris:
            return head + "SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object . }"
        names = [f"hrn:{iri[len(str(HRN)):]}" if iri.startswith(str(HRN)) else f"<{iri}>" for iri in iris]
        branches = [f"{{ ?subject {name} ?object . BIND({name} AS ?predicate) }}" for name in names]
        return head + "SELECT ?subject ?predicate ?object WHERE {\n  " + "\n  UNION ".join(branches) + "\n}"
