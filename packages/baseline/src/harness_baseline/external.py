"""The External path: an off-the-shelf agent from the separate external-harness repo.

That repo runs mini-SWE-agent with read-only commands over the same corpus and ontology (see its
README). Here it is one subprocess per question: ``<HARNESS_EXTERNAL_COMMAND> run --config ... --events``.
Its JSON events become trace rows as they arrive (model calls on the Agent lane of the timeline,
commands as markers). Its ``result.json`` then goes through the same outcome code as every other
path: triples whose source is a corpus passage or DBLP fact enter the answer graph, predicates it
invented get a generic parent, the DBLP backbone is added, and checks and artifacts are the usual
ones plus ``external_result.json`` and ``external_trajectory.json``. Nothing of our pipeline
(planner, retrieval, schema inference, extraction, review, verification) runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness_extraction import AnswerResult, ExtractedTriple, ModelFailure, SchemaInference
from harness_extraction.call_control import call_interrupted
from harness_semantic import CSKG_INVERSES, content_core_on

from .pipeline import Baseline20
from .state import BaselineResult, BaselineState, NoResults

# DBLP (and plain) names the agent may copy from the ontology, as the harness names them. Venue and
# year variants collapse onto the harness's venue and year; creator variants onto authoredBy.
_DBLP_NAMES = {
    "author": "authoredBy", "hasAuthor": "authoredBy", "createdBy": "authoredBy", "creator": "authoredBy",
    "yearOfPublication": "year", "yearOfEvent": "year", "publicationYear": "year",
    "publishedIn": "venue", "publishedInStream": "venue", "publishedInSeries": "venue",
    "publishedInJournal": "venue", "publishedInBook": "venue", "publishedAsPartOf": "venue",
}
# Inverse properties: (author, authorOf, paper) is (paper, authoredBy, author).
# Inverse names read the other way round: DBLP's, and CS-KG's ("usedBy" -> "uses") when the content core is on.
_INVERSE = {"authorOf": "authoredBy", "creatorOf": "authoredBy"}


def _inverse(raw: str) -> str | None:
    return _INVERSE.get(raw) or (CSKG_INVERSES.get(raw) if content_core_on() else None)
# Not relations between research entities: dropped (and counted).
_META = {"type", "a", "label", "comment", "subClassOf", "subPropertyOf", "sameAs", "bibtexType", "numberOfCreators"}
_NAME = re.compile(r"[a-z][A-Za-z0-9_]{0,63}")
# litellm exception names in the external result: the model endpoint failed, not the agent.
_PROVIDER_ERRORS = re.compile(r"APIConnectionError|AuthenticationError|RateLimitError|Timeout|ServiceUnavailable|"
                              r"InternalServerError|APIError|BadRequestError|NotFoundError|ContextWindowExceeded")
# What the external process may see of our environment: enough to run uv/Python, reach the model
# endpoint and expand its config, never our keys or data paths (DATASETS_ROOT, ARTIFACT_ROOT, ...).
_PASS_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "SHELL", "TERM",
             "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
             "LITELLM_BASE_URL", "VLLM_API_KEY")
_PASS_PREFIXES = ("UV_", "XDG_", "XH_")


class ExternalFailure(RuntimeError):
    """The external harness could not produce a result (not configured, crashed, no result file)."""


def predicate_name(raw: str) -> str | None:
    """A lowerCamelCase property name from what the agent wrote ("uses dataset" -> "usesDataset",
    "hrn:usesDataset" -> "usesDataset"); None when nothing usable is left."""
    text = re.split(r"[#/:]", str(raw).strip())[-1]
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return None
    if len(words) == 1 and _NAME.fullmatch(words[0][0].lower() + words[0][1:]):
        name = words[0][0].lower() + words[0][1:]
    else:
        name = words[0].lower() + "".join(word[:1].upper() + word[1:] for word in words[1:])
    if name in _META:
        return None
    name = _DBLP_NAMES.get(name, name)
    return name if _NAME.fullmatch(name) else None


def _external_env(corpus_root: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _PASS_ENV or key.startswith(_PASS_PREFIXES)}
    env["CORPUS_ROOT"] = str(corpus_root.resolve())          # absolute: the config may expand ${CORPUS_ROOT}
    return env


class ExternalHarness(Baseline20):
    """Runs the external harness for one question and turns its result into a Run."""

    path_name = "external"

    def __init__(self, *args: Any, command: str, config_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.command = shlex.split(command)
        self.config_path = config_path

    # ─── One run ──────────────────────────────────────────────────────────

    def run(self, question: str, budget: str = "fast", run_id: str | None = None,
            concept: dict[str, Any] | None = None) -> BaselineResult:
        run_id = run_id or f"r-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)
        self.started = time.monotonic()
        self.current_run_id = run_id
        self.trace, self.checks = [], []
        self.stage = "external harness"
        files: dict[str, bytes] = {}
        try:
            result, files = self._call(question, run_id)
            state = self._state(question, result)
            self.stage = "triple provenance"
            run, artifacts = self._success(run_id, started_at, question, budget, state)
            model = str(result.get("config", {}).get("model") or "")
            run = run.model_copy(update={"model_agent": model, "model_small": model})
        except NoResults as exc:
            self._check(self.stage, "fail", str(exc))
            run, artifacts = self._failure(run_id, started_at, question, budget, "no_results", str(exc))
        except ModelFailure as exc:
            self._check(self.stage, "fail", f"model call failed: {exc}")
            run, artifacts = self._failure(run_id, started_at, question, budget, "provider_error", str(exc))
        except ExternalFailure as exc:
            reason = str(exc).split(":", 1)[0]
            self._check(self.stage, "fail", str(exc))
            run, artifacts = self._failure(run_id, started_at, question, budget, reason, str(exc))
        artifacts.update(files)
        return {"run": run.model_copy(update={"checks": list(self.checks)}), "artifacts": artifacts}

    # ─── The subprocess ──────────────────────────────────────────────────

    def _call(self, question: str, run_id: str) -> tuple[dict[str, Any], dict[str, bytes]]:
        if not self.command or not self.config_path:
            raise ExternalFailure("external_failed: HARNESS_EXTERNAL_COMMAND and HARNESS_EXTERNAL_CONFIG must be set")
        folder = Path(tempfile.mkdtemp(prefix=f"xh-{run_id}-"))
        try:
            return self._call_in(folder, question, run_id)
        finally:
            shutil.rmtree(folder, ignore_errors=True)              # later runs must not find this one

    def _call_in(self, folder: Path, question: str, run_id: str) -> tuple[dict[str, Any], dict[str, bytes]]:
        question_file = folder / "question.txt"
        question_file.write_text(question, encoding="utf-8")
        out = folder / "run"
        command = [*self.command, "run", "--config", self.config_path, "--question-file", str(question_file),
                   "--run-id", run_id, "--out", str(out), "--events"]
        self._emit("Starting external harness", f"{Path(self.config_path).name} · {' '.join(self.command[-2:])}")
        stderr = (folder / "stderr.log").open("w+", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, stdin=subprocess.DEVNULL,
                                       text=True, encoding="utf-8", errors="replace",
                                       env=_external_env(Path(self.corpus.root)), start_new_session=os.name == "posix")
        except OSError as exc:
            stderr.close()
            raise ExternalFailure(f"external_failed: cannot start {self.command[0]!r} ({exc})") from None
        lines: queue.Queue[str | None] = queue.Queue()

        def read() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)                                     # always ends the loop below

        threading.Thread(target=read, name=f"xh-{run_id}", daemon=True).start()
        try:
            while True:
                reason = call_interrupted()                 # Stop or the run deadline
                if reason:
                    raise ModelFailure(reason)
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                self._event(line)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise ExternalFailure("external_failed: the process closed its output but did not exit") from None
        except BaseException:
            self._kill(process)
            raise
        finally:
            stderr.seek(0)
            errors = stderr.read()[-2000:]
            stderr.close()
        result_path = out / "result.json"
        if process.returncode != 0 or not result_path.is_file():
            last = " ".join(errors.strip().splitlines()[-3:])[:400]
            raise ExternalFailure(f"external_failed: exit code {process.returncode}"
                                  + (f"; {last}" if last else "; no result.json"))
        body = result_path.read_bytes()
        files = {"external_result.json": body}
        trajectory = out / "trajectory.json"
        if trajectory.is_file():
            files["external_trajectory.json"] = trajectory.read_bytes()
        return json.loads(body), files

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        """SIGTERM first (xh then ends the agent's running command, which has its own session), then
        SIGKILL for whatever is left of the process group."""
        if process.poll() is not None:
            return
        for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                else:
                    process.kill()
            try:
                process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                continue

    def _event(self, line: str) -> None:
        """One JSON event from ``xh run --events`` as a trace row (other output is ignored)."""
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        model = str(event.get("model") or "").split("/", 1)[-1] or "external model"
        if kind == "model_call":
            seconds = float(event.get("seconds") or 0)
            if event.get("ok"):
                tokens = int(event.get("prompt_tokens") or 0) + int(event.get("completion_tokens") or 0)
                self._emit("Model call finished", f"{model} · external agent · {seconds:.1f}s"
                           + (f" · {tokens} tokens" if tokens else ""))
            else:
                self._emit("Model call failed", f"{model} · external agent · {event.get('error', 'error')} · {seconds:.1f}s")
        elif kind == "command":
            self._emit("External command", " ".join(str(event.get("command", "")).split())[:200])
        elif kind == "finish":
            self._emit("External harness finished", f"{event.get('status')} · {event.get('triples', 0)} triples")

    # ─── Result → pipeline state ─────────────────────────────────────────

    def _state(self, question: str, result: dict[str, Any]) -> BaselineState:
        usage = result.get("usage") or {}
        harness = result.get("harness") or {}
        summary = (f"{harness.get('name', 'external')} {harness.get('version', '')} · {usage.get('model_calls', 0)} "
                   f"model calls · {usage.get('commands', 0)} commands · {result.get('duration_s', 0)}s").replace("  ", " ")
        status = str(result.get("status") or "")
        self._same_corpus(result.get("knowledge_base") or {})
        if status != "answered":
            detail = result.get("problem") or result.get("error") or result.get("exit_status") or status
            if status == "error" and _PROVIDER_ERRORS.search(str(detail)):
                raise ModelFailure(str(detail))
            why = [f"exit {result['exit_status']}" if result.get("exit_status") else "",
                   f"{usage['format_errors']} replies without a command" if usage.get("format_errors") else ""]
            raise ExternalFailure(f"external_{status or 'failed'}: {detail} ({summary}"
                                  + "".join(f"; {item}" for item in why if item) + ")")
        dropped = result.get("dropped_triples") or []
        triples, unmapped = self._triples(result.get("triples") or [])
        notes = [f"{len(dropped)} dropped by the answer contract" if dropped else "",
                 "answer taken from the agent's last reply (no answer.json)"
                 if result.get("answer_source") == "final_message" else "",
                 f"{unmapped} with a predicate that is not a relation (dropped)" if unmapped else ""]
        audit = [f"{result['evidence_not_in_source']} quote evidence not found in their passage"
                 if result.get("evidence_not_in_source") else "",
                 f"{len(result['outside_paths'])} commands looked outside the run folder (sandbox off): "
                 f"{result['outside_paths'][0][:80]}" if result.get("outside_paths") else ""]
        self._check("external harness", "warn" if any(audit) else "pass",
                    "; ".join(item for item in [f"{summary}; {len(result.get('triples') or [])} triples", *notes, *audit]
                              if item))
        self._check("retrieval", "skipped", f"the external agent searched on its own ({usage.get('commands', 0)} commands)")
        if not triples:
            answer = " ".join(str(result.get("answer") or "").split())[:300]
            raise NoResults(f"the external agent answered without a usable triple ({summary})"
                            + (f": {answer}" if answer else ""))
        self._check("extraction", "pass", f"{len(triples)} triples written by the external agent")
        cited = list(dict.fromkeys(triple.source_id for triple in triples))
        answer = AnswerResult(answer=str(result.get("answer") or ""), cited_source_ids=cited,
                              sparql=self.extraction.answer_query(triples, cited))
        passages = [self.corpus.by_passage[source] for source in cited if source in self.corpus.by_passage]
        inferred = SchemaInference(
            label="External agent", gloss=f"Triples written by {harness.get('name', 'the external harness')}.",
            classes=["Paper"], properties=list(dict.fromkeys(triple.predicate for triple in triples)))
        commands = [event for event in result.get("events") or [] if event.get("type") == "command"]
        return {
            "question": question, "passages": passages, "inferred": inferred, "extracted": triples, "answer": answer,
            "plan": None, "paper_ids": [], "computed": [],
            "retrieval": {"stop_reason": "external", "commands": [event.get("command") for event in commands],
                          "usage": usage, "config": result.get("config")},
            "extraction_batches": [{"stage": "external", "triples": len(triples), "dropped": len(dropped)}],
        }

    def _same_corpus(self, knowledge_base: dict[str, Any]) -> None:
        """The external index must be built from our corpus: same folder and the same number of passages."""
        folder = knowledge_base.get("corpus_dir")
        ours = Path(self.corpus.root).resolve()
        if folder and Path(folder).resolve() != ours:
            raise ExternalFailure(f"external_failed: the external harness read the corpus in {folder}, not {ours}")
        passages = knowledge_base.get("passages")
        if passages is not None and int(passages) != len(self.corpus.passages):
            raise ExternalFailure(f"external_failed: its index has {passages} passages, the corpus "
                                  f"{len(self.corpus.passages)} (rebuild it: xh index)")

    def _triples(self, rows: list[dict[str, Any]]) -> tuple[list[ExtractedTriple], int]:
        """The agent's triples with harness property names, and how many had no usable predicate. A name
        the schema lacks becomes a run-local property with a generic parent (like any invented property)."""
        triples, unmapped = [], 0
        for row in rows:
            raw = re.split(r"[#/:]", str(row.get("predicate") or "").strip())[-1]
            subject, object_ = str(row["subject"]), str(row["object"])
            if _inverse(raw):
                name, subject, object_ = _inverse(raw), object_, subject
            else:
                name = predicate_name(raw)
            if name is None:
                unmapped += 1
                continue
            if name not in self.extraction.property_names:
                self.extraction.register_local_relation(
                    name, f"Relation used by the external agent: {name}.",
                    domain_name="Paper", range_name="Paper" if object_ in self.corpus.by_paper else "xsd:string")
            triples.append(ExtractedTriple(
                subject=subject, predicate=name, object=object_, source_id=str(row["source_id"]),
                evidence=str(row.get("evidence") or "") or f"Stated by the external agent from {row['source_id']}."))
        return triples, unmapped

    def _workflow_tool_calls(self, state: BaselineState, run_id: str) -> list[dict[str, Any]]:
        """The agent's commands, one tool call each (the Benchmark page counts them)."""
        return [{"call_id": f"tool-external-{index:03d}", "agent": "external",
                 "tool": (str(command).split() or ["bash"])[0], "transport": "bash",
                 "arguments": {"command": command}, "status": "ok"}
                for index, command in enumerate(state.get("retrieval", {}).get("commands") or [], start=1)]


__all__ = ["ExternalFailure", "ExternalHarness", "predicate_name"]
