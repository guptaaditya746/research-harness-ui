"""File artifacts with atomic writes; DB index is introduced by the Alembic schema."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from uuid import uuid4

from harness_schema.models import Run, required_artifacts


def invariant_failure(run: Run, number: int) -> Run:
    if number not in range(1, 9):
        raise ValueError("invariant number must be 1..8")
    return run.model_copy(
        update={"status": "incomplete", "reason": f"invariant:{number}", "chosen_reading": None}
    )


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in run_id
        ):
            raise ValueError("invalid run id")
        return self.root / run_id

    def write(self, run_id: str, name: str, body: bytes) -> dict[str, str | int]:
        allowed = set(required_artifacts("agentic")) | set(required_artifacts("baseline-enhanced"))
        if name not in allowed:
            raise ValueError("unknown artifact name")
        target_dir = self._run_dir(run_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target_dir, delete=False) as temporary:
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)
        temp_path.replace(target_dir / name)
        return {
            "name": name,
            "path": str(target_dir / name),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    def read(self, run_id: str, name: str) -> bytes:
        allowed = set(required_artifacts("agentic")) | set(required_artifacts("baseline-enhanced"))
        if name not in allowed:
            raise ValueError("unknown artifact name")
        return (self._run_dir(run_id) / name).read_bytes()

    def finalize(self, run: Run, artifacts: dict[str, bytes]) -> None:
        expected = required_artifacts(run.path)
        if run.status not in {"answered", "awaiting_reading", "incomplete", "stopped"}:
            raise ValueError("only terminal runs can be finalized")
        extras = set(artifacts) - set(expected)
        if extras:
            raise ValueError(f"unexpected artifacts: {sorted(extras)}")
        for name in expected:
            if name == "run.json":
                body = run.model_dump_json(by_alias=True, indent=2).encode()
            else:
                body = artifacts.get(name, b"")
            self.write(run.id, name, body)
        manifest = {name: hashlib.sha256(self.read(run.id, name)).hexdigest() for name in expected}
        (self._run_dir(run.id) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def load_run(self, run_id: str) -> Run:
        return Run.model_validate_json(self.read(run_id, "run.json"))

    def list_runs(self) -> list[Run]:
        if not self.root.exists():
            return []
        runs = [Run.model_validate_json(path.read_text()) for path in self.root.glob("*/run.json")]
        return sorted(runs, key=lambda run: run.started_at, reverse=True)

    def update_run(self, run: Run) -> None:
        self.write(run.id, "run.json", run.model_dump_json(by_alias=True, indent=2).encode())
        manifest_path = self._run_dir(run.id) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["run.json"] = hashlib.sha256(self.read(run.id, "run.json")).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


class GoldCandidateStore:
    """Atomic JSONL storage matching the dataset_questions shape used by the editor."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = RLock()

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        with NamedTemporaryFile("w", dir=self.path.parent, delete=False) as temporary:
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)
        temp_path.replace(self.path)

    def list(self) -> list[dict]:
        with self.lock:
            return self._rows()

    def promote(self, run: Run) -> dict:
        if run.rating is None:
            raise ValueError("a run must be human-rated before promotion")
        with self.lock:
            rows = self._rows()
            existing = next((row for row in rows if row["run_id"] == run.id), None)
            if existing:
                return existing
            reading = run.readings[run.chosen_reading or 0] if run.readings else None
            row = {
                "id": f"gold-{uuid4().hex[:10]}",
                "run_id": run.id,
                "question": run.question,
                "ambiguous": False,
                "annotators": [{"source": "run_rating", "rating": run.rating}],
                "readings": [reading.model_dump(by_alias=True)] if reading else [],
                "gold_triples": [
                    triple.model_dump(by_alias=True)
                    for triple in (reading.triples if reading else [])
                ],
                "paper_ids": sorted(
                    {
                        triple.s
                        for triple in (reading.triples if reading else [])
                        if triple.s.startswith("p")
                    }
                ),
                "notes": "Promoted from a human-rated run; review required.",
                "status": "candidate",
            }
            rows.append(row)
            self._write(rows)
            return row

    def update(self, candidate_id: str, changes: dict) -> dict:
        allowed = {"ambiguous", "readings", "gold_triples", "paper_ids", "notes", "status"}
        if set(changes) - allowed:
            raise ValueError("unknown candidate fields")
        with self.lock:
            rows = self._rows()
            for index, row in enumerate(rows):
                if row["id"] == candidate_id:
                    rows[index] = {**row, **changes}
                    self._write(rows)
                    return rows[index]
        raise KeyError(candidate_id)
