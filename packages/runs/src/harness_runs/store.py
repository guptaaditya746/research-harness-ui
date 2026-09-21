"""File artifacts with atomic writes; DB index is introduced by the Alembic schema."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

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
        if name not in required_artifacts("agentic"):
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
        if name not in required_artifacts("agentic"):
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
