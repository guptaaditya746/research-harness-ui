"""Dense passage embeddings: build once per corpus, search by cosine similarity.

The embedding model is the harness's ``HARNESS_MODEL_EMBEDDING`` alias
(``qwen-embed``), served by thesis-llm-setup behind LiteLLM's OpenAI-compatible
``/embeddings`` endpoint. Vectors are L2-normalised and stored as float16 in
``<corpus>/index/dense/embeddings.npy`` next to ``passage_ids.json`` and
``meta.json`` (model, dimension, count, a hash of passages.jsonl). A plain
matrix product is exact and fast at this scale (1,000 papers ≈ 100k passages ≈
200 MB at 1,024 dimensions), so no approximate index is needed.

Queries get Qwen3-Embedding's instruction prefix; passages are embedded as is,
which is how that model family is meant to be used (asymmetric retrieval).

Build: ``uv run python -m harness_corpus --embed`` (``make embed``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import httpx
import numpy as np
from harness_config import settings

LOGGER = logging.getLogger(__name__)
QUERY_INSTRUCTION = (
    "Instruct: Given a question about scientific papers, retrieve the passages that answer it\nQuery: "
)
DENSE_DIR = ("index", "dense")


def passages_digest(root: Path) -> str:
    path = root / "passages.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def embed_texts(texts: Sequence[str], *, model: str | None = None, batch_size: int = 64,
                timeout_s: float = 120.0, client: httpx.Client | None = None) -> np.ndarray:
    """Embed texts through the OpenAI-compatible endpoint; rows are L2-normalised float32."""
    config = settings()
    alias = model or config.model_embedding
    url = f"{config.litellm_base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    owned = client is None
    http = client or httpx.Client(timeout=timeout_s)
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = [text[:8000] for text in texts[start:start + batch_size]]
            response = http.post(url, headers=headers, json={"model": alias, "input": batch})
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
    finally:
        if owned:
            http.close()
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class DenseIndex:
    """Loaded embeddings for one corpus; ``available`` is False when absent or stale."""

    def __init__(self, root: Path, passage_ids: Iterable[str] | None = None) -> None:
        self.root = root
        folder = root.joinpath(*DENSE_DIR)
        self.available = False
        self.meta: dict = {}
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None
        if not (folder / "embeddings.npy").exists():
            return
        try:
            self.meta = json.loads((folder / "meta.json").read_text())
            self.ids = json.loads((folder / "passage_ids.json").read_text())
            self.matrix = np.load(folder / "embeddings.npy", mmap_mode="r")
        except (OSError, ValueError) as exc:
            LOGGER.warning("Dense index at %s is unreadable (%s); using BM25 only", folder, exc)
            return
        if self.matrix.shape[0] != len(self.ids):
            LOGGER.warning("Dense index at %s has %d vectors for %d ids; using BM25 only",
                           folder, self.matrix.shape[0], len(self.ids))
            return
        if self.meta.get("passages_sha256") not in (None, passages_digest(root)):
            LOGGER.warning("Dense index at %s was built for other passages; rebuild with `make embed`", folder)
            return
        if passage_ids is not None and set(passage_ids) - set(self.ids):
            LOGGER.warning("Dense index at %s misses passages; rebuild with `make embed`", folder)
            return
        self.row_of = {passage_id: row for row, passage_id in enumerate(self.ids)}
        self.available = True

    def search(self, query_vector: np.ndarray, allowed: set[str] | None = None,
               k: int = 200) -> list[tuple[str, float]]:
        """Passage ids by cosine similarity, best first, restricted to ``allowed``."""
        if not self.available or self.matrix is None:
            return []
        scores = np.asarray(self.matrix @ query_vector.astype(self.matrix.dtype), dtype=np.float32)
        if allowed is not None:
            rows = np.fromiter((self.row_of[pid] for pid in allowed if pid in self.row_of), dtype=np.int64)
            if rows.size == 0:
                return []
            chosen = rows[np.argsort(-scores[rows], kind="stable")[:k]]
        else:
            chosen = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[row], float(scores[row])) for row in chosen]


def build_dense_index(root: Path, *, model: str | None = None, batch_size: int = 64) -> dict:
    """Embed every passage of the corpus at ``root`` and write the index."""
    rows = [json.loads(line) for line in (root / "passages.jsonl").read_text().splitlines() if line.strip()]
    started = time.monotonic()
    matrix = embed_texts([row["text"] for row in rows], model=model, batch_size=batch_size)
    folder = root.joinpath(*DENSE_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", matrix.astype(np.float16))
    (folder / "passage_ids.json").write_text(json.dumps([row["passage_id"] for row in rows]))
    meta = {"model": model or settings().model_embedding, "dimension": int(matrix.shape[1]) if len(rows) else 0,
            "count": len(rows), "passages_sha256": passages_digest(root),
            "query_instruction": QUERY_INSTRUCTION, "seconds": round(time.monotonic() - started, 1)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion (Cormack et al., 2009): sum of 1 / (k + rank) over the rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))


__all__ = ["QUERY_INSTRUCTION", "DenseIndex", "build_dense_index", "embed_texts", "passages_digest", "rrf"]
