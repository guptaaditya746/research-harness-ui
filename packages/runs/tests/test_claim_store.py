from pathlib import Path

from harness_runs.claims import ClaimStore
from harness_schema.models import ClaimSubgraph


def test_claim_store_reuses_attributed_claims(tmp_path: Path) -> None:
    store = ClaimStore(tmp_path / "claims.sqlite")
    claim = ClaimSubgraph.model_validate({
        "id": "claim-one", "kind": "ResultClaim", "paperId": "paper-a",
        "fields": {"object": {"value": "41.5% success rate improvement", "sourceIds": ["p1"]}},
        "sourceIds": ["p1"], "evidence": {"p1": "41.5% improvement"},
    })
    store.upsert([claim])
    assert [item.id for item in store.search("success rate improvement")] == ["claim-one"]
    assert store.search("unrelated geology") == []
