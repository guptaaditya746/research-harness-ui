import asyncio
import json
from pathlib import Path

from harness_corpus.ingest import ingest, local_identifier
from harness_corpus.store import CorpusStore, matches
from harness_corpus.tei import parse_tei
from harness_mcp_corpus.server import create_server


def tei(title: str, doi: str, paragraph: str, reference: str = "") -> bytes:
    return f"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc>
    <biblStruct><analytic><title level="a">{title}</title><author><persName>Ada Scholar</persName></author></analytic>
    <idno type="DOI">{doi}</idno><date when="2024"/></biblStruct></sourceDesc></fileDesc>
    <profileDesc><abstract><p>Research on robotics.</p></abstract></profileDesc></teiHeader>
    <text><body><div><head>Introduction</head><p>{paragraph}</p></div></body></text>
    <listBibl>{reference}</listBibl></TEI>""".encode()


def test_ingest_and_structural_citation(tmp_path: Path, monkeypatch) -> None:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    root = tmp_path / "corpus"
    cache = root / ".cache" / "tei"
    cache.mkdir(parents=True)
    for name in ("17", "2"):
        (pdfs / f"{name}.pdf").write_bytes(f"fixture-{name}".encode())
    ref = '<biblStruct><analytic><title level="a">World Models</title></analytic><idno type="DOI">10.1/world</idno></biblStruct>'
    (cache / "17.grobid.tei.xml").write_bytes(
        tei("Robot Control", "10.1/robot", "World models predict robot motion.", ref)
    )
    (cache / "2.grobid.tei.xml").write_bytes(
        tei("World Models", "10.1/world", "A predictive model learns from video.")
    )
    report = ingest(pdfs, root, "http://unused", skip_grobid=True, skip_dblp=True)
    assert report["papers"] == 2
    papers = [json.loads(line) for line in (root / "papers.jsonl").read_text().splitlines()]
    assert all(
        p["identifier"] == local_identifier(pdfs / f"{p['paper_id'][1:]}.pdf") for p in papers
    )
    assert all(
        p["identifier_source"] == "local" and p["dblp_status"] == "unresolved" for p in papers
    )
    assert len({p["identifier"] for p in papers}) == 2
    store = CorpusStore(root)
    cited = store.citations("p17")
    assert cited["total"] == 1
    assert cited["items"][0]["resolved_paper_id"] == "p2"
    assert cited["items"][0]["fact_id"] == "p17-ref-1-cites"
    fact = [f for f in store.facts("p17")["items"] if f["predicate"] == "cites"]
    assert fact[0]["object"] == "p2"
    assert fact[0]["source_id"] == cited["items"][0]["reference_id"]
    assert all(
        row["predicate"] == "author"
        for row in store.facts("p17", predicate="author")["items"]
    )
    assert store.citations("p2", "in")["total"] == 1
    hits = store.search('"world models" AND robot')
    assert hits["total"] == 1
    assert hits["items"][0]["passage_id"] == "p17-s1-p1"
    assert store.bulk_check("video", ["p17", "p2"])["items"][1]["hit"]
    assert (
        store.read_passage("p17-s1-p1")["items"][0]["text"] == "World models predict robot motion."
    )
    mcp = create_server(root)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names >= {
        "list_papers",
        "paper_outline",
        "abstracts",
        "search",
        "grep",
        "read_passage",
        "facts",
        "citations",
        "bulk_check",
        "keyphrases",
        "results",
        "expand",
        "prior_triples",
        "rank",
        "rank_evidence",
    }
    response = asyncio.run(mcp.call_tool("citations", {"paper_id": "p17"}))
    assert "p2" in str(response)
    assert any(
        row["passage_id"] == "p17-s1-p1"
        for row in store.expand("p17-s1-p1", 1)["items"]
    )
    assert store.results("predict")["total"] == 0  # results contain numeric claims only
    ranked = store.rank_evidence("robot", ["p17-s1-p1"])
    assert ranked["total"] == 1
    assert ranked["items"][0]["passage_id"] == "p17-s1-p1"
    assert ranked["items"][0]["score"] > 0
    store.by_paper["p28"] = {"paper_id": "p28", "title": "DayDreamer: World Models for Physical Robot Learning"}
    assert store.mentioned_paper_ids("What limitation does DayDreamer report?") == ["p28"]
    assert "items" in store.keyphrases()
    monkeypatch.setattr(
        "harness_corpus.ingest.lookup_dblp_many",
        lambda papers, client: {"p17": "https://dblp.org/rec/conf/test/Robot24"},
    )
    ingest(pdfs, root, "http://unused", skip_grobid=True)
    joined = [json.loads(line) for line in (root / "papers.jsonl").read_text().splitlines()]
    by_id = {p["paper_id"]: p for p in joined}
    assert by_id["p17"]["identifier"] == "dblp:conf/test/Robot24"
    assert by_id["p17"]["identifier_source"] == "dblp"
    assert by_id["p17"]["dblp_status"] == "verified"
    assert by_id["p2"]["identifier_source"] == "local"


def test_manifest_metadata_overrides_parsed_bibliography(tmp_path: Path) -> None:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "2025.acl-long.1.pdf").write_bytes(b"fixture-pdf")
    (pdfs / "manifest.tsv").write_text(
        "filename\ttitle\tdoi\tyear\tvenue\tauthors\tanthology_id\n"
        '2025.acl-long.1.pdf\tOfficial Title\t10.18653/v1/2025.acl-long.1\t2025\tACL 2025\t["Ada Scholar", "B. Author"]\t2025.acl-long.1\n'
    )
    root = tmp_path / "corpus"
    cache = root / ".cache" / "tei"
    cache.mkdir(parents=True)
    (cache / "2025.acl-long.1.grobid.tei.xml").write_bytes(
        tei("Noisy Parsed Title", "10.1/old", "A passage about language models.")
    )
    ingest(pdfs, root, "http://unused", skip_grobid=True, skip_dblp=True)
    paper = json.loads((root / "papers.jsonl").read_text().splitlines()[0])
    assert paper["title"] == "Official Title"
    assert paper["doi"] == "10.18653/v1/2025.acl-long.1"
    assert paper["year"] == 2025
    assert paper["venue"] == "ACL 2025"
    assert paper["authors"] == ["Ada Scholar", "B. Author"]
    assert paper["anthology_id"] == "2025.acl-long.1"
    assert "A passage about language models" in (root / "passages.jsonl").read_text()


def test_search_dsl() -> None:
    text = "World models predict future robot motion"
    assert matches('"world models" AND robot', text)
    assert matches("world NEAR/3 predict", text)
    assert matches("vision OR robot", text)
    assert matches("robot AND NOT video", text)
    assert not matches("world AND video", text)


def test_tei_does_not_turn_affiliations_into_authors() -> None:
    xml = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader><fileDesc><sourceDesc><biblStruct><analytic>
        <title level="a">FOCUS</title>
        <author><persName><forename>Xianmin</forename><surname>Wang</surname></persName></author>
        <author><affiliation><orgName>Guangzhou University</orgName><country>China</country></affiliation></author>
      </analytic></biblStruct></sourceDesc></fileDesc></teiHeader>
      <text><body><div><p>Abstract.</p></div></body></text>
    </TEI>'''
    paper, _, _ = parse_tei(xml, "p6")
    assert paper["authors"] == ["Xianmin Wang"]
    assert paper["excluded_affiliation_authors"] == 1


def test_model_written_patterns_have_a_time_budget() -> None:
    import time as _time

    from harness_corpus.store import BoundedPattern

    started = _time.monotonic()
    pattern = BoundedPattern(r"(x+x+)+y", budget_s=0.3)
    try:
        pattern.search("x" * 5000)
    except ValueError as exc:
        assert "time budget" in str(exc)
    else:
        raise AssertionError("pathological pattern was not bounded")
    assert _time.monotonic() - started < 2
    assert BoundedPattern("succ?ess rate").search("Success rate improved")
    try:
        BoundedPattern("a" * 400)
    except ValueError as exc:
        assert "longer than" in str(exc)


def _aggregate_store(tmp_path):
    import json

    from harness_corpus.store import CorpusStore

    rows = {
        "papers.jsonl": [
            {"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025, "dblp_iri": "https://dblp.org/rec/x/vpp"},
            {"paper_id": "p6", "title": "FOCUS: object-centric world models", "year": 2024, "dblp_iri": None},
            {"paper_id": "p7", "title": "FOCUS: Object-Centric World Models", "year": 2025, "dblp_iri": None},
            {"paper_id": "p9", "title": "Dream to Manipulate", "year": 2014, "dblp_iri": None},
        ],
        "passages.jsonl": [],
        "facts.jsonl": [
            *({"fact_id": f"{p}-y", "subject": p, "predicate": "year", "object": y, "source_id": f"{p}-metadata"}
              for p, y in (("p4", "2025"), ("p6", "2024"), ("p7", "2025"), ("p9", "2014"))),
            *({"fact_id": f"{p}-{a}", "subject": p, "predicate": "author", "object": a, "source_id": f"{p}-metadata"}
              for p, a in (("p4", "Ada"), ("p6", "Ada"), ("p7", "Ada"), ("p9", "Ada"), ("p9", "Bo"))),
        ],
        "references.jsonl": [
            {"reference_id": "p9-ref-1", "paper_id": "p9", "resolved_paper_id": "p4"},
            {"reference_id": "p6-ref-1", "paper_id": "p6", "resolved_paper_id": "p4"},
        ],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def test_paper_ids_accept_documented_formats_and_reject_unknown_ones(tmp_path) -> None:
    from harness_corpus.store import UnknownIdentifier

    store = _aggregate_store(tmp_path)
    assert store.paper_key("p4") == store.paper_key("https://dblp.org/rec/x/vpp") == "p4"
    assert store.paper_key("p9-metadata") == store.paper_key("https://harness.local/resource/p9") == "p9"
    assert store.facts("https://dblp.org/rec/x/vpp", predicate="year")["items"][0]["object"] == "2025"
    assert store.facts("p4", predicate="authoredBy")["total"] == 1          # alias of author
    try:
        store.facts("https://dblp.org/rec/unknown")
    except UnknownIdentifier as exc:
        assert "unknown paper_id" in str(exc) and "p4" in str(exc)
    else:
        raise AssertionError("unknown id returned an empty result")
    try:
        store.facts("p4", predicate="citations")
    except UnknownIdentifier as exc:
        assert "Available predicates" in str(exc)
    else:
        raise AssertionError("unknown predicate returned an empty result")


def test_aggregates_are_exhaustive_and_cite_their_facts(tmp_path) -> None:
    store = _aggregate_store(tmp_path)
    authors = store.aggregate("count_papers_by_author", top_k=1)
    assert [(i["author"], i["count"]) for i in authors["items"]] == [("Ada", 3)]   # p6/p7 same work
    assert authors["computed"] and authors["coverage"]["exhaustive"]
    assert set(authors["items"][0]["fact_ids"]) == {"p4-Ada", "p6-Ada", "p9-Ada"}
    earliest = store.aggregate("min", predicate="year")
    assert earliest["items"][0]["value"] == "2014" and earliest["items"][0]["paper_ids"] == ["p9"]
    by_year = store.aggregate("count_by_value", predicate="year")
    assert by_year["items"][0] == {**by_year["items"][0], "value": "2025", "count": 2}
    cited = store.aggregate("count_incoming_citations")
    assert cited["items"][0]["paper_id"] == "p4" and cited["items"][0]["count"] == 2
    by_author = store.aggregate("count_citations_by_author", top_k=1)
    assert by_author["items"][0]["author"] == "Ada" and by_author["items"][0]["count"] == 2
    table = store.metadata_table(["year"])
    assert table["total"] == 4 and {row["paper_id"] for row in table["items"]} == {"p4", "p6", "p7", "p9"}
