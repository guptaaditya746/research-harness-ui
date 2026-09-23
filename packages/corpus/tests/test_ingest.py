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
    }
    response = asyncio.run(mcp.call_tool("citations", {"paper_id": "p17"}))
    assert "p2" in str(response)
    assert any(
        row["passage_id"] == "p17-s1-p1"
        for row in store.expand("p17-s1-p1", 1)["items"]
    )
    assert store.results("predict")["total"] == 0  # results contain numeric claims only
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
