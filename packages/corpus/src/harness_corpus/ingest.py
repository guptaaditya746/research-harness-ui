"""Build a queryable corpus from GROBID TEI and explicit PDF metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import bm25s
import httpx
from grobid_client.grobid_client import GrobidClient

from harness_corpus.dblp import DBLPLookupError
from harness_corpus.dblp import lookup_many as lookup_dblp_many
from harness_corpus.tei import clean, paper_id, parse_tei


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def normal_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def local_identifier(pdf: Path) -> str:
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"local:paper/{digest.hexdigest()[:16]}"


def resolve_references(papers: list[dict], references: list[dict], cutoff: float = 0.9) -> None:
    by_doi = {p["doi"]: p["paper_id"] for p in papers if p["doi"]}
    titled = [(normal_title(p["title"]), p["paper_id"]) for p in papers if p["title"]]
    for ref in references:
        target = by_doi.get(ref["doi"]) if ref["doi"] else None
        if target is None and ref["title"]:
            title = normal_title(ref["title"])
            candidates = [
                (SequenceMatcher(None, title, other).ratio(), pid) for other, pid in titled
            ]
            if candidates:
                score, candidate = max(candidates)
                if score >= cutoff:
                    target = candidate
        ref["resolved_paper_id"] = target


def structural_facts(papers: list[dict], references: list[dict]) -> list[dict]:
    facts: list[dict] = []
    for paper in papers:
        pid = paper["paper_id"]
        for predicate, values in (
            ("title", [paper["title"]]),
            ("author", paper["authors"]),
            ("year", [paper["year"]]),
            ("venue", [paper["venue"]]),
        ):
            for value in values:
                if value is not None and value != "":
                    facts.append(
                        {
                            "fact_id": f"{pid}-metadata-{len(facts)}",
                            "subject": pid,
                            "predicate": predicate,
                            "object": value,
                            "source": "metadata",
                            "source_id": f"{pid}-metadata",
                        }
                    )
    for ref in references:
        facts.append(
            {
                "fact_id": f"{ref['reference_id']}-cites",
                "subject": ref["paper_id"],
                "predicate": "cites",
                "object": ref["resolved_paper_id"] or ref["reference_id"],
                "source": "reference_list",
                "source_id": ref["reference_id"],
                "in_corpus": ref["resolved_paper_id"] is not None,
            }
        )
    return facts


def write_outline(root: Path, paper: dict, passages: list[dict]) -> None:
    headings: dict[str, str] = {}
    for passage in passages:
        headings.setdefault(passage["section_path"], passage["text"].split(". ")[0] + ".")
    lines = [f"# {paper['title']}", "", "## Abstract", paper["abstract"] or "(none)", ""]
    for heading, first in headings.items():
        lines.extend([f"## {heading}", first, ""])
    target = root / "outlines" / f"{paper['paper_id']}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))


def from_docling(pdf: Path, pid: str) -> tuple[dict, list[dict], list[dict]]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(f"GROBID could not parse {pdf}; install docling for fallback") from exc
    markdown = DocumentConverter().convert(pdf).document.export_to_markdown()
    chunks = [clean(chunk) for chunk in re.split(r"\n\s*\n", markdown) if clean(chunk)]
    title = chunks[0].lstrip("# ") if chunks else pdf.stem
    paper = {
        "paper_id": pid,
        "dblp_iri": None,
        "title": title,
        "authors": [],
        "year": None,
        "venue": "",
        "doi": "",
        "abstract": "",
    }
    passages = []
    offset = 0
    heading = "Body"
    for chunk in chunks[1:]:
        if chunk.startswith("#"):
            heading = chunk.lstrip("# ")
            continue
        passages.append(
            {
                "passage_id": f"{pid}-s1-p{len(passages) + 1}",
                "paper_id": pid,
                "section_path": heading,
                "para_idx": len(passages),
                "text": chunk,
                "char_start": offset,
                "char_end": offset + len(chunk),
            }
        )
        offset += len(chunk) + 1
    return paper, passages, []


def ingest(
    pdf_dir: Path,
    root: Path,
    grobid_url: str,
    *,
    skip_grobid: bool = False,
    skip_dblp: bool = False,
) -> dict:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"no PDFs found in {pdf_dir}")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".ingesting"
    if marker.exists():
        raise RuntimeError("an ingest is already in progress")
    marker.write_text(str(os.getpid()))
    try:
        tei_dir = root / ".cache" / "tei"
        tei_dir.mkdir(parents=True, exist_ok=True)
        if not skip_grobid:
            GrobidClient(grobid_server=grobid_url).process(
                "processFulltextDocument",
                str(pdf_dir),
                output=str(tei_dir),
                n=2,
                consolidate_citations=False,
                force=False,
                skip_errors=True,
            )
        metadata = {}
        manifest = pdf_dir / "manifest.tsv"
        if manifest.exists():
            with manifest.open(newline="") as handle:
                metadata = {row["filename"]: row for row in csv.DictReader(handle, delimiter="\t")}
        papers: list[dict] = []
        passages: list[dict] = []
        references: list[dict] = []
        fallback: list[str] = []
        local_ids: set[str] = set()
        for pdf in pdfs:
            pid = paper_id(pdf)
            tei = tei_dir / f"{pdf.stem}.grobid.tei.xml"
            try:
                if not tei.exists():
                    raise FileNotFoundError(tei)
                paper, extracted, refs = parse_tei(tei.read_bytes(), pid)
                if not extracted:
                    raise ValueError("no passages in TEI")
            except (FileNotFoundError, ValueError):
                paper, extracted, refs = from_docling(pdf, pid)
                fallback.append(pid)
            row = metadata.get(pdf.name, {})
            paper["title"] = paper["title"] or row.get("title", "")
            paper["doi"] = paper["doi"] or row.get("doi", "").lower().removeprefix(
                "https://doi.org/"
            )
            paper["year"] = paper["year"] or (int(row["year"]) if row.get("year") else None)
            local_id = local_identifier(pdf)
            if local_id in local_ids:
                raise ValueError(f"duplicate PDF content gives the same local identifier: {pdf}")
            local_ids.add(local_id)
            paper["identifier"] = local_id
            paper["identifier_source"] = "local"
            paper["dblp_status"] = "unresolved"
            papers.append(paper)
            passages.extend(extracted)
            references.extend(refs)
            write_outline(root, paper, extracted)
        dblp_errors: list[str] = []
        if not skip_dblp:
            try:
                with httpx.Client(timeout=60) as client:
                    dblp_matches = lookup_dblp_many(papers, client)
                for paper in papers:
                    iri = dblp_matches.get(paper["paper_id"])
                    if iri:
                        paper["dblp_iri"] = iri
                        paper["identifier"] = iri.replace("https://dblp.org/rec/", "dblp:", 1)
                        paper["identifier_source"] = "dblp"
                        paper["dblp_status"] = "verified"
            except (httpx.HTTPError, DBLPLookupError) as exc:
                dblp_errors.append(f"batch: {exc}")
        resolve_references(papers, references)
        write_jsonl(root / "papers.jsonl", papers)
        write_jsonl(root / "passages.jsonl", passages)
        write_jsonl(root / "references.jsonl", references)
        write_jsonl(root / "facts.jsonl", structural_facts(papers, references))
        (root / "tables").mkdir(exist_ok=True)
        index_dir = root / "index" / "bm25"
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever = bm25s.BM25()
        retriever.index(
            bm25s.tokenize([p["text"] for p in passages], stopwords="en", show_progress=False),
            show_progress=False,
        )
        retriever.save(
            str(index_dir),
            corpus=[{"passage_id": p["passage_id"]} for p in passages],
            show_progress=False,
        )
        report = {
            "papers": len(papers),
            "passages": len(passages),
            "references": len(references),
            "fallback_papers": fallback,
            "dblp_errors": dblp_errors,
            "unresolved_dblp": [p["paper_id"] for p in papers if not p["dblp_iri"]],
        }
        (root / "ingest_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        marker.unlink(missing_ok=True)


def enrich_dblp(root: Path) -> dict:
    """Verify DBLP identifiers for an existing corpus without reprocessing PDFs."""
    papers_path = root / "papers.jsonl"
    if not papers_path.exists():
        raise ValueError(f"missing corpus papers: {papers_path}")
    papers = [json.loads(line) for line in papers_path.read_text().splitlines() if line]
    with httpx.Client(timeout=60) as client:
        matches = lookup_dblp_many(papers, client)
    for paper in papers:
        iri = matches.get(paper["paper_id"])
        if iri:
            paper["dblp_iri"] = iri
            paper["identifier"] = iri.replace("https://dblp.org/rec/", "dblp:", 1)
            paper["identifier_source"] = "dblp"
            paper["dblp_status"] = "verified"
    write_jsonl(papers_path, papers)
    report_path = root / "ingest_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["dblp_errors"] = []
    report["verified_dblp"] = sorted(matches)
    report["unresolved_dblp"] = [p["paper_id"] for p in papers if p["dblp_status"] != "verified"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {
        "papers": len(papers),
        "verified": len(matches),
        "unresolved": len(report["unresolved_dblp"]),
    }


def prune_unresolved_dblp(root: Path) -> dict:
    """Remove unresolved papers from built artifacts and rebuild dependent indexes."""
    papers = [json.loads(line) for line in (root / "papers.jsonl").read_text().splitlines() if line]
    removed = {p["paper_id"] for p in papers if p.get("dblp_status") != "verified"}
    kept_papers = [p for p in papers if p["paper_id"] not in removed]
    passages = [
        json.loads(line) for line in (root / "passages.jsonl").read_text().splitlines() if line
    ]
    kept_passages = [p for p in passages if p["paper_id"] not in removed]
    references = [
        json.loads(line) for line in (root / "references.jsonl").read_text().splitlines() if line
    ]
    kept_references = [r for r in references if r["paper_id"] not in removed]
    for reference in kept_references:
        if reference.get("resolved_paper_id") in removed:
            reference["resolved_paper_id"] = None

    write_jsonl(root / "papers.jsonl", kept_papers)
    write_jsonl(root / "passages.jsonl", kept_passages)
    write_jsonl(root / "references.jsonl", kept_references)
    write_jsonl(root / "facts.jsonl", structural_facts(kept_papers, kept_references))
    for pid in removed:
        (root / "outlines" / f"{pid}.md").unlink(missing_ok=True)

    index_dir = root / "index" / "bm25"
    if index_dir.exists():
        import shutil

        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)
    retriever = bm25s.BM25()
    retriever.index(
        bm25s.tokenize([p["text"] for p in kept_passages], stopwords="en", show_progress=False),
        show_progress=False,
    )
    retriever.save(
        str(index_dir),
        corpus=[{"passage_id": p["passage_id"]} for p in kept_passages],
        show_progress=False,
    )

    report_path = root / "ingest_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(
        {
            "papers": len(kept_papers),
            "passages": len(kept_passages),
            "references": len(kept_references),
            "unresolved_dblp": [],
            "verified_dblp": sorted(p["paper_id"] for p in kept_papers),
            "excluded_unresolved_dblp": sorted(removed),
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {
        "removed": sorted(removed),
        "papers": len(kept_papers),
        "passages": len(kept_passages),
        "references": len(kept_references),
    }
