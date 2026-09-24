"""Stage 150 ACL 2025 and 150 EMNLP 2025 PDFs alongside the current corpus.

The source of truth is the ACL Anthology's official collection XML. This
command only prepares an input directory; it never replaces corpus_data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from harness_corpus.tei import paper_id

COLLECTIONS = (
    ("2025.acl", "long", "ACL 2025"),
    ("2025.emnlp", "main", "EMNLP 2025"),
)


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "research-harness-corpus-ingest/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def selected_papers(xml: bytes, collection: str, volume: str, venue: str, count: int) -> list[dict]:
    root = ElementTree.fromstring(xml)
    if root.attrib.get("id") != collection:
        raise ValueError(f"unexpected collection: {root.attrib.get('id')}")
    selected_volume = root.find(f"./volume[@id='{volume}']")
    if selected_volume is None:
        raise ValueError(f"missing volume {collection}-{volume}")
    papers = sorted(
        (paper for paper in selected_volume.findall("paper") if paper.find("pdf") is not None),
        key=lambda paper: int(paper.attrib["id"]),
    )
    if len(papers) < count:
        raise ValueError(f"{collection}-{volume} has only {len(papers)} PDF papers")
    # Cover the full proceedings rather than taking only its first pages.
    picked = [papers[((2 * i + 1) * len(papers)) // (2 * count)] for i in range(count)]
    result = []
    for paper in picked:
        anthology_id = f"{collection}-{volume}.{paper.attrib['id']}"
        title_element = paper.find("title")
        title = "".join(title_element.itertext()).strip() if title_element is not None else ""
        authors = [
            " ".join(filter(None, (author.findtext("first"), author.findtext("last"))))
            for author in paper.findall("author")
        ]
        result.append({
            "filename": f"{anthology_id}.pdf",
            "anthology_id": anthology_id,
            "title": " ".join(title.split()),
            "authors": json.dumps(authors, ensure_ascii=False),
            "doi": paper.findtext("doi", default=""),
            "year": "2025",
            "venue": venue,
            "source_url": f"https://aclanthology.org/{anthology_id}/",
            "pdf_url": f"https://aclanthology.org/{anthology_id}.pdf",
        })
    if len({row["anthology_id"] for row in result}) != count:
        raise ValueError("selection repeated an Anthology ID")
    return result


def stage_pdf(row: dict, destination: Path) -> tuple[str, str]:
    path = destination / row["filename"]
    if path.exists():
        if path.stat().st_size > 10_000 and path.open("rb").read(5) == b"%PDF-":
            return row["anthology_id"], "cached"
        raise ValueError(f"existing file is not a usable PDF: {path}")
    temporary = path.with_suffix(".pdf.part")
    for attempt in range(4):
        try:
            payload = fetch(row["pdf_url"])
            if len(payload) < 10_000 or not payload.startswith(b"%PDF-"):
                raise ValueError(f"invalid PDF response for {row['anthology_id']}")
            with temporary.open("wb") as output:
                output.write(payload)
            os.replace(temporary, path)
            return row["anthology_id"], "downloaded"
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-pdfs", type=Path, required=True)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--per-venue", type=int, default=150)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == args.existing_pdfs.resolve() or destination == args.existing_corpus.resolve():
        raise ValueError("destination must be a separate staging directory")
    destination.mkdir(parents=True, exist_ok=True)

    existing = [json.loads(line) for line in (args.existing_corpus / "papers.jsonl").read_text().splitlines()]
    source_by_id = {
        paper_id(path): path for path in args.existing_pdfs.glob("*.pdf")
    }
    if len(source_by_id) != len(list(args.existing_pdfs.glob("*.pdf"))):
        raise ValueError("source PDF filenames have colliding paper IDs")
    for paper in existing:
        source = source_by_id.get(paper["paper_id"])
        if source is None:
            raise ValueError(f"no PDF for current corpus paper {paper['paper_id']}")
        target = destination / source.name
        if target.exists():
            if not target.samefile(source):
                raise ValueError(f"staging filename already exists: {target}")
        else:
            os.link(source, target)

    rows = []
    metadata_hashes = {}
    for collection, volume, venue in COLLECTIONS:
        url = f"https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{collection}.xml"
        xml = fetch(url)
        metadata_hashes[collection] = hashlib.sha256(xml).hexdigest()
        rows.extend(selected_papers(xml, collection, volume, venue, args.per_venue))

    if len({paper_id(destination / row["filename"]) for row in rows}) != len(rows):
        raise ValueError("selected filenames have colliding corpus IDs")
    if {paper_id(destination / row["filename"]) for row in rows} & {paper["paper_id"] for paper in existing}:
        raise ValueError("selected papers collide with current corpus IDs")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(stage_pdf, row, destination) for row in rows]
        for completed, future in enumerate(as_completed(futures), 1):
            anthology_id, result = future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(f"{completed}/{len(futures)} PDFs ready ({anthology_id}: {result})", flush=True)

    columns = ("filename", "anthology_id", "title", "authors", "doi", "year", "venue", "source_url", "pdf_url")
    with (destination / "manifest.tsv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (destination / "source_report.json").write_text(json.dumps({
        "existing_papers": len(existing),
        "new_papers": len(rows),
        "pdfs": len(list(destination.glob("*.pdf"))),
        "selection": "evenly spaced across numbered paper records in each proceedings volume",
        "metadata_sha256": metadata_hashes,
    }, indent=2) + "\n")
    print(f"staged {len(existing)} existing + {len(rows)} new PDFs in {destination}")


if __name__ == "__main__":
    main()
