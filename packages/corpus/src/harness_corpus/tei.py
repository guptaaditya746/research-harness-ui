"""Parse GROBID TEI into stable paper, passage, and reference records."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"t": "http://www.tei-c.org/ns/1.0"}


def clean(value: str | None) -> str:
    return " ".join((value or "").split())


def node_text(node: ET.Element | None) -> str:
    return clean(" ".join(node.itertext())) if node is not None else ""


def paper_id(path: Path) -> str:
    stem = path.name.removesuffix(".tei.xml").removesuffix(".pdf")
    return (
        f"p{int(stem)}" if stem.isdigit() else re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    )


def parse_tei(xml: bytes, pid: str) -> tuple[dict, list[dict], list[dict]]:
    root = ET.fromstring(xml)
    header = root.find(".//t:teiHeader/t:fileDesc/t:sourceDesc/t:biblStruct", NS)
    if header is None:
        header = root.find(".//t:teiHeader/t:fileDesc/t:titleStmt", NS)
    title = node_text(header.find(".//t:title[@level='a']", NS)) if header is not None else ""
    if not title and header is not None:
        title = node_text(header.find(".//t:title", NS))
    authors = []
    affiliation_only_authors = 0
    if header is not None:
        for author in header.findall(".//t:author", NS):
            # GROBID may emit an affiliation-only ``author`` node when it
            # cannot associate an institution with a person.  Falling back
            # to the complete node text turns institutions and countries
            # into fake authors.  A DBLP author fact requires a person name.
            person = author.find("t:persName", NS)
            name = node_text(person)
            if name:
                authors.append(name)
            elif author.find("t:affiliation", NS) is not None:
                affiliation_only_authors += 1
    doi = ""
    if header is not None:
        doi = node_text(header.find(".//t:idno[@type='DOI']", NS))
    date = root.find(".//t:teiHeader//t:date", NS)
    year_match = re.search(
        r"(?:19|20)\d{2}",
        (date.get("when", "") if date is not None else "") + " " + node_text(date),
    )
    year = int(year_match.group()) if year_match else None
    venue = node_text(root.find(".//t:teiHeader//t:title[@level='j']", NS))
    abstract = node_text(root.find(".//t:profileDesc/t:abstract", NS))
    paper = {
        "paper_id": pid,
        "dblp_iri": None,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi.lower().removeprefix("https://doi.org/"),
        "abstract": abstract,
        "excluded_affiliation_authors": affiliation_only_authors,
    }
    passages = []
    if abstract:
        passages.append(
            {
                "passage_id": f"{pid}-abstract",
                "paper_id": pid,
                "section_path": "Abstract",
                "para_idx": 0,
                "text": abstract,
                "char_start": 0,
                "char_end": len(abstract),
            }
        )
    offset = len(abstract) + 1 if abstract else 0
    body = root.find(".//t:text/t:body", NS)
    if body is not None:
        for section_idx, section in enumerate(body.findall(".//t:div", NS)):
            heading = node_text(section.find("t:head", NS)) or f"Section {section_idx + 1}"
            for para_idx, para in enumerate(section.findall("t:p", NS)):
                value = node_text(para)
                if not value:
                    continue
                passages.append(
                    {
                        "passage_id": f"{pid}-s{section_idx + 1}-p{para_idx + 1}",
                        "paper_id": pid,
                        "section_path": heading,
                        "para_idx": para_idx,
                        "text": value,
                        "char_start": offset,
                        "char_end": offset + len(value),
                    }
                )
                offset += len(value) + 1
        if not any(p["section_path"] != "Abstract" for p in passages):
            for para_idx, para in enumerate(body.findall(".//t:p", NS)):
                value = node_text(para)
                if value:
                    passages.append(
                        {
                            "passage_id": f"{pid}-s1-p{para_idx + 1}",
                            "paper_id": pid,
                            "section_path": "Body",
                            "para_idx": para_idx,
                            "text": value,
                            "char_start": offset,
                            "char_end": offset + len(value),
                        }
                    )
                    offset += len(value) + 1
    references = []
    for idx, item in enumerate(root.findall(".//t:listBibl/t:biblStruct", NS)):
        ref_title = node_text(item.find(".//t:title[@level='a']", NS)) or node_text(
            item.find(".//t:title", NS)
        )
        ref_doi = (
            node_text(item.find(".//t:idno[@type='DOI']", NS))
            .lower()
            .removeprefix("https://doi.org/")
        )
        references.append(
            {
                "reference_id": f"{pid}-ref-{idx + 1}",
                "paper_id": pid,
                "title": ref_title,
                "doi": ref_doi,
                "resolved_paper_id": None,
                "raw": node_text(item),
            }
        )
    return paper, passages, references
