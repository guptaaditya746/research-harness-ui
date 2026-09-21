"""Resolve real DBLP publication IRIs; never fabricate a DBLP record."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from urllib.parse import quote

import httpx

API = "https://dblp.org/search/publ/api"
SPARQL_API = "https://sparql.dblp.org/sparql"
SCHEMA = "https://dblp.org/rdf/schema#"


class DBLPLookupError(RuntimeError):
    """DBLP did not return its documented API response."""


def normal_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def normal_doi(value: str) -> str:
    return value.casefold().removeprefix("https://doi.org/")


def _sparql(client: httpx.Client, query: str) -> list[dict]:
    response = client.post(
        SPARQL_API,
        content=query,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
        },
    )
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", ""):
        raise DBLPLookupError(
            f"DBLP SPARQL returned {response.headers.get('content-type', 'unknown content')} instead of JSON"
        )
    return response.json().get("results", {}).get("bindings", [])


def lookup_many(papers: list[dict], client: httpx.Client) -> dict[str, str]:
    """Resolve a corpus in two indexed SPARQL queries: DOI first, then exact title."""
    by_doi = {normal_doi(p.get("doi", "")): p["paper_id"] for p in papers if p.get("doi")}
    matches: dict[str, str] = {}
    doi_iris: set[str] = set()
    for doi in by_doi:
        escaped = quote(doi, safe="/:;()-.")
        doi_iris.update({f"https://doi.org/{escaped}", f"https://doi.org/{escaped.upper()}"})
    if doi_iris:
        values = " ".join(f"<{iri}>" for iri in sorted(doi_iris))
        query = f"""PREFIX dblp: <{SCHEMA}>
SELECT ?publ ?doi WHERE {{ VALUES ?doi {{ {values} }} ?publ dblp:doi ?doi. }}"""
        for row in _sparql(client, query):
            publication = row.get("publ", {}).get("value", "")
            doi = normal_doi(row.get("doi", {}).get("value", ""))
            if doi in by_doi and publication.startswith("https://dblp.org/rec/"):
                matches[by_doi[doi]] = publication

    unmatched = [p for p in papers if p["paper_id"] not in matches and p.get("title")]
    by_title = {normal_title(p["title"]): p["paper_id"] for p in unmatched}
    if by_title:
        title_variants = {p["title"].strip() for p in unmatched}
        title_variants.update(title.rstrip(".") + "." for title in list(title_variants))
        values = " ".join(json.dumps(title, ensure_ascii=False) for title in sorted(title_variants))
        query = f"""PREFIX dblp: <{SCHEMA}>
SELECT ?publ ?title WHERE {{ VALUES ?title {{ {values} }} ?publ dblp:title ?title. }}"""
        for row in _sparql(client, query):
            publication = row.get("publ", {}).get("value", "")
            title = normal_title(row.get("title", {}).get("value", ""))
            if title in by_title and publication.startswith("https://dblp.org/rec/"):
                matches[by_title[title]] = publication
    return matches


def lookup(doi: str, title: str, client: httpx.Client, cutoff: float = 0.9) -> str | None:
    """Legacy single-record search fallback for callers outside corpus ingest."""
    for query in [doi, title]:
        if not query:
            continue
        response = client.get(API, params={"q": query, "format": "json", "h": 10})
        response.raise_for_status()
        if "json" not in response.headers.get("content-type", ""):
            raise DBLPLookupError(
                f"DBLP returned {response.headers.get('content-type', 'unknown content')} instead of JSON"
            )
        hits = response.json().get("result", {}).get("hits", {}).get("hit", [])
        if isinstance(hits, dict):
            hits = [hits]
        for hit in hits:
            info = hit.get("info", {})
            candidate_doi = normal_doi(str(info.get("doi", "")))
            candidate_title = re.sub(r"<[^>]+>", "", str(info.get("title", "")))
            doi_match = bool(doi and candidate_doi == normal_doi(doi))
            title_match = bool(
                title
                and SequenceMatcher(
                    None, normal_title(title), normal_title(candidate_title)
                ).ratio()
                >= cutoff
            )
            url = str(info.get("url", ""))
            if (doi_match or title_match) and url.startswith("https://dblp.org/rec/"):
                return url
    return None
