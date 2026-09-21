import httpx
from harness_corpus.dblp import lookup, lookup_many


def test_lookup_accepts_only_verified_dblp_record() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"hits": {"hit": [
            {"info": {"title": "Wrong paper", "doi": "10.1/wrong", "url": "https://dblp.org/rec/bad"}},
            {"info": {"title": "World Models", "doi": "10.1/world", "url": "https://dblp.org/rec/journals/test/world"}},
        ]}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert lookup("10.1/world", "World Models", client) == "https://dblp.org/rec/journals/test/world"
        assert lookup("10.1/missing", "Missing Paper", client) is None


def test_lookup_many_uses_doi_then_exact_title() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.content.decode()
        if "VALUES ?doi" in query:
            return httpx.Response(200, json={"results": {"bindings": [{
                "publ": {"type": "uri", "value": "https://dblp.org/rec/journals/corr/abs-2301-05832"},
                "doi": {"type": "uri", "value": "https://doi.org/10.48550/ARXIV.2301.05832"},
            }]}})
        return httpx.Response(200, json={"results": {"bindings": [{
            "publ": {"type": "uri", "value": "https://dblp.org/rec/conf/test/Other24"},
            "title": {"type": "literal", "value": "Other Paper."},
        }]}})

    papers = [
        {"paper_id": "p2", "doi": "10.48550/arxiv.2301.05832", "title": "World Models"},
        {"paper_id": "p3", "doi": "10.1/not-in-dblp", "title": "Other Paper"},
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert lookup_many(papers, client) == {
            "p2": "https://dblp.org/rec/journals/corr/abs-2301-05832",
            "p3": "https://dblp.org/rec/conf/test/Other24",
        }
