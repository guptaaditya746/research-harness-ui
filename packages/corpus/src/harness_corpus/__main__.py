"""CLI for one-time PDF ingest."""

import argparse
import json
import os
from pathlib import Path

from harness_corpus.ingest import enrich_dblp, ingest, prune_unresolved_dblp


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a lexical research corpus")
    parser.add_argument("pdf_dir", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=Path(os.getenv("CORPUS_ROOT", "corpus_data")))
    parser.add_argument("--grobid-url", default=os.getenv("GROBID_URL", "http://127.0.0.1:8070"))
    parser.add_argument("--skip-grobid", action="store_true")
    parser.add_argument("--skip-dblp", action="store_true")
    parser.add_argument("--dblp-only", action="store_true", help="enrich an existing corpus")
    parser.add_argument("--prune-unresolved-dblp", action="store_true")
    args = parser.parse_args()
    if args.prune_unresolved_dblp:
        result = prune_unresolved_dblp(args.root)
    elif args.dblp_only:
        result = enrich_dblp(args.root)
    else:
        if args.pdf_dir is None:
            parser.error("pdf_dir is required unless --dblp-only is used")
        result = ingest(
            args.pdf_dir,
            args.root,
            args.grobid_url,
            skip_grobid=args.skip_grobid,
            skip_dblp=args.skip_dblp,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
