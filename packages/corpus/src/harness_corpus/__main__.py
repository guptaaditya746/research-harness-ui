"""CLI for one-time PDF ingest."""

import argparse
import json
from pathlib import Path

from harness_config import settings

from harness_corpus.dense import build_dense_index
from harness_corpus.ingest import enrich_dblp, ingest, prune_unresolved_dblp


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a lexical research corpus")
    parser.add_argument("pdf_dir", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=settings().corpus_root or Path("corpus_data"))
    parser.add_argument("--grobid-url", default=settings().grobid_url)
    parser.add_argument("--skip-grobid", action="store_true")
    parser.add_argument("--skip-dblp", action="store_true")
    parser.add_argument("--dblp-only", action="store_true", help="enrich an existing corpus")
    parser.add_argument("--prune-unresolved-dblp", action="store_true")
    parser.add_argument("--embed", action="store_true",
                        help="build the dense passage index with HARNESS_MODEL_EMBEDDING (needs the model server)")
    parser.add_argument("--embed-batch", type=int, default=64)
    args = parser.parse_args()
    if args.embed:
        result = build_dense_index(args.root, batch_size=args.embed_batch)
    elif args.prune_unresolved_dblp:
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
