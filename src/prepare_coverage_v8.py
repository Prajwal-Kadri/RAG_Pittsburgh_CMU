"""Build a separate corpus from fetched publisher evidence; no answer keys."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .clean_corpus import clean_records
from .chunking import chunk_text
from .collect_sources_v8 import pdf_rows, publisher_blocks
from .common import read_jsonl, write_jsonl
from .source_quality import source_page


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/raw/pages_coverage_v7.jsonl")
    parser.add_argument("--sources", default="data/raw/pages_sources_v8.jsonl")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--source-labels", default="data/source_labels_v8.json")
    parser.add_argument("--out", default="data/raw/pages_coverage_v8.jsonl")
    parser.add_argument("--publisher-extracts", default="data/raw/publisher_extracts_v8.json")
    args = parser.parse_args()
    if Path(args.out).resolve() in {Path(args.base).resolve(), Path(args.sources).resolve()}:
        parser.error("Use a separate corpus output path")
    base, fetched = read_jsonl(args.base), read_jsonl(args.sources)
    extracts_path = Path(args.publisher_extracts)
    if extracts_path.exists():
        for item in json.loads(extracts_path.read_text()):
            if not item["text"].strip():
                raise ValueError("Empty cached publisher extract: " + item["url"])
            digest = hashlib.sha256(item["text"].encode()).hexdigest()
            fetched.append({"url": item["url"] + "#cached-publisher-extract",
                            "source_url": item["url"], "domain": urlsplit(item["url"]).netloc,
                            "source_sha256": digest, "collected_at": "2026-10-25",
                            "title": item["title"], "text": item["title"] + ". " + item["text"],
                            "extraction": "published_cached_excerpt",
                            "retrieval_method": item["retrieval_method"]})
    labels = json.loads(Path(args.source_labels).read_text())
    cache = Path(args.cache_dir)
    sources, enriched = {}, {}
    for record in fetched:
        parent = source_page(record["source_url"])
        if re.search(r"/errors/not-found|/404(?:/|$)", parent):
            continue
        sources[parent] = record
    for parent, first in sorted(sources.items()):
        label = labels.get(first["source_url"], first["title"])
        metadata = {k: first[k] for k in ("source_url", "domain", "source_sha256", "collected_at")}
        for record in fetched:
            if source_page(record["source_url"]) != parent or record["source_sha256"] != first["source_sha256"]:
                continue
            # Keep the general budget/forecast sections; personnel listings
            # hundreds of pages later are outside this collection's scope.
            if "operating-budget" in parent and record.get("page", 0) > 65:
                continue
            text = record["text"]
            if label != record["title"] and text.startswith(record["title"]):
                text = label + text[len(record["title"]):]
            if "lorem ipsum" in text.lower():
                continue
            if record.get("extraction") == "published_badge" and "Image URL: data:" in text:
                continue
            enriched[record["url"]] = {**record, "title": label, "text": text}
        html = cache / (first["source_sha256"] + ".html")
        if html.exists():
            for record in publisher_blocks(html.read_bytes(), metadata, label):
                if "lorem ipsum" not in record["text"].lower():
                    enriched[record["url"]] = record
        pdf = cache / (first["source_sha256"] + ".pdf")
        if pdf.exists():
            for record in pdf_rows(pdf, metadata, label, max_pages=65 if "operating-budget" in parent else None):
                if "operating-budget" not in parent or record["page"] <= 65:
                    enriched[record["url"]] = record
    # A refreshed publisher page replaces its old fragments as a unit.
    retained = [r for r in base if source_page(r.get("source_url", r["url"])) not in sources]
    result, counts = clean_records(retained + list(enriched.values()))
    write_jsonl(result, args.out)
    companion = Path(args.out).with_name("pages_sources_v8_enriched.jsonl")
    write_jsonl(enriched.values(), companion)
    # Art catalog fields/numbers must survive byte-for-byte in every record.
    catalog = {r["url"]: r["text"] for r in base if r["domain"] == "collection.carnegieart.org"}
    current = {r["url"]: r["text"] for r in result if r["domain"] == "collection.carnegieart.org"}
    if catalog != current:
        raise ValueError("Catalog records changed during source expansion")
    manifest = {
        "version": "coverage_v8", "counts": counts, "source_documents": len(sources),
        "new_source_records": len(enriched), "replaced_baseline_records": len(base) - len(retained),
        "catalog_records_preserved": len(catalog),
        "index_settings": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                           "max_words": 180, "overlap_sentences": 1,
                           "num_chunks": sum(len(chunk_text(r["text"], 180, 1)) for r in result)},
        "question_files": {name: sha(name) for name in
                           ("data/public_evaluation_data.json", "data/leaderboard_queries.json")
                           if Path(name).is_file()},
        "code_files": {name: sha(name) for name in
                       ("src/collect_sources_v8.py", "src/prepare_coverage_v8.py", "src/retrieval_v8.py")},
        "files": {args.base: sha(args.base), args.sources: sha(args.sources),
                  args.source_labels: sha(args.source_labels), args.out: sha(args.out)},
        "sources": [{"url": parent, "sha256": r["source_sha256"], "collected_at": r["collected_at"]}
                    for parent, r in sorted(sources.items())],
        "extractions": dict(Counter(r.get("extraction", "legacy") for r in result)),
        "note": "Source discovery was informed by leaderboard questions. This is development-informed collection, not untouched held-out evaluation. No gold answers or model-written facts were inserted. Publisher text, literal page metadata, and labelled PDF rows only; factual correctness and answer metrics require separate validation."
    }
    if extracts_path.exists():
        manifest["files"][str(extracts_path)] = sha(extracts_path)
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "source_documents", "new_source_records", "replaced_baseline_records", "catalog_records_preserved")}, indent=2))


if __name__ == "__main__":
    main()
