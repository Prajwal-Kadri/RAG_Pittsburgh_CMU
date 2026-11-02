"""Create a separate cleaned corpus; preserve source text and hashes in inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .common import read_jsonl, write_jsonl
from .source_quality import clean_text, record_junk


def clean_records(records):
    result, seen, counts = [], set(), Counter()
    for record in records:
        counts["input_records"] += 1
        # Catalog fields and social-link evidence already have explicit structure.
        text = record["text"] if record.get("extraction") == "published_links" or record["domain"] == "collection.carnegieart.org" else clean_text(record["text"])
        if record_junk(record) or len(text.strip()) < 40:
            counts["dropped_noise_records"] += 1
            continue
        key = " ".join(text.casefold().split())
        if key in seen:
            counts["dropped_exact_duplicates"] += 1
            continue
        seen.add(key)
        changed = text != record["text"]
        counts["trimmed_records"] += int(changed)
        title = record.get("title", "").strip() or (record["domain"] + " " + urlsplit(record["url"]).path)
        result.append({**record, "text": text, "title": title, "source_record_url": record["url"],
                       "representation_version": "source_clean_v7"})
    counts["output_records"] = len(result)
    return result, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    # Refreshed source copies replace older records with the same URL.
    merged = {}
    hashes = {}
    for filename in args.inputs:
        path = Path(filename)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in read_jsonl(path):
            merged[row["url"]] = row
    cleaned, counts = clean_records(list(merged.values()))
    write_jsonl(cleaned, args.out)
    manifest = {"input_sha256": hashes, "counts": counts,
                "output_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
                "note": "Source cleanup only. No evaluation references, generated answers, or question IDs are used."}
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
