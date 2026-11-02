"""Build a v7 index with source titles attached to each evidence chunk."""
import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from .common import read_jsonl, write_jsonl, tokenize
from .chunking import chunk_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", help="Optional embedding device, e.g. cpu for local checks")
    args = parser.parse_args()
    chunks = []
    for page_id, page in enumerate(read_jsonl(args.input)):
        title = page["title"].strip()
        for chunk_id, text in enumerate(chunk_text(page["text"], 180, 1)):
            # Same sentence chunker and body size; the source title supplies
            # entity context to chunks that start midway through a source page.
            if title and not text.casefold().startswith(title.casefold()):
                text = title + ". " + text
            chunks.append({"id": len(chunks), "page_id": page_id, "chunk_id": chunk_id,
                           "url": page["url"], "title": title, "text": text,
                           "source_url": page.get("source_url", page["url"].split("#")[0]),
                           "extraction": page.get("extraction", "legacy_text")})
    if not chunks:
        raise ValueError("No usable chunks")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, root / "chunks.jsonl")
    encoder = SentenceTransformer(args.model, device=args.device)
    vectors = encoder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(root / "dense.faiss"))
    (root / "bm25_tokens.json").write_text(json.dumps([tokenize(c["text"]) for c in chunks]))
    config = {"embedding_model": args.model, "num_chunks": len(chunks), "max_words": 180,
              "overlap_sentences": 1, "representation": "source_title_v7"}
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Indexed {len(chunks)} title-context chunks in {root}")


if __name__ == "__main__":
    main()
