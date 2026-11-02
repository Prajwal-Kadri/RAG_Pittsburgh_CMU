from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .common import read_jsonl, tokenize


class Retriever:
    def __init__(self, artifact_dir: str | Path):
        root = Path(artifact_dir)
        self.chunks = read_jsonl(root / "chunks.jsonl")
        self.bm25 = BM25Okapi(json.loads((root / "bm25_tokens.json").read_text(encoding="utf-8")))
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.encoder = SentenceTransformer(config["embedding_model"])
        self.index = faiss.read_index(str(root / "dense.faiss"))

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in ids]

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(vector, k)
        return [(int(i), float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]

    def hybrid_rrf(self, query: str, k: int, candidate_k: int = 30, rrf_k: int = 60) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranking in (self.sparse(query, candidate_k), self.dense(query, candidate_k)):
            for rank, (chunk_id, _) in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]

    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> list[dict]:
        if mode == "sparse":
            ranked = self.sparse(query, k)
        elif mode == "dense":
            ranked = self.dense(query, k)
        elif mode == "hybrid":
            ranked = self.hybrid_rrf(query, k)
        else:
            raise ValueError("mode must be sparse, dense, or hybrid")
        return [{**self.chunks[chunk_id], "score": score} for chunk_id, score in ranked]

