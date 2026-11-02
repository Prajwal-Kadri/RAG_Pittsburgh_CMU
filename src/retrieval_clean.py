"""Opt-in v7 retrieval: content terms, evidence diversity, and noise filtering.

Uses the same MiniLM encoder, BM25 index, and 30-candidate RRF pools. No gold
answers, evaluation IDs, generation, or learned reranking are used.
"""
import hashlib
from pathlib import Path

import numpy as np

from .common import tokenize
from .retrieval import Retriever
from .source_quality import candidate_junk, near_duplicate, source_page


STOP = set("a an the what which who whom when where how why is are was were be been being do does did will would can could should of in on at to from for with by and or it its this that these those tell give please".split())


def select_candidates(candidates, k=5, page_cap=2):
    selected, deferred, counts = [], [], {}
    for candidate in candidates:
        if any(near_duplicate(candidate["text"], old["text"]) for old in selected):
            continue
        parent = source_page(candidate.get("source_url", candidate["url"]))
        if counts.get(parent, 0) >= page_cap:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        counts[parent] = counts.get(parent, 0) + 1
        if len(selected) == k:
            return selected
    # Backfill distinct evidence when only a few source pages are available.
    for candidate in deferred:
        if not any(near_duplicate(candidate["text"], old["text"]) for old in selected):
            selected.append(candidate)
            if len(selected) == k:
                break
    return selected


class CleanRetriever(Retriever):
    def __init__(self, artifact_dir, device=None):
        super().__init__(artifact_dir)
        if device:
            self.encoder.to(device)
        from . import source_quality
        self.policy_config = {"version": "clean-v1", "candidate_k": 30, "rrf_k": 60,
                              "backfill_candidate_k": [30, 120, "all"],
                              "page_cap": 2, "near_duplicate_threshold": 0.9,
                              "hybrid_rank_weight": 0.4, "hybrid_content_weight": 0.6,
                              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "quality_sha256": hashlib.sha256(Path(source_quality.__file__).read_bytes()).hexdigest()}

    def sparse(self, query, k):
        content = [w for w in tokenize(query) if w not in STOP]
        scores = self.bm25.get_scores(content or tokenize(query))
        return [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:k]]

    def _candidates(self, query, mode, pool):
        if mode == "hybrid":
            ranked = self.hybrid_rrf(query, 2 * pool, candidate_k=pool)
        elif mode == "sparse":
            ranked = self.sparse(query, pool)
        elif mode == "dense":
            ranked = self.dense(query, pool)
        else:
            raise ValueError("mode must be sparse, dense, or hybrid")
        query_terms = set(tokenize(query)) - STOP
        weights = {w: max(0.1, self.bm25.idf.get(w, 0.0)) for w in query_terms}
        total = sum(weights.values())
        candidates = []
        for chunk_id, original_score in ranked:
            chunk = self.chunks[chunk_id]
            if candidate_junk(chunk, query):
                continue
            terms = set(tokenize(chunk["text"]))
            coverage = sum(value for word, value in weights.items() if word in terms) / total if total else 0.0
            score = original_score
            if mode == "hybrid":
                rrf_normalized = original_score * 61 / 2
                score = 0.4 * rrf_normalized + 0.6 * coverage
            candidates.append({**chunk, "score": score, "retrieval_score": original_score,
                               "query_term_coverage": coverage})
        if mode == "hybrid":
            candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def search(self, query, mode="hybrid", k=5):
        if k < 1:
            raise ValueError("k must be positive")
        total = len(self.chunks)
        for pool in sorted({min(total, size) for size in (max(30, k), max(120, k), total)}):
            selected = select_candidates(self._candidates(query, mode, pool), k)
            if len(selected) >= min(k, total):
                return selected
        raise ValueError("The index contains too few distinct usable chunks; inspect the corpus")
