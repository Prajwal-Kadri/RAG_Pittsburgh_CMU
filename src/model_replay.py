"""Compare an open-weight reader on exact evidence from a completed run.

No retrieval, source collection, answer-key lookup, training, or question-type
rules. Reference answers never enter model messages. Each model gets its own
closed-book predictions, and checkpoint fingerprints prevent mixed runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .common import read_jsonl
from .llm_predict import LLMReader, fingerprint, load_checkpoint, messages_for, save_json
from .make_submission import unique_object


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODES = ("closed_book", "sparse", "dense", "hybrid")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def load_inputs(baseline, question_path, modes):
    questions = read(question_path)
    if (not isinstance(questions, list) or not questions or any(
            not isinstance(q, dict) or "id" not in q or not isinstance(q.get("question"), str)
            for q in questions)):
        raise ValueError("Questions must be a nonempty array of IDs and question strings")
    ids = {str(q["id"]) for q in questions}
    if len(ids) != len(questions):
        raise ValueError("Duplicate question IDs")
    config = read(baseline / "run_config.json")
    if (not isinstance(config.get("k"), int) or config["k"] < 1) or config.get("do_sample") is not False or config.get("max_new_tokens") != 96:
        raise ValueError("Baseline must record a positive k, greedy generation and 96 answer tokens")
    if not config.get("model_commit"):
        raise ValueError("Baseline must record its model commit")
    prompt = messages_for("", [], config["prompt_version"])[0]["content"]
    if config.get("retrieval_system_prompt") != prompt:
        raise ValueError("Baseline prompt differs from current message generation")
    artifacts = Path(config["artifacts"])
    paths = {"questions": Path(question_path), **{n: artifacts / n for n in
             ("config.json", "chunks.jsonl", "bm25_tokens.json", "dense.faiss")}}
    hashes = {name: fingerprint(path) for name, path in paths.items()}
    if config.get("input_hashes") != hashes:
        raise ValueError("Baseline questions or index changed")
    chunks = read_jsonl(artifacts / "chunks.jsonl")
    cached, originals = {}, {}
    sources = {"baseline_config": baseline / "run_config.json"}
    for mode in modes:
        predictions, evidence = load_checkpoint(baseline, mode, ids, chunks)
        if set(predictions) != ids or any(not a.strip() for a in predictions.values()):
            raise ValueError("Incomplete baseline predictions: " + mode)
        originals[mode] = predictions
        sources[mode + "_predictions"] = baseline / (mode + ".json")
        if mode != "closed_book":
            if set(evidence) != ids or any(not 1 <= len(h) <= config["k"] for h in evidence.values()):
                raise ValueError("Baseline must have valid saved chunks per question: " + mode)
            cached[mode] = evidence
            sources[mode + "_evidence"] = baseline / (mode + "_evidence.json")
    return questions, ids, config, chunks, cached, originals, hashes, sources


class ReplayReader(LLMReader):
    """Unquantized BF16, streamed directly onto one suitably sized CUDA GPU."""
    def __init__(self, model_name, revision, prompt_version, minimum_gpu_gib=75):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import accelerate  # Required for streaming, without loading 61GB on CPU first.

        if not torch.cuda.is_available():
            raise ValueError("This model experiment requires a CUDA GPU")
        if torch.cuda.get_device_properties(0).total_memory < minimum_gpu_gib * 1024 ** 3:
            raise ValueError(f"Need at least {minimum_gpu_gib} GiB GPU memory; no quantization or CPU offload is enabled")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU must support BF16")
        self.torch, self.device, self.prompt_version = torch, "cuda:0", prompt_version
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        count = sum(p.numel() for p in self.model.parameters())
        if count > 32_000_000_000:
            raise ValueError("Loaded model exceeds the 32B TOTAL parameter cap")
        if getattr(self.model.config, "_commit_hash", None) != revision:
            raise ValueError("Loaded model commit differs from the pinned revision")
        self.details = {"total_parameters": count, "dtype": "bfloat16", "quantization": None,
                        "minimum_gpu_gib": minimum_gpu_gib,
                        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "transformers": transformers.__version__, "accelerate": accelerate.__version__}


def resolve_model(model, revision):
    from huggingface_hub import HfApi
    info = HfApi().model_info(model, revision=revision)
    total = getattr(getattr(info, "safetensors", None), "total", None)
    if not isinstance(total, int) or not 0 < total <= 32_000_000_000:
        raise ValueError("Require verified safetensors total parameter count <=32B before loading")
    if info.created_at is None or info.created_at >= datetime(2026, 1, 1, tzinfo=timezone.utc):
        raise ValueError("Model repository must predate January 1, 2026; confirm release provenance separately")
    return {"model": model, "revision": info.sha, "reported_total_parameters": total,
            "repository_created_at": info.created_at.isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", help="Optional immutable model commit")
    parser.add_argument("--min-gpu-gib", type=int, default=75,
                        help="Physical GPU-memory floor: 75 for the 30.5B experiment, 40 for 14B")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_gpu_gib <= 1024:
        parser.error("--min-gpu-gib must be between 1 and 1024")
    baseline, root = Path(args.baseline_dir), Path(args.out_dir)
    if baseline.resolve() == root.resolve() or baseline.resolve() in root.resolve().parents or root.resolve() in baseline.resolve().parents:
        parser.error("Use a separate output directory outside the baseline")
    modes = list(dict.fromkeys(args.modes))
    questions, ids, baseline_config, chunks, cached, originals, hashes, sources = load_inputs(baseline, args.questions, modes)
    from . import llm_predict
    previous = read(root / "run_config.json") if args.resume else None
    if not args.resume and root.exists() and any(root.iterdir()):
        parser.error("Output already exists; choose a new directory or use --resume")
    identity = {"model": args.model, "baseline_dir": str(baseline), "questions": args.questions,
                "modes": modes, "input_hashes": hashes,
                "saved_run_hashes": {name: fingerprint(path) for name, path in sources.items()},
                "runner_sha256": fingerprint(Path(__file__)), "generation_code_sha256": fingerprint(Path(llm_predict.__file__)),
                "k": baseline_config["k"], "max_new_tokens": 96, "do_sample": False,
                "prompt_version": baseline_config["prompt_version"], "retrieval_system_prompt": baseline_config["retrieval_system_prompt"],
                "dtype": "bfloat16", "quantization": None, "attention_implementation": "sdpa",
                "minimum_gpu_gib": args.min_gpu_gib,
                "baseline_model": baseline_config["model"], "baseline_model_commit": baseline_config["model_commit"],
                "retrieval": "exact ordered baseline evidence; no retrieval performed"}
    if previous:
        if previous.get("experiment") != identity or (args.revision and args.revision != previous["model_metadata"]["revision"]):
            parser.error("Cannot resume: evidence, code, model, prompt or settings changed")
        metadata = previous["model_metadata"]
    else:
        metadata = resolve_model(args.model, args.revision)
    record = {"experiment": identity, "model_metadata": metadata,
              "note": "Reference answers are used only by the separate evaluator. No v9 targeted repair rules. Closed book is regenerated by the new model."}
    if previous and "runtime" in previous:
        record["runtime"] = previous["runtime"]
    save_json(root / "run_config.json", record)
    checkpoints = {}
    for mode in modes:
        predictions, evidence = load_checkpoint(root, mode, ids, chunks) if args.resume else ({}, {})
        if mode != "closed_book":
            if evidence and evidence != cached[mode]:
                raise ValueError("Saved evidence differs from the exact baseline order: " + mode)
            save_json(root / (mode + "_evidence.json"), cached[mode])
        checkpoints[mode] = predictions
    reader = None
    from tqdm import tqdm
    for mode in modes:
        predictions = checkpoints[mode]
        pending = [q for q in questions if str(q["id"]) not in predictions]
        print(f"{mode}: {len(predictions)}/{len(ids)} saved; {len(pending)} remaining", flush=True)
        if pending and reader is None:
            reader = ReplayReader(args.model, metadata["revision"], baseline_config["prompt_version"], args.min_gpu_gib)
            record["runtime"] = reader.details
            save_json(root / "run_config.json", record)
        for q in tqdm(pending, desc=mode + " model replay"):
            qid = str(q["id"])
            hits = None if mode == "closed_book" else cached[mode][qid]
            answer = reader.answer(q["question"], hits, 96)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Model returned an empty answer for " + qid)
            predictions[qid] = answer
            save_json(root / (mode + ".json"), predictions)
        print("Wrote", len(predictions), "predictions to", root / (mode + ".json"), flush=True)
    save_json(root / "answer_summary.json", {mode: {
        "questions": len(ids), "changed_answers": sum(a != originals[mode][qid] for qid, a in predictions.items()),
        "literal_abstentions": sum(a.strip().rstrip(".").casefold() == "i don't know" for a in predictions.values())}
        for mode, predictions in checkpoints.items()})
    print("Finished model replay. No leaderboard submission or gold scoring was performed by this module.")


if __name__ == "__main__":
    main()
