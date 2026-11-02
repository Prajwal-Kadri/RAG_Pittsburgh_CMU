"""Validate leaderboard predictions and add the required AndrewID field.

Uses the standard library only. Does not generate, change, or score answers.
"""
import argparse
import json
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def make_submission(questions, predictions, andrewid):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty JSON array")
    if any(not isinstance(q, dict) or "id" not in q
           or not isinstance(q.get("question"), str) for q in questions):
        raise ValueError("Each question must contain an id and a question string")
    ids = [str(q["id"]) for q in questions]
    if len(set(ids)) != len(ids) or any(not qid or qid == "andrewid" for qid in ids):
        raise ValueError("Question IDs must be unique and cannot use the andrewid field")
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must map question IDs to answer strings")
    missing, extra = set(ids) - set(predictions), set(predictions) - set(ids)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if any(not isinstance(answer, str) or not answer.strip() for answer in predictions.values()):
        raise ValueError("Every answer must be a nonempty string")
    andrewid = andrewid.strip()
    if not andrewid or andrewid == "YOUR_ANDREWID":
        raise ValueError("Supply your actual AndrewID")
    return {"andrewid": andrewid, **{qid: predictions[qid] for qid in ids}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/leaderboard_queries.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--andrewid", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        paths = [Path(args.questions), Path(args.predictions)]
        target = Path(args.out)
        if target.resolve() in {p.resolve() for p in paths}:
            raise ValueError("Choose a separate output path to preserve questions and predictions")
        questions, predictions = [json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
                                  for p in paths]
        submission = make_submission(questions, predictions, args.andrewid)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(submission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(target)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Validated {len(questions)} answers for AndrewID {submission['andrewid']}")
    print(f"Submission JSON saved to {target}")


if __name__ == "__main__":
    main()
