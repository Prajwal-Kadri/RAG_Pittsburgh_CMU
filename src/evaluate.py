"""Local QA metrics; uses only Python's standard library.

These are transparent development metrics, not a reproduction of the staff's
LLM judge or a guarantee of the official leaderboard score.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string


def tokens(text):
    # SQuAD-style normalization: lowercase, remove ASCII punctuation/articles,
    # and collapse whitespace. Keep Unicode letters and accents unchanged.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return text.split()


def score_pair(prediction, reference):
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        equal = float(p == r)
        return dict(exact_match=equal, answer_recall=equal,
                    token_f1=equal, rouge_l=equal)
    overlap = sum((Counter(p) & Counter(r)).values())
    precision, recall = overlap / len(p), overlap / len(r)
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    # Longest common subsequence, with O(len(reference)) working memory.
    row = [0] * (len(r) + 1)
    for word in p:
        previous = 0
        for j, other in enumerate(r, 1):
            old = row[j]
            row[j] = previous + 1 if word == other else max(row[j], row[j - 1])
            previous = old
    rouge_l = 2 * row[-1] / (len(p) + len(r))
    return dict(exact_match=float(p == r), answer_recall=recall,
                token_f1=f1, rouge_l=rouge_l)


def evaluate(questions, predictions):
    if not isinstance(questions, list) or not questions:
        raise ValueError('References must be a nonempty JSON array of question records.')
    if not isinstance(predictions, dict):
        raise ValueError('Predictions must be a JSON object mapping IDs to answer strings.')
    ids = []
    for item in questions:
        if not isinstance(item, dict) or 'id' not in item:
            raise ValueError('Every reference record must contain an id.')
        answers = item.get('answers')
        if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and tokens(a) for a in answers):
            raise ValueError(f'Invalid or empty reference answers for ID {item["id"]}.')
        ids.append(str(item['id']))
    if len(set(ids)) != len(ids):
        raise ValueError('Reference IDs must be unique.')
    missing = set(ids) - set(predictions)
    extra = set(predictions) - set(ids)
    if missing or extra:
        raise ValueError(f'ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    if not all(isinstance(v, str) for v in predictions.values()):
        raise ValueError('Every prediction must be a string.')
    details = []
    for item, qid in zip(questions, ids):
        variants = [score_pair(predictions[qid], answer) for answer in item['answers']]
        # Standard multiple-reference convention: maximize each metric separately.
        scores = {key: max(v[key] for v in variants) for key in variants[0]}
        details.append(dict(id=qid, question=item.get('question', ''),
                            prediction=predictions[qid], references=item['answers'], **scores))
    means = {key: sum(d[key] for d in details) / len(details)
             for key in ('exact_match', 'answer_recall', 'token_f1', 'rouge_l')}
    return dict(num_questions=len(details), metrics=means, per_question=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', required=True)
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--out', help='Optional JSON with means and per-question scores.')
    args = parser.parse_args()
    try:
        questions = json.loads(Path(args.references).read_text(encoding='utf-8'))
        results = {}
        for filename in args.predictions:
            predictions = json.loads(Path(filename).read_text(encoding='utf-8'))
            results[filename] = evaluate(questions, predictions)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('All metrics below are percentages (higher is better).')
    print(f'{"System":36} {"N":>5} {"EM":>8} {"Recall":>8} {"F1":>8} {"ROUGE-L":>8}')
    for filename, result in results.items():
        m = result['metrics']
        print(f'{Path(filename).stem:36} {result["num_questions"]:5d}'
              f' {100*m["exact_match"]:8.2f} {100*m["answer_recall"]:8.2f}'
              f' {100*m["token_f1"]:8.2f} {100*m["rouge_l"]:8.2f}')
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Saved detailed scores to {target}')


if __name__ == '__main__':
    main()
