# Pittsburgh & CMU Question Answering

A retrieval-augmented generation (RAG) system that answers questions about Pittsburgh and Carnegie Mellon University. It searches a collection of public sources, selects relevant passages, and uses a language model to generate an answer grounded in those passages.

Built for CMU's Advanced Natural Language Processing course.

> **Looking for the code?** This `readme-only` branch contains only this document. Please write a mail to pkadri@andrew.cmu.edu tor request access to the private repository.

## How it works

1. **Collect and prepare sources.** Public webpages and PDFs are cleaned and split into passages.
2. **Search two ways.** BM25 retrieves passages by keyword, while a sentence-transformer and FAISS retrieve semantically similar passages.
3. **Combine and select evidence.** Reciprocal rank fusion combines the rankings; additional selection and relevance rules refine the passages supplied to the reader.
4. **Generate an answer.** Qwen2.5-14B-Instruct responds using the selected evidence.

The project also includes evaluation code for comparing generated answers with reference answers.

## Results at a glance

The saved runs below were evaluated on **157 public development questions**. Scores are percentages; higher is better.

| Configuration | Exact match | Answer recall | Token F1 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Earlier hybrid, 14B, 15 passages | 63.69 | 74.14 | 74.10 | 73.61 |
| Revised evidence, 14B, 15 passages | 49.68 | 80.58 | 65.42 | 64.62 |
| Revised evidence, 14B, 25 passages | 45.86 | 81.85 | 64.40 | 63.40 |

The revised-evidence runs use additional sources and a different evidence-selection process, so these rows are not a controlled comparison of passage count alone. They illustrate a trade-off in these saved runs: higher answer recall came with lower exact match and F1.

The scores are development-set results, not an estimate on a held-out dataset. See [saved metrics on `main`](https://github.com/Prajwal-Kadri/11711_HomeWork2/blob/main/results/public_metrics.json) for the underlying values.

## Project contents

The following files are on the [`main` branch](https://github.com/Prajwal-Kadri/11711_HomeWork2/tree/main):

| Path | Purpose |
| --- | --- |
| [`src/`](https://github.com/Prajwal-Kadri/11711_HomeWork2/tree/main/src) | Collection, cleaning, chunking, indexing, retrieval, generation, and evaluation code |
| [`run.py`](https://github.com/Prajwal-Kadri/11711_HomeWork2/blob/main/run.py) | Revised-evidence pipeline for 15 and 25 passages |
| [`data/`](https://github.com/Prajwal-Kadri/11711_HomeWork2/tree/main/data) | Compressed source corpus, seed URLs, and public development questions with reference answers |
| [`results/`](https://github.com/Prajwal-Kadri/11711_HomeWork2/tree/main/results) | Saved public predictions and aggregate metrics |
| [`requirements.txt`](https://github.com/Prajwal-Kadri/11711_HomeWork2/blob/main/requirements.txt) | Python dependencies |

## Getting started

### Requirements

- Python 3.10 or newer.
- A CUDA GPU with **at least 40 GiB of memory** and BF16 support to run the 14B reader. The index can be built on a CPU.
- Network access to download the embedding model and language model.

The reader uses [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), pinned to revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. The embedding model is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

### 1. Get the code and install dependencies

Clone the **main** branch, which contains the files needed below:

```bash
git clone --branch main https://github.com/Prajwal-Kadri/11711_HomeWork2.git
cd 11711_HomeWork2

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment if the default installation is unsuitable.

### 2. Build the search index

```bash
gzip -dc data/corpus.jsonl.gz > data/corpus.jsonl

python -m src.build_index_v7 \
  --input data/corpus.jsonl \
  --out artifacts_coverage_v8 \
  --device cpu
```

The original index is not included. The rebuilt index makes the pipeline runnable, but exact equivalence to the index used for the saved results has not been verified.

### 3. Generate answers

Run the pipeline on the included public development questions:

```bash
python run.py \
  --questions data/public_evaluation_data.json \
  --out runs/public
```

For your own input, use a JSON array with an `id` and a `question` for each item:

```json
[
  {
    "id": "1",
    "question": "Which three rivers meet in Pittsburgh?"
  }
]
```

The run produces predictions under `runs/public/run/k15/hybrid.json` and `runs/public/run/k25/hybrid.json`, along with evidence and checkpoints. Use a new output directory when changing inputs or configuration.

### 4. Evaluate predictions

For inputs with reference answers, calculate local metrics with:

```bash
python -m src.evaluate \
  --references data/public_evaluation_data.json \
  --predictions runs/public/run/k15/hybrid.json runs/public/run/k25/hybrid.json \
  --out runs/public/scores.json
```

To inspect previously saved aggregate scores without running a model, open [`results/public_metrics.json`](https://github.com/Prajwal-Kadri/11711_HomeWork2/blob/main/results/public_metrics.json) on `main`.

## Tools and models

- [Sentence Transformers](https://www.sbert.net/) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for dense embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search and [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword retrieval
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) for answer generation
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), [pypdf](https://pypdf.readthedocs.io/), and [pdfplumber](https://github.com/jsvine/pdfplumber) for source processing
