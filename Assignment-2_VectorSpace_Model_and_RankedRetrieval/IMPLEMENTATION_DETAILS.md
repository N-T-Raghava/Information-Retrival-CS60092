# Assignment 2: PyTerrier Ranked Retrieval

## Implementation

The system uses PyTerrier 1.1.2 and Terrier 5.11 for sparse ranked retrieval on Cranfield. `IterDictIndexer` builds an inverted index from the document title and body. Terrier performs tokenisation, stopword handling, Porter stemming, term weighting, cosine-style normalisation, and ranking. The project no longer uses a custom pickle index or hand-written document scoring.

`src/retrieval/terrier_pipeline.py` is the integration layer. It starts Terrier, builds or loads an index, runs `BatchRetrieve` with Terrier's `TF_IDF` weighting model, and converts Terrier's standard `qid`, `docno`, `rank`, and `score` output to the project's TREC writer format.

The Cranfield parser remains responsible for reading documents, queries, and relevance judgments. The evaluator computes MAP, P@5, P@10, R@10, MRR, nDCG@10, nDCG@20, and R-Precision from the Terrier run files.

## Experiment Plan

Only sparse vector-space retrieval is used, as required. The experiments vary:

| Experiment | Stopwords | Stemming | Terrier model | c |
|---|---:|---:|---|---:|
| `terrier_tfidf_baseline` | yes | yes | `TF_IDF` | 1.0 |
| `terrier_tfidf_c05` | yes | yes | `TF_IDF` | 0.5 |
| `terrier_tfidf_c2` | yes | yes | `TF_IDF` | 2.0 |
| `terrier_tfidf_no_stemming` | yes | no | `TF_IDF` | 1.0 |
| `terrier_tfidf_no_stopwords` | no | yes | `TF_IDF` | 1.0 |

Each row is indexed separately so that an ablation changes both document and query processing consistently. `c` is Terrier's TF_IDF normalisation parameter. No BM25, language model, learning-to-rank, probabilistic advanced model, or dense neural model is used.

## Running

Install dependencies and a JDK 11 or newer:

```bash
pip install -r requirements.txt
```

Set `JAVA_HOME`, then run:

```bash
python scripts/run_all.py \
  --config configs/experiments.yaml \
  --data-dir data \
  --index-path indexes/cranfield
```

The command writes one TREC run per experiment, `results/experiment_summary.csv` and `.json`, and Terrier indexing/query timings under `benchmarks/`.

For an existing baseline index, arbitrary queries use tab-separated `qid<TAB>query` input:

```bash
python scripts/search.py \
  --index indexes/terrier_tfidf_baseline \
  --query-file test_queries.txt \
  --output results/unknown_queries.txt \
  --model tfidf
```

## Reproducibility

The configuration, collection, qrels, Terrier index settings, model, parameter values, run files, evaluation metrics, and timing measurements are all recorded in the repository. The experiment command must be rerun on a JDK-enabled machine after implementation changes; pre-existing result files are not treated as results of the Terrier migration.
