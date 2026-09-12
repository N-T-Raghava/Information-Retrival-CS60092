# IR Assignment 2: Vector Space Model and Ranked Retrieval

This project implements ranked retrieval on the Cranfield collection using the open-source Terrier search engine through PyTerrier. Only Terrier's sparse `TF_IDF` vector-space model is used. Refer to [the implementation write-up](IMPLEMENTATION_DETAILS.md) for details.

## Setup

PyTerrier runs Terrier through Java. Install Java 11 or newer and configure it before running the experiments.

```powershell
winget install --id EclipseAdoptium.Temurin.11.JDK --exact --accept-source-agreements --accept-package-agreements
```

After installation, reopen PowerShell and configure the JDK. Update the path if a different Temurin version is installed:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
[Environment]::SetEnvironmentVariable('JAVA_HOME', $env:JAVA_HOME, 'User')
[Environment]::SetEnvironmentVariable('Path', "$userPath;$env:JAVA_HOME\bin", 'User')
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

Install the Python dependencies from the Assignment 2 directory:

```powershell
cd Assignment-2_VectorSpace_Model_and_RankedRetrieval
pip install -r requirements.txt
```

## Run the Experiments

Run the complete staged experiment plan:

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data --index-path indexes/cranfield --force-index
```

The runner performs:

- Group A: TF-IDF baseline
- Group B: `c` tuning for `0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`
- Group C: stopword and stemming ablations using the best `c` from Group B

Results are written to `results/`, Terrier indexes to `indexes/`, and timing data to `benchmarks/`.

To build one Terrier index separately:

```powershell
python scripts/build_index.py --data-dir data --index-path indexes/terrier_tfidf_baseline --force
```

For ad-hoc queries, use a tab-separated file containing `qid<TAB>query text`:

```powershell
python scripts/search.py --index indexes/terrier_tfidf_baseline --query-file test_queries.txt --output results/unknown_queries.txt --model tfidf --top-k 1000
```

- Refer [USAGE_GUIDE](USAGE_GUIDE.md) for more commands and instructions

## Project Structure

- `src/` - Main source code
  - `data/` - Data parsing and models
  - `retrieval/` - PyTerrier/Terrier indexing and retrieval adapter
  - `evaluation/` - Evaluation metrics
  - `benchmarking/` - Performance benchmarking
  - `io/` - Result file I/O
- `data/` - Cranfield collection
- `indexes/` - Built indexes
- `results/` - Experiment results
- `benchmarks/` - Benchmark data
- `configs/` - Configuration files

## Retrieval Experiments

- Group A: Terrier `TF_IDF` baseline with `c=1.0`
- Group B: Terrier `TF_IDF` tuning with `c` values `0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`
- Group C: four stopword/stemming ablations using the MAP-best `c`
- Every experiment uses its own Terrier index

## Evaluation Metrics

- MAP (Mean Average Precision)
- P@5, P@10 (Precision at k)
- R@10 (Recall at 10)
- MRR (Mean Reciprocal Rank)
- nDCG@10, nDCG@20 (Normalized Discounted Cumulative Gain)
- R-Precision

## Key Features

- **Reproducible**: Deterministic processing, all parameters configured
- **Modular**: Independent components for preprocessing, indexing, retrieval, evaluation
- **Configurable**: YAML-based experiment configuration
- **Tested**: Comprehensive pytest suite
- **Unknown query compatible**: Works with arbitrary queries and query formats
- **Benchmarked**: Separate Terrier indexing and query timing

PyTerrier requires Java 11 or newer. Set `JAVA_HOME` to the JDK directory before running an index or experiment command.
