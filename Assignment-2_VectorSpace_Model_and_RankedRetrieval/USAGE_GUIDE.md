# Usage Guide

## Requirements

Install Python dependencies and Java 11 or newer. On Windows, install Eclipse Temurin 11 with `winget`:

```powershell
winget install --id EclipseAdoptium.Temurin.11.JDK --exact --accept-source-agreements --accept-package-agreements
```

Close and reopen PowerShell after installation. Configure the installed JDK explicitly:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
[Environment]::SetEnvironmentVariable('JAVA_HOME', $env:JAVA_HOME, 'User')
[Environment]::SetEnvironmentVariable('Path', "$userPath;$env:JAVA_HOME\bin", 'User')
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

If the JDK version changes, replace the `JAVA_HOME` path with the folder shown under `C:\Program Files\Eclipse Adoptium`.

Install the Python dependencies:

```powershell
cd Assignment-2_VectorSpace_Model_and_RankedRetrieval
pip install -r requirements.txt
```

PyTerrier uses Java to run Terrier. Do not run the experiment command until `java -version` succeeds.

## Run Experiments

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data --index-path indexes/cranfield --force-index
```

The runner creates one Terrier index per experiment under the parent of `--index-path`, writes TREC runs under `results/`, and records indexing and query timings under `benchmarks/`.

## Build One Index

```powershell
python scripts/build_index.py --data-dir data --index-path indexes/terrier_tfidf_baseline --force
```

This creates a Terrier index with stopword removal and Porter stemming. The experiment runner is preferred when comparing preprocessing settings.

## Search New Queries

Input is tab-separated: `qid<TAB>query text`.

```powershell
python scripts/search.py --index indexes/terrier_tfidf_baseline --query-file test_queries.txt --output results/unknown_queries.txt --model tfidf --top-k 1000
```

Output uses standard TREC format:

```text
qid Q0 docno rank score run_name
```

## Experiments

All configured runs use Terrier's sparse `TF_IDF` model. The configuration varies the `c` normalisation parameter and tests stopword/stemming ablations. Dense retrieval and advanced probabilistic models are intentionally excluded.
