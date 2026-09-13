# IR Assignment 2: Ranked Retrieval

This project evaluates two sparse vector-space retrieval systems on the Cranfield collection:

- Terrier `TF_IDF`, the standard baseline.
- Custom SMART VSM with pivoted normalization and Rocchio feedback.

Both systems use the same parsed collection, relevance judgments, deterministic topic split, metrics, and output directories. There is one experiment configuration and one runner.

The detailed implementation report is available in [MaamTeam_IR_Assignment_II_Report](MaamTeam_IR_Assignment_II_Report.pdf).

## Setup

Install Python dependencies and Java 11 or newer for Terrier:

```powershell
cd Assignment-2_VectorSpace_Model_and_RankedRetrieval
pip install -r requirements.txt
```

On Windows PowerShell, configure the JDK before running Terrier. Update the path if a different Temurin version is installed:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot'
if (-not (Test-Path $env:JAVA_HOME)) { throw "JDK not found at $env:JAVA_HOME" }
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
[Environment]::SetEnvironmentVariable('JAVA_HOME', $env:JAVA_HOME, 'User')
if ($userPath -notlike "*$env:JAVA_HOME\bin*") {
	[Environment]::SetEnvironmentVariable('Path', "$userPath;$env:JAVA_HOME\bin", 'User')
}
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

The `JAVA_HOME` directory must contain a Java 11 or newer JDK. Restart PowerShell after changing the user-level environment variables if a new shell does not see the update.

## Run All Experiments

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data
```

For a clean Terrier index rebuild:

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data --force-index
```

Useful options:

- `--top-k 1000` overrides the configured retrieval depth.
- `--index-path indexes/terrier_tfidf_baseline` controls the Terrier index location.
- `--results-dir results` controls experiment outputs.
- `--benchmarks-dir benchmarks` controls timing outputs.

## Experiment Stages

The runner executes these stages in order:

1. Preprocessing sweep for title weighting, stopwords, stemming, field inclusion, and title deduplication.
2. SMART scheme comparison.
3. SMART pivot-slope tuning.
4. Rocchio feedback tuning.
5. Combined SMART model and feedback selection.
6. Terrier `TF_IDF` baseline evaluation using the selected preprocessing.
7. Final held-out and all-topic evaluation.

The held-out split uses every third topic for validation; the remaining topics are used for training-stage selection.

## Outputs

All outputs are generated under the configured directories:

- `results/MaamTeam_stage1_preprocessing.csv` through `results/MaamTeam_stage5_best_combination.csv`
- `results/MaamTeam_terrier_tfidf.txt`
- `results/MaamTeam_results.txt`
- `results/MaamTeam_best_config.json`
- `results/experiment_summary.csv` and `.json`
- `benchmarks/timing.csv` and `.json`

The final configuration records both the selected SMART system and Terrier baseline metrics.

## Source Layout

- `src/data/` parses Cranfield documents, queries, and qrels.
- `src/retrieval/terrier_pipeline.py` owns Terrier indexing and retrieval.
- `src/retrieval/smart_vsm.py` owns SMART weighting and Rocchio feedback.
- `src/evaluation/metrics.py` computes all reported metrics.
- `src/io/run_writer.py` writes TREC-format runs.
- `configs/experiments.yaml` contains all experiment settings.
- `scripts/run_all.py` is the only experiment entry point.
