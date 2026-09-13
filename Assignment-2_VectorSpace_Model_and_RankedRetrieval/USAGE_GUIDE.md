# Usage Guide

## Requirements

The Terrier baseline requires Python 3.11-compatible dependencies and Java 11 or newer.

```powershell
cd Assignment-2_VectorSpace_Model_and_RankedRetrieval
pip install -r requirements.txt
```

Configure Java explicitly in PowerShell before running the Terrier baseline. Change the JDK path when necessary:

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

Do not continue until `java -version` reports Java 11 or newer. Restart PowerShell after changing the user-level environment variables if required.

## Run the Pipeline

The project intentionally has one configuration file and one runner:

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data
```

The runner builds or loads the Terrier index, evaluates Terrier `TF_IDF`, runs the SMART VSM experiments, evaluates Rocchio feedback, and writes all results and timings.

Use `--force-index` when the collection or Terrier settings have changed:

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data --force-index
```

Use `--top-k` for a smaller diagnostic run or a different retrieval depth:

```powershell
python scripts/run_all.py --config configs/experiments.yaml --data-dir data --top-k 100
```

## Configuration

Edit [configs/experiments.yaml](configs/experiments.yaml) to change:

- preprocessing combinations
- SMART document/query schemes
- pivot slopes
- Rocchio alpha, beta, gamma, feedback-document, and feedback-term values
- Terrier model, `c`, and enabled state
- default `top_k`

No experiment settings are hard-coded in separate scripts.

## Outputs

Results are written to `results/`:

- stage CSV files for each experiment group
- `MaamTeam_terrier_tfidf.txt` for the Terrier baseline
- `MaamTeam_results.txt` for the selected SMART system
- `MaamTeam_best_config.json` for the selected parameters and final metrics
- `experiment_summary.csv` and `experiment_summary.json`

Timing data is written to `benchmarks/timing.csv` and `benchmarks/timing.json`.

Runs use standard TREC format:

```text
qid Q0 docno rank score run_name
```

## Reproducibility

Parsing and topic splitting are deterministic. Re-running with the same data, configuration, and software versions regenerates the experiment tables and run files. Existing generated results are overwritten; indexes are reused unless `--force-index` is supplied.
