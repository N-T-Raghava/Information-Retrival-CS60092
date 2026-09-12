import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.timing import Benchmark, BenchmarkSuite
from src.data.parser import CranfieldParser, validate_data
from src.evaluation.metrics import Evaluator
from src.io.run_writer import RunWriter
from src.retrieval.terrier_pipeline import build_index, retrieve


def run_experiment(
    name, stage, documents, queries, judged_query_ids, qrel_dict,
    preprocessing, c, args, top_k, benchmarks,
):
    model = "TF_IDF"
    index_key = f"{stage}_{name}_c{str(c).replace('.', 'p')}"
    index_path = Path(args.index_path).parent / index_key
    print(f"[{stage}] {name} (c={c})", flush=True)
    indexref, index_time = build_index(
        documents, str(index_path),
        stopwords=preprocessing["stopwords"],
        stemming=preprocessing["stemming"],
        force=args.force_index,
    )
    start = time.perf_counter()
    ranked, retrieval_time = retrieve(
        indexref, queries, model=model, top_k=top_k, c=float(c), run_name=name
    )
    elapsed = time.perf_counter() - start
    RunWriter.write_run(ranked, str(Path(args.results_dir) / f"{name}.txt"))
    metrics = Evaluator.evaluate_all(ranked, judged_query_ids, qrel_dict)

    benchmark = Benchmark(name)
    benchmark.record("index_time", index_time)
    benchmark.record("retrieval_time", retrieval_time)
    benchmark.record("avg_query_time", retrieval_time / len(queries))
    benchmark.add_metadata("engine", "PyTerrier/Terrier")
    benchmark.add_metadata("stage", stage)
    benchmark.add_metadata("model", model)
    benchmarks.add(benchmark)

    row = {
        "stage": stage,
        "experiment": name,
        "engine": "PyTerrier/Terrier",
        "model": model,
        "index_time": index_time,
        "retrieval_time": elapsed,
        "avg_query_time": retrieval_time / len(queries),
        "evaluation_queries": len(judged_query_ids),
        "c": float(c),
        "stopwords": preprocessing["stopwords"],
        "stemming": preprocessing["stemming"],
    }
    row.update(metrics)
    print(f"  MAP={metrics['MAP']:.6f} nDCG@10={metrics['nDCG@10']:.6f}", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged sparse PyTerrier experiments")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--index-path", default="indexes/cranfield")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--benchmarks-dir", default="benchmarks")
    parser.add_argument("--force-index", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_dir = Path(args.data_dir)
    documents, doc_dict = CranfieldParser.parse_documents(str(data_dir / "cran.all.1400"))
    queries, query_dict = CranfieldParser.parse_queries(str(data_dir / "cran.qry"))
    query_ids = [query.qid for query in queries]
    qrels, qrel_dict = CranfieldParser.parse_qrels(str(data_dir / "cranqrel"), query_ids=query_ids)
    validate_data(doc_dict, query_dict, qrels)
    judged_query_ids = query_ids
    top_k = config.get("retrieval", {}).get("top_k", 1000)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.benchmarks_dir).mkdir(parents=True, exist_ok=True)
    benchmarks = BenchmarkSuite()
    rows = []
    print(f"Loaded {len(documents)} documents, {len(queries)} queries, {len(qrels)} qrels")
    print(f"Evaluation coverage: {len(judged_query_ids)}/{len(query_ids)} queries")

    baseline = config["baseline"]
    rows.append(run_experiment(
        baseline["name"], "A_baseline", documents, queries, judged_query_ids,
        qrel_dict, baseline["preprocessing"], baseline["retrieval"]["c"],
        args, top_k, benchmarks,
    ))

    tuning_preprocessing = {"stopwords": True, "stemming": True}
    tuning_rows = []
    for c in config["parameter_tuning"]["c_values"]:
        tuning_rows.append(run_experiment(
            f"terrier_tfidf_c{str(c).replace('.', '')}", "B_parameter_tuning",
            documents, queries, judged_query_ids, qrel_dict,
            tuning_preprocessing, c, args, top_k, benchmarks,
        ))
    rows.extend(tuning_rows)
    best_map_row = max(tuning_rows, key=lambda row: row["MAP"])
    best_ndcg_row = max(tuning_rows, key=lambda row: row["nDCG@10"])
    best_c = best_map_row["c"]
    print(f"Best c by MAP: {best_c} ({best_map_row['MAP']:.6f})")
    print(f"Best c by nDCG@10: {best_ndcg_row['c']} ({best_ndcg_row['nDCG@10']:.6f})")

    for ablation in config["preprocessing_ablation"]:
        rows.append(run_experiment(
            ablation["name"], "C_preprocessing_ablation", documents, queries,
            judged_query_ids, qrel_dict, ablation["preprocessing"], best_c,
            args, top_k, benchmarks,
        ))

    frame = pd.DataFrame(rows).sort_values(["stage", "MAP"], ascending=[True, False])
    frame["best_c_by_map"] = best_c
    frame["best_c_by_ndcg10"] = best_ndcg_row["c"]
    frame.to_csv(Path(args.results_dir) / "experiment_summary.csv", index=False)
    with open(Path(args.results_dir) / "experiment_summary.json", "w", encoding="utf-8") as output:
        json.dump(frame.to_dict("records"), output, indent=2)
    benchmarks.save_csv(str(Path(args.benchmarks_dir) / "timing.csv"))
    benchmarks.save_json(str(Path(args.benchmarks_dir) / "timing.json"))
    print("\nExperiment summary:")
    print(frame[["stage", "experiment", "c", "MAP", "P@10", "nDCG@10"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
