import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parser import CranfieldParser, validate_data
from src.evaluation.metrics import Evaluator
from src.io.run_writer import RunWriter
from src.retrieval.smart_vsm import PreprocConfig, RocchioConfig, SmartVSM
from src.retrieval.terrier_pipeline import build_index, retrieve


def evaluate(engine, queries, qids, qrels, top_k, feedback=None):
    start = time.perf_counter()
    if feedback is None:
        results = engine.search_all(queries, top_k)
    else:
        results = []
        for query in queries:
            results.extend(engine.search_with_rocchio(query, feedback, top_k))
    elapsed = time.perf_counter() - start
    metrics = Evaluator.evaluate_all(results, qids, qrels)
    metrics.update({"query_count": len(queries), "elapsed_s": elapsed,
                    "ms_per_query": elapsed * 1000 / max(len(queries), 1)})
    return metrics, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all MaamTeam SMART VSM experiments")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--benchmarks-dir", default="benchmarks")
    parser.add_argument("--index-path", default="indexes/terrier_tfidf_baseline")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data_dir = Path(args.data_dir)
    documents, document_dict = CranfieldParser.parse_documents(str(data_dir / "cran.all.1400"))
    queries, query_dict = CranfieldParser.parse_queries(str(data_dir / "cran.qry"))
    query_ids = [query.qid for query in queries]
    _, qrels = CranfieldParser.parse_qrels(str(data_dir / "cranqrel"), query_ids)
    validate_data(document_dict, query_dict, list(qrels.values()))
    train = [query for index, query in enumerate(queries) if index % 3]
    held_out = [query for index, query in enumerate(queries) if index % 3 == 0]
    train_ids = [query.qid for query in train]
    held_out_ids = [query.qid for query in held_out]
    top_k = args.top_k or config["retrieval"]["top_k"]
    result_dir = Path(args.results_dir)
    benchmark_dir = Path(args.benchmarks_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    timings = []

    def record(stage, name, metrics, metadata):
        rows.append({"stage": stage, "experiment": name, **metrics})
        timings.append({"stage": stage, "experiment": name, **metadata,
                        "elapsed_s": metrics["elapsed_s"], "ms_per_query": metrics["ms_per_query"]})

    preprocessing_rows = []
    for item in config["preprocessing"]:
        preproc = PreprocConfig(**{key: value for key, value in item.items() if key != "name"})
        metrics, _ = evaluate(SmartVSM(documents, preproc, "lnc", "ltc"), train, train_ids, qrels, top_k)
        preprocessing_rows.append((item, metrics))
        record("preprocessing", item["name"], metrics, {"model": "lnc.ltc"})
    best_preproc_data, _ = max(preprocessing_rows, key=lambda entry: entry[1]["MAP"])
    best_preproc = PreprocConfig(**{key: value for key, value in best_preproc_data.items() if key != "name"})
    pd.DataFrame([metrics | {"experiment": item["name"]} for item, metrics in preprocessing_rows]).to_csv(
        result_dir / "MaamTeam_stage1_preprocessing.csv", index=False
    )

    terrier_metrics = None
    if config.get("terrier", {}).get("enabled", False):
        terrier_config = config["terrier"]
        indexref, index_elapsed = build_index(
            documents, args.index_path, stopwords=best_preproc.stopwords,
            stemming=best_preproc.stem, force=args.force_index,
        )
        terrier_results, query_elapsed = retrieve(
            indexref, queries, model=terrier_config.get("model", "TF_IDF"),
            top_k=top_k, c=terrier_config.get("c", 1.0),
            run_name=terrier_config.get("name", "terrier_tfidf_baseline"),
        )
        terrier_train_results = [result for result in terrier_results if result.qid in train_ids]
        terrier_metrics = Evaluator.evaluate_all(terrier_train_results, train_ids, qrels)
        terrier_all_metrics = Evaluator.evaluate_all(terrier_results, query_ids, qrels)
        terrier_metrics.update({"query_count": len(queries), "elapsed_s": query_elapsed,
                                "ms_per_query": query_elapsed * 1000 / max(len(queries), 1),
                                "index_elapsed_s": index_elapsed})
        record("terrier_baseline", terrier_config.get("name", "terrier_tfidf_baseline"), terrier_metrics,
               {"model": terrier_config.get("model", "TF_IDF"), "index_elapsed_s": terrier_metrics["index_elapsed_s"]})
        RunWriter.write_run(terrier_results, str(result_dir / "MaamTeam_terrier_tfidf.txt"))

    model_rows = []
    for scheme in config["models"]:
        doc_scheme, query_scheme = scheme.lower().split(".")
        metrics, _ = evaluate(SmartVSM(documents, best_preproc, doc_scheme, query_scheme), train, train_ids, qrels, top_k)
        model_rows.append((scheme, metrics))
        record("models", scheme, metrics, {"model": scheme})
    best_model, _ = max(model_rows, key=lambda entry: entry[1]["MAP"])
    pd.DataFrame([metrics | {"experiment": scheme} for scheme, metrics in model_rows]).to_csv(
        result_dir / "MaamTeam_stage2_models.csv", index=False
    )

    best_doc, best_query = best_model.lower().split(".")
    slope_rows = []
    for slope in config["pivot_slopes"]:
        metrics, _ = evaluate(SmartVSM(documents, best_preproc, best_doc, best_query, slope), train, train_ids, qrels, top_k)
        slope_rows.append((slope, metrics))
        record("parameters", f"{best_model}_s{slope}", metrics, {"model": best_model, "pivot_slope": slope})
    best_slope, _ = max(slope_rows, key=lambda entry: entry[1]["MAP"])
    pd.DataFrame([metrics | {"experiment": f"{best_model}_s{slope}"} for slope, metrics in slope_rows]).to_csv(
        result_dir / "MaamTeam_stage3_parameters.csv", index=False
    )

    feedback_rows = []
    for item in config["rocchio"]:
        feedback = RocchioConfig(**item)
        metrics, _ = evaluate(SmartVSM(documents, best_preproc, best_doc, best_query, best_slope), train, train_ids, qrels, top_k, feedback)
        feedback_rows.append((item, metrics))
        record("rocchio", str(item), metrics, {"model": best_model, "pivot_slope": best_slope})
    pd.DataFrame([metrics | {"experiment": str(item)} for item, metrics in feedback_rows]).to_csv(
        result_dir / "MaamTeam_stage4_rocchio.csv", index=False
    )

    combined_rows = []
    for scheme in config["models"]:
        doc_scheme, query_scheme = scheme.lower().split(".")
        for item in config["rocchio"]:
            metrics, _ = evaluate(SmartVSM(documents, best_preproc, doc_scheme, query_scheme, best_slope), train, train_ids, qrels, top_k, RocchioConfig(**item))
            combined_rows.append((scheme, item, metrics))
            record("combined", f"{scheme}+{item}", metrics, {"model": scheme, "pivot_slope": best_slope})
    best_model, best_feedback_data, _ = max(combined_rows, key=lambda entry: entry[2]["MAP"])
    best_doc, best_query = best_model.lower().split(".")
    pd.DataFrame([metrics | {"model": scheme, "feedback": str(item)} for scheme, item, metrics in combined_rows]).sort_values("MAP", ascending=False).to_csv(
        result_dir / "MaamTeam_stage5_best_combination.csv", index=False
    )

    final_engine = SmartVSM(documents, best_preproc, best_doc, best_query, best_slope)
    final_feedback = RocchioConfig(**best_feedback_data)
    final_results = []
    for query in queries:
        final_results.extend(final_engine.search_with_rocchio(query, final_feedback, top_k))
    RunWriter.write_run(final_results, str(result_dir / "MaamTeam_results.txt"))
    held_out_results = [result for result in final_results if result.qid in held_out_ids]
    final_config = {
        "selected_system": f"SmartVSM {best_model} + Rocchio",
        "preprocessing": asdict(best_preproc),
        "smart_scheme": {"doc": best_doc, "query": best_query, "pivot_slope": best_slope},
        "rocchio": asdict(final_feedback), "top_k": top_k,
        "terrier_baseline": {"training": terrier_metrics, "all_topics": terrier_all_metrics},
        "held_out": Evaluator.evaluate_all(held_out_results, held_out_ids, qrels),
        "all_topics": Evaluator.evaluate_all(final_results, query_ids, qrels),
    }
    with open(result_dir / "MaamTeam_best_config.json", "w", encoding="utf-8") as stream:
        json.dump(final_config, stream, indent=2)
    pd.DataFrame(rows).to_csv(result_dir / "experiment_summary.csv", index=False)
    with open(result_dir / "experiment_summary.json", "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    pd.DataFrame(timings).to_csv(benchmark_dir / "timing.csv", index=False)
    with open(benchmark_dir / "timing.json", "w", encoding="utf-8") as stream:
        json.dump(timings, stream, indent=2)
    print(json.dumps(final_config["all_topics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
