import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import Query
from src.retrieval.terrier_pipeline import load_index, retrieve
from src.io.run_writer import RunWriter


def load_queries_csv(query_file: str):
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        # Skip header if present
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t', 1)
            if len(parts) == 2:
                qid, query_text = parts
                queries.append(Query(qid=qid.strip(), query=query_text.strip()))
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Search with arbitrary queries")
    parser.add_argument("--index", required=True, help="Index path")
    parser.add_argument("--query-file", required=True, help="Query file (CSV: qid<tab>query)")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--top-k", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--model", default="tfidf", help="Retrieval model")
    parser.add_argument("--run-name", default="unknown_queries", help="Run name")
    
    args = parser.parse_args()
    
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries_csv(args.query_file)
    print(f"Loaded {len(queries)} queries")
    
    if len(queries) == 0:
        print("Error: No queries loaded")
        return 1
    
    print("Initializing retriever...")
    print(f"Running {args.model} retrieval...")
    start_time = time.time()
    models = {"tfidf": "TF_IDF", "TF_IDF": "TF_IDF"}
    if args.model not in models:
        print(f"Unknown model: {args.model}")
        return 1
    indexref = load_index(args.index)
    results, query_time = retrieve(
        indexref, queries, model=models[args.model], top_k=args.top_k, run_name=args.run_name
    )
    total_time = time.time() - start_time
    
    print(f"Retrieved {len(results)} results in {total_time:.2f}s")
    print(f"Average query time: {query_time/len(queries):.4f}s")
    
    print(f"Writing results to {args.output}...")
    RunWriter.write_run(results, args.output)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
