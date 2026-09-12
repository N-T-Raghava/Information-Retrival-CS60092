#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parser import CranfieldParser, validate_data
from src.retrieval.terrier_pipeline import build_index


def main():
    parser = argparse.ArgumentParser(description="Build IR index")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--index-path", default="indexes/cranfield", help="Index output path")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Lowercase text")
    parser.add_argument("--stopwords", action="store_true", default=True, help="Remove stopwords")
    parser.add_argument("--stemming", action="store_true", default=True, help="Apply stemming")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    doc_file = data_dir / "cran.all.1400"
    query_file = data_dir / "cran.qry"
    qrel_file = data_dir / "cranqrel"
    
    if not doc_file.exists():
        print(f"Error: {doc_file} not found")
        return 1
    
    print("Loading data...")
    start_time = time.time()
    
    documents, doc_dict = CranfieldParser.parse_documents(str(doc_file))
    queries, query_dict = CranfieldParser.parse_queries(str(query_file))
    qrels, qrel_dict = CranfieldParser.parse_qrels(
        str(qrel_file), query_ids=[query.qid for query in queries]
    )

    validate_data(doc_dict, query_dict, qrels)
    
    parse_time = time.time() - start_time
    print(f"Parsed {len(documents)} documents, {len(queries)} queries, {len(qrels)} qrels in {parse_time:.2f}s")
    
    print("Building index...")
    index, build_time = build_index(
        documents,
        args.index_path,
        stopwords=args.stopwords,
        stemming=args.stemming,
        force=args.force,
    )
    print(f"Index built in {build_time:.2f}s")
    print(f"Terrier index: {args.index_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
