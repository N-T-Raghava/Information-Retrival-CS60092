from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import contextlib
import io
import time
import shutil
import pandas as pd
import pyterrier as pt

from ..data.models import Document, Query, RankedResult


def start_terrier() -> None:
    if not pt.started():
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pt.java.init()
            pt.set_property("terrier.logging", "ERROR")
            
        except Exception as error:
            if "JAVA_HOME" in str(error) or "find JDK" in str(error):
                raise RuntimeError(
                    "PyTerrier requires a JDK. Install Java 11+ and set JAVA_HOME "
                    "before running this assignment."
                ) from error
            raise


def documents_frame(documents: Iterable[Document]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "docno": document.docno,
            "text": f"{document.title} {document.text}",
        }
        for document in documents
    )


def build_index(
    documents: Iterable[Document],
    index_path: str,
    stopwords: bool = True,
    stemming: bool = True,
    force: bool = False,
) -> Tuple[object, float]:

    start_terrier()
    path = Path(index_path).resolve()
    data_properties = path / "data.properties"
    
    if force and path.exists():
        shutil.rmtree(path)

    if data_properties.exists():
        return pt.IndexFactory.of(str(path)), 0.0

    path.mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(
        str(path),
        meta={"docno": 32},
        text_attrs=["text"],
        stemmer="porter" if stemming else None,
        stopwords="terrier" if stopwords else None,
        blocks=False,
    )
    start = time.perf_counter()
    indexref = indexer.index(documents_frame(documents).to_dict("records"))
    
    return indexref, time.perf_counter() - start


def load_index(index_path: str) -> object:
    
    start_terrier()
    path = Path(index_path).resolve()
    if not (path / "data.properties").exists():
        raise FileNotFoundError(
            f"No Terrier index found at {path}. Run scripts/build_index.py first."
        )
        
    return pt.IndexFactory.of(str(path))


def retrieve(
    indexref: object,
    queries: Iterable[Query],
    model: str = "TF_IDF",
    top_k: int = 1000,
    c: float = 1.0,
    run_name: str = "terrier_tfidf",
) -> Tuple[List[RankedResult], float]:

    start_terrier()
    query_frame = pd.DataFrame(
        {"qid": query.qid, "query": query.query} for query in queries
    )
    
    controls: Dict[str, str] = {"c": str(c)} if model == "TF_IDF" else {}
    transformer = pt.BatchRetrieve(
        indexref,
        wmodel=model,
        num_results=top_k,
        controls=controls,
    )
    
    start_time = time.perf_counter()
    terrier_results = transformer.transform(query_frame)
    elapsed = time.perf_counter() - start_time

    results = [
        RankedResult(
            qid=str(row.qid),
            docno=str(row.docno),
            rank=int(row.rank) + 1,
            score=float(row.score),
            run_name=run_name,
        )
        for row in terrier_results.itertuples(index=False)
    ]
    
    return results, elapsed