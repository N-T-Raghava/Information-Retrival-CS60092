from dataclasses import dataclass


@dataclass
class Document:
    docno: str
    title: str
    author: str
    bibliography: str
    text: str

    def __post_init__(self):
        if not self.docno:
            raise ValueError("Document must have a docno")


@dataclass
class Query:
    qid: str
    query: str

    def __post_init__(self):
        if not self.qid:
            raise ValueError("Query must have a qid")
        if not self.query:
            raise ValueError(f"Query {self.qid} must have text")


@dataclass
class Qrel:
    qid: str
    docno: str
    relevance: int

    def __post_init__(self):
        if not self.qid:
            raise ValueError("Qrel must have a qid")
        if not self.docno:
            raise ValueError("Qrel must have a docno")
        if not isinstance(self.relevance, int) or self.relevance < 0:
            raise ValueError(f"Qrel must have non-negative integer relevance, got {self.relevance}")


@dataclass
class RankedResult:
    qid: str
    docno: str
    rank: int
    score: float
    run_name: str
