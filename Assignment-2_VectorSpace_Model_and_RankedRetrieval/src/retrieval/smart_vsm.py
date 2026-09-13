from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List

from nltk.stem import PorterStemmer

from ..data.models import Document, Query, RankedResult


@dataclass(frozen=True)
class PreprocConfig:
    stem: bool = True
    stopwords: bool = True
    title_weight: int = 3
    include_author: bool = False
    include_bib: bool = False
    dedupe_title: bool = False


@dataclass(frozen=True)
class RocchioConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.0
    fb_docs: int = 5
    fb_terms: int = 15
    neg_docs: int = 0
    iterations: int = 1
    first_pass_k: int = 0


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those",
}
_TOKEN = re.compile(r"[a-z0-9]+")
_PORTER = PorterStemmer()


def _stem(token: str) -> str:
    return _PORTER.stem(token)


def tokenize(text: str, config: PreprocConfig) -> List[str]:
    terms = _TOKEN.findall(text.lower())
    if config.stopwords:
        terms = [term for term in terms if term not in _STOPWORDS]
    if config.stem:
        terms = [_stem(term) for term in terms]
    return [term for term in terms if len(term) >= 2 or term.isdigit()]


class SmartVSM:
    def __init__(self, documents: Iterable[Document], config: PreprocConfig = PreprocConfig(),
                 doc_scheme: str = "lnu", query_scheme: str = "ltc", pivot_slope: float = 0.15):
        self.documents = list(documents)
        self.config = config
        self.doc_scheme = doc_scheme.lower()
        self.query_scheme = query_scheme.lower()
        self.pivot_slope = pivot_slope
        self.doc_terms: Dict[str, Counter] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.postings: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.df: Counter = Counter()
        self.average_unique_terms = 0.0
        self._validate_scheme(self.doc_scheme)
        self._validate_scheme(self.query_scheme)
        self._build()

    @staticmethod
    def _validate_scheme(scheme: str) -> None:
        if len(scheme) != 3 or scheme[0] not in "nlab" or scheme[1] not in "ntp" or scheme[2] not in "ncu":
            raise ValueError(f"Unsupported SMART scheme: {scheme!r}")

    def _document_text(self, document: Document) -> str:
        title = document.title
        body = document.text
        if self.config.dedupe_title and title:
            prefix = title.strip()
            if body.lower().startswith(prefix.lower()):
                body = body[len(prefix):]
        parts = [title] * max(0, self.config.title_weight) + [body]
        if self.config.include_author:
            parts.append(document.author)
        if self.config.include_bib:
            parts.append(document.bibliography)
        return " ".join(parts)

    def _build(self) -> None:
        for document in self.documents:
            terms = Counter(tokenize(self._document_text(document), self.config))
            self.doc_terms[document.docno] = terms
            for term in terms:
                self.df[term] += 1
        self.average_unique_terms = sum(len(terms) for terms in self.doc_terms.values()) / max(len(self.doc_terms), 1)
        for docno, terms in self.doc_terms.items():
            vector = self._weight(terms, self.doc_scheme, len(terms))
            self.doc_vectors[docno] = vector
            for term, weight in vector.items():
                self.postings[term][docno] = weight

    def _idf(self, term: str) -> float:
        return math.log((len(self.documents) + 1) / (self.df[term] + 1)) + 1.0

    def _weight(self, terms: Counter, scheme: str, unique_terms: int = 0) -> Dict[str, float]:
        values = {}
        for term, frequency in terms.items():
            tf = 1.0 + math.log(frequency) if scheme[0] == "l" and frequency else float(frequency)
            if scheme[0] == "b":
                tf = 1.0
            if len(scheme) > 1 and scheme[1] == "t":
                df = self._idf(term)
            elif len(scheme) > 1 and scheme[1] == "p":
                ratio = (len(self.documents) - self.df[term]) / max(self.df[term], 1)
                df = max(0.0, math.log(ratio))
            else:
                df = 1.0
            values[term] = tf * df
        if len(scheme) > 2 and scheme[2] == "c":
            norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
            values = {term: value / norm for term, value in values.items()}
        elif len(scheme) > 2 and scheme[2] == "u":
            denominator = ((1.0 - self.pivot_slope) * self.average_unique_terms
                           + self.pivot_slope * max(unique_terms, 1))
            values = {term: value / (denominator or 1.0) for term, value in values.items()}
        return values

    def _query_vector(self, query: str) -> Dict[str, float]:
        return self._weight(Counter(tokenize(query, self.config)), self.query_scheme)

    def search(self, query: Query, top_k: int = 1000) -> List[RankedResult]:
        vector = self._query_vector(query.query)
        scores = defaultdict(float)
        for term, query_weight in vector.items():
            for docno, doc_weight in self.postings.get(term, {}).items():
                scores[docno] += query_weight * doc_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [RankedResult(query.qid, docno, rank, score, "SmartVSM")
                for rank, (docno, score) in enumerate(ranked, 1)]

    def search_all(self, queries: Iterable[Query], top_k: int = 1000) -> List[RankedResult]:
        results = []
        for query in queries:
            results.extend(self.search(query, top_k))
        return results

    def search_with_rocchio(self, query: Query, feedback: RocchioConfig, top_k: int = 1000) -> List[RankedResult]:
        vector = self._query_vector(query.query)
        for _ in range(max(1, feedback.iterations)):
            initial = self.search(query, feedback.first_pass_k or max(top_k, feedback.fb_docs, feedback.neg_docs))
            positive = initial[:feedback.fb_docs]
            negative = initial[-feedback.neg_docs:] if feedback.neg_docs else []
            expanded = defaultdict(float, {term: feedback.alpha * weight for term, weight in vector.items()})
            for docs, coefficient in ((positive, feedback.beta), (negative, -feedback.gamma)):
                for result in docs:
                    for term, weight in self.doc_vectors.get(result.docno, {}).items():
                        expanded[term] += coefficient * weight / max(len(docs), 1)
            original = set(vector)
            kept = sorted(expanded, key=lambda term: (-expanded[term], term))
            kept = list(original) + [term for term in kept if term not in original]
            vector = {term: expanded[term] for term in kept[:feedback.fb_terms + len(original)] if expanded[term] > 0}
        scores = defaultdict(float)
        for term, query_weight in vector.items():
            for docno, doc_weight in self.postings.get(term, {}).items():
                scores[docno] += query_weight * doc_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [RankedResult(query.qid, docno, rank, score, "SmartVSM+Rocchio")
                for rank, (docno, score) in enumerate(ranked, 1)]

