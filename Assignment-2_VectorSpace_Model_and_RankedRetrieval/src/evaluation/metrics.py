from typing import List, Dict, Tuple, Set
import numpy as np
from ..data.models import Qrel
from ..data.models import RankedResult


class Evaluator:
    # Relevance grade mapping
    # Original: 1=complete answer, 2=high relevance, 3=useful, 4=minimum
    # For nDCG, map to: 1->4, 2->3, 3->2, 4->1
    RELEVANCE_MAPPING = {
        1: 4,  # complete answer -> highest relevance score
        2: 3,  # high relevance
        3: 2,  # useful
        4: 1,  # minimum -> lowest relevance score
    }
    
    
    @staticmethod
    def build_qrel_dict(qrels: List[Qrel]) -> Dict[Tuple[str, str], Qrel]:
        qrel_dict = {}
        for qrel in qrels:
            qrel_dict[(qrel.qid, qrel.docno)] = qrel
        return qrel_dict
    
    
    @staticmethod
    def get_relevant_docs(qid: str, qrel_dict: Dict[Tuple[str, str], Qrel],
                         binary: bool = False) -> Set[str]:

        relevant = set()
        for (q, docno), qrel in qrel_dict.items():
            if q == qid:
                if binary:
                    if qrel.relevance <= 3:  # Grades 1,2,3 are relevant
                        relevant.add(docno)
                else:
                    if qrel.relevance <= 4:  # All grades
                        relevant.add(docno)
        return relevant
    
    
    @staticmethod
    def get_graded_relevance(qid: str, qrel_dict: Dict[Tuple[str, str], Qrel]) -> Dict[str, int]:
        relevance = {}
        for (q, docno), qrel in qrel_dict.items():
            if q == qid:
                # Map to DCG scale
                relevance[docno] = Evaluator.RELEVANCE_MAPPING.get(qrel.relevance, 0)
        return relevance
    
    
    @staticmethod
    def get_relevant_with_grades(qid: str, qrel_dict: Dict[Tuple[str, str], Qrel]) -> Dict[str, int]:
        relevance = {}
        for (q, docno), qrel in qrel_dict.items():
            if q == qid and qrel.relevance <= 3:  # Consider only grades 1-3
                relevance[docno] = Evaluator.RELEVANCE_MAPPING.get(qrel.relevance, 0)
        return relevance
    
    
    @staticmethod
    def precision_at_k(results: List[RankedResult], qid: str, k: int,
                      qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
        
        query_results = [r for r in results if r.qid == qid]
        query_results = sorted(query_results, key=lambda x: x.rank)[:k]
        
        relevant = Evaluator.get_relevant_docs(qid, qrel_dict, binary=True)
        
        if len(query_results) == 0:
            return 0.0
        
        hits = sum(1 for r in query_results if r.docno in relevant)
        return hits / len(query_results)
    
    
    @staticmethod
    def recall_at_k(results: List[RankedResult], qid: str, k: int,
                   qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
       
        query_results = [r for r in results if r.qid == qid]
        query_results = sorted(query_results, key=lambda x: x.rank)[:k]
        
        relevant = Evaluator.get_relevant_docs(qid, qrel_dict, binary=True)
        
        if len(relevant) == 0:
            return 0.0
        
        hits = sum(1 for r in query_results if r.docno in relevant)
        return hits / len(relevant)
    
    
    @staticmethod
    def average_precision(results: List[RankedResult], qid: str,
                         qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
        
        query_results = [r for r in results if r.qid == qid]
        query_results = sorted(query_results, key=lambda x: x.rank)
        
        relevant = Evaluator.get_relevant_docs(qid, qrel_dict, binary=True)
        
        if len(relevant) == 0:
            return 0.0
        
        ap = 0.0
        hits = 0
        for i, result in enumerate(query_results):
            if result.docno in relevant:
                hits += 1
                ap += hits / (i + 1)
        
        return ap / len(relevant)
    
    
    @staticmethod
    def reciprocal_rank(results: List[RankedResult], qid: str,
                       qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
        
        query_results = [r for r in results if r.qid == qid]
        query_results = sorted(query_results, key=lambda x: x.rank)
        
        relevant = Evaluator.get_relevant_docs(qid, qrel_dict, binary=True)
        
        for i, result in enumerate(query_results):
            if result.docno in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    
    @staticmethod
    def ndcg_at_k(results: List[RankedResult], qid: str, k: int,
                 qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
        
        query_results = [r for r in results if r.qid == qid]
        query_results = sorted(query_results, key=lambda x: x.rank)[:k]
        
        relevance_dict = Evaluator.get_graded_relevance(qid, qrel_dict)
        
        if len(relevance_dict) == 0:
            return 0.0

        dcg = 0.0
        for i, result in enumerate(query_results):
            rel = relevance_dict.get(result.docno, 0)
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # log2(i+2) because rank starts at 1

        ideal_relevances = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    
    @staticmethod
    def r_precision(results: List[RankedResult], qid: str,
                   qrel_dict: Dict[Tuple[str, str], Qrel]) -> float:
        
        relevant = Evaluator.get_relevant_docs(qid, qrel_dict, binary=True)
        r = len(relevant)
        
        if r == 0:
            return 0.0
        
        return Evaluator.precision_at_k(results, qid, r, qrel_dict)
    
    
    @staticmethod
    def evaluate_all(results: List[RankedResult], qids: List[str],
                    qrel_dict: Dict[Tuple[str, str], Qrel]) -> Dict[str, float]:

        metrics = {
            'MAP': [],
            'P@5': [],
            'P@10': [],
            'R@10': [],
            'MRR': [],
            'nDCG@10': [],
            'nDCG@20': [],
            'R-Precision': []
        }
        
        for qid in qids:
            metrics['MAP'].append(Evaluator.average_precision(results, qid, qrel_dict))
            metrics['P@5'].append(Evaluator.precision_at_k(results, qid, 5, qrel_dict))
            metrics['P@10'].append(Evaluator.precision_at_k(results, qid, 10, qrel_dict))
            metrics['R@10'].append(Evaluator.recall_at_k(results, qid, 10, qrel_dict))
            metrics['MRR'].append(Evaluator.reciprocal_rank(results, qid, qrel_dict))
            metrics['nDCG@10'].append(Evaluator.ndcg_at_k(results, qid, 10, qrel_dict))
            metrics['nDCG@20'].append(Evaluator.ndcg_at_k(results, qid, 20, qrel_dict))
            metrics['R-Precision'].append(Evaluator.r_precision(results, qid, qrel_dict))
        
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = np.mean(values) if values else 0.0
        
        return avg_metrics
