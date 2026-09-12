from pathlib import Path
import re
from typing import List, Dict, Tuple
from .models import Document, Query, Qrel


class CranfieldParser:

    @staticmethod
    def _normalise_query_id(qid: str) -> str:
        return str(int(qid)) if qid.isdigit() else qid

    @staticmethod
    def parse_documents(doc_file: str) -> Tuple[List[Document], Dict[str, Document]]:
        documents = []
        doc_dict = {}
        
        with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        doc_sections = re.split(r'(?m)^\.I\s+', content)[1:]
        
        for section in doc_sections:
            if not section.strip():
                continue
            
            lines = section.split('\n')
            docno = lines[0].strip()
            
            if not docno:
                continue
            
            title = ""
            author = ""
            bibliography = ""
            text = ""

            current_field = None
            current_content = []
            
            for line in lines[1:]:
                if line.startswith('.T'):
                    if current_field:
                        value = '\n'.join(current_content).strip()
                        if current_field == 'T':
                            title = value
                        elif current_field == 'A':
                            author = value
                        elif current_field == 'B':
                            bibliography = value
                        elif current_field == 'W':
                            text = value
                    current_field = 'T'
                    current_content = [line[2:].strip()]
                elif line.startswith('.A'):
                    if current_field:
                        value = '\n'.join(current_content).strip()
                        if current_field == 'T':
                            title = value
                        elif current_field == 'A':
                            author = value
                        elif current_field == 'B':
                            bibliography = value
                        elif current_field == 'W':
                            text = value
                    current_field = 'A'
                    current_content = [line[2:].strip()]
                elif line.startswith('.B'):
                    if current_field:
                        value = '\n'.join(current_content).strip()
                        if current_field == 'T':
                            title = value
                        elif current_field == 'A':
                            author = value
                        elif current_field == 'B':
                            bibliography = value
                        elif current_field == 'W':
                            text = value
                    current_field = 'B'
                    current_content = [line[2:].strip()]
                elif line.startswith('.W'):
                    if current_field:
                        value = '\n'.join(current_content).strip()
                        if current_field == 'T':
                            title = value
                        elif current_field == 'A':
                            author = value
                        elif current_field == 'B':
                            bibliography = value
                        elif current_field == 'W':
                            text = value
                    current_field = 'W'
                    current_content = [line[2:].strip()]
                else:
                    if current_field:
                        current_content.append(line)
            
            if current_field:
                value = '\n'.join(current_content).strip()
                if current_field == 'T':
                    title = value
                elif current_field == 'A':
                    author = value
                elif current_field == 'B':
                    bibliography = value
                elif current_field == 'W':
                    text = value
            
            doc = Document(
                docno=docno,
                title=title,
                author=author,
                bibliography=bibliography,
                text=text
            )
            
            if docno in doc_dict:
                raise ValueError(f"Duplicate document ID: {docno}")
            
            documents.append(doc)
            doc_dict[docno] = doc
        
        return documents, doc_dict


    @staticmethod
    def parse_queries(query_file: str) -> Tuple[List[Query], Dict[str, Query]]:
        queries = []
        query_dict = {}
        
        with open(query_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
  
        query_sections = re.split(r'(?m)^\.I\s+', content)[1:]
        
        for section in query_sections:
            if not section.strip():
                continue
            
            lines = section.split('\n')
            # Keep the original Cranfield identifier (for example, 001 or 365).
            qid = lines[0].strip()
            
            if not qid:
                continue
            
            query_text = ""
            for i, line in enumerate(lines[1:], 1):
                if line.startswith('.W'):
                    # Get remaining lines as query text
                    query_text = '\n'.join(lines[i+1:]).strip()
                    break
            
            if not query_text:
                raise ValueError(f"Query {qid} has no text")
            
            query = Query(qid=qid, query=query_text)
            
            if qid in query_dict:
                raise ValueError(f"Duplicate query ID: {qid}")
            
            queries.append(query)
            query_dict[qid] = query
        
        return queries, query_dict


    @staticmethod
    def parse_qrels(
        qrel_file: str,
        query_ids: List[str] = None,
    ) -> Tuple[List[Qrel], Dict[Tuple[str, str], Qrel]]:
        """Parse qrels and optionally translate positional query numbers.

        The supplied Cranfield qrels number the selected queries 1..225, while
        cran.qry stores their original collection IDs. When query_ids is
        provided, qrel query 1 maps to query_ids[0], qrel query 2 to
        query_ids[1], and so on.
        """
        qrels = []
        qrel_dict = {}
        
        with open(qrel_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                raw_qid = parts[0]
                if query_ids is not None:
                    if not raw_qid.isdigit() or not 1 <= int(raw_qid) <= len(query_ids):
                        raise ValueError(
                            f"Qrel query number {raw_qid!r} cannot be mapped to "
                            f"{len(query_ids)} query records"
                        )
                    qid = query_ids[int(raw_qid) - 1]
                else:
                    qid = CranfieldParser._normalise_query_id(raw_qid)
                docno = parts[1]
                relevance = int(parts[2])
                
                # Skip non-judged documents (relevance == -1)
                if relevance < 0:
                    continue
                
                qrel = Qrel(qid=qid, docno=docno, relevance=relevance)
                qrels.append(qrel)
                qrel_dict[(qid, docno)] = qrel
        
        return qrels, qrel_dict


def validate_data(documents: Dict[str, Document], queries: Dict[str, Query],
                  qrels: List[Qrel], verbose: bool = False) -> None:
    
    if len(documents) != len(set(documents.keys())):
        raise ValueError("Duplicate document IDs found")

    if len(queries) != len(set(queries.keys())):
        raise ValueError("Duplicate query IDs found")
  
    unknown_queries = set()
    unknown_docs = set()
    
    for qrel in qrels:
        if qrel.qid not in queries:
            unknown_queries.add(qrel.qid)
        if qrel.docno not in documents:
            unknown_docs.add(qrel.docno)
    
    if unknown_docs:
        raise ValueError(f"Qrels reference unknown document IDs: {unknown_docs}")
   
    if unknown_queries and verbose:
        print(f"Warning: {len(unknown_queries)} qrels reference unknown query IDs")

    for doc in documents.values():
        if not doc.text or doc.text.isspace():
            pass
    
    for query in queries.values():
        if not query.query or query.query.isspace():
            raise ValueError(f"Query {query.qid} has empty query text")
