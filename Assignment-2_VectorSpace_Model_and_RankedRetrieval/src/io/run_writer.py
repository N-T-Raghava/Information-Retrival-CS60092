from pathlib import Path
from typing import List, Dict
import json
from ..data.models import RankedResult


class RunWriter:

    @staticmethod
    def write_run(results: List[RankedResult], output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by qid and rank
        sorted_results = sorted(results, key=lambda x: (x.qid, x.rank))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in sorted_results:
                line = f"{result.qid} Q0 {result.docno} {result.rank} {result.score:.6f} {result.run_name}\n"
                f.write(line)
    
    
    @staticmethod
    def write_metrics(metrics: Dict[str, float], output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for metric_name, value in sorted(metrics.items()):
                f.write(f"{metric_name}: {value:.6f}\n")


class ResultLoader:

    @staticmethod
    def load_run(run_file: str) -> List[RankedResult]:
        results = []
        with open(run_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                result = RankedResult(
                    qid=parts[0],
                    docno=parts[2],
                    rank=int(parts[3]),
                    score=float(parts[4]),
                    run_name=parts[5]
                )
                results.append(result)
        
        return results
