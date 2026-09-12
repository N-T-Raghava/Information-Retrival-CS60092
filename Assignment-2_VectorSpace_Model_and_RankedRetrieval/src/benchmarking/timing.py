import time
import json
from pathlib import Path
from typing import Dict, List


class Benchmark:
    """Records benchmarking information."""
    
    def __init__(self, name: str):
        self.name = name
        self.timings = {}
        self.metadata = {}
    
    def record(self, key: str, value: float) -> None:
        self.timings[key] = value
    
    def add_metadata(self, key: str, value) -> None:
        self.metadata[key] = value
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'timings': self.timings,
            'metadata': self.metadata
        }


class BenchmarkSuite:
    """Collection of benchmarks."""
    
    def __init__(self):
        self.benchmarks: List[Benchmark] = []
    
    def add(self, benchmark: Benchmark) -> None:
        self.benchmarks.append(benchmark)
    
    def save_json(self, output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [b.to_dict() for b in self.benchmarks]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_csv(self, output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.benchmarks:
            return
        
        with open(output_path, 'w') as f:
            keys = set()
            for b in self.benchmarks:
                keys.update(b.timings.keys())
                keys.update(b.metadata.keys())
            
            header = ['name'] + sorted(keys)
            f.write(','.join(header) + '\n')
            
            for b in self.benchmarks:
                row = [b.name]
                for key in sorted(keys):
                    value = b.timings.get(key) or b.metadata.get(key, '')
                    row.append(str(value))
                f.write(','.join(row) + '\n')
