from typing import Dict, List
import numpy as np

"""Simple placeholder for KG embeddings. Replace with PyKEEN training pipeline."""

class SimpleTransEStore:
    def __init__(self, dim: int = 100):
        self.dim = dim
        self._store: Dict[str, np.ndarray] = {}

    def get(self, qid: str) -> np.ndarray:
        if qid not in self._store:
            # Random initialization; in real system load trained embedding
            self._store[qid] = np.random.normal(0, 0.01, size=(self.dim,)).astype(np.float32)
        return self._store[qid]

    def batch_get(self, qids: List[str]) -> np.ndarray:
        return np.vstack([self.get(q) for q in qids])
