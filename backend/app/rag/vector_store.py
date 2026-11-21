import json
import os
import numpy as np
import faiss
from backend.app.core.config import settings

class VectorStore:
    def __init__(self, directory=None):
        self.directory = directory or settings.rag_index_dir
        self.index = None
        self.records = []

    def build(self, vectors, records):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(records):
            raise ValueError("Vectors and records must be aligned 2D data.")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.records = records

    def save(self):
        if self.index is None:
            raise RuntimeError("Index is empty.")
        os.makedirs(self.directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.directory, "index.faiss"))
        with open(os.path.join(self.directory, "records.json"), "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def load(self):
        index_path = os.path.join(self.directory, "index.faiss")
        records_path = os.path.join(self.directory, "records.json")
        if not os.path.isfile(index_path) or not os.path.isfile(records_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(records_path, encoding="utf-8") as f:
            self.records = json.load(f)
        return True

    def search(self, vector, top_k=4):
        if self.index is None:
            raise RuntimeError("No index loaded. Ingest documents first.")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(vector, min(top_k, len(self.records)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({**self.records[idx], "score": float(score)})
        return results
