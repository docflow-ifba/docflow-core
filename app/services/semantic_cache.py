import faiss
import numpy as np
from typing import Dict, Tuple, List, Optional
import pickle
import os

class SemanticCache:
    def __init__(self, embedding_model, similarity_threshold=0.85):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[int, Tuple[str, str, np.ndarray]] = {}  # id -> (query, response, embedding)
        self.index = faiss.IndexFlatIP(768)  # dimensão do embedding
        self.cache_file = "semantic_cache.pkl"
        self.load_cache()

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)

    def add_to_cache(self, query: str, response: str) -> None:
        embedding = self.get_embedding(query)
        embedding = embedding / np.linalg.norm(embedding)

        idx = len(self.cache)
        self.cache[idx] = (query, response, embedding)
        self.index.add(np.array([embedding], dtype=np.float32))
        self.save_cache()

    def find_similar_query(self, query: str) -> Optional[str]:
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        if self.index.ntotal == 0:
            return None

        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), 1)

        if D[0][0] >= self.similarity_threshold:
            cached_query, cached_response, _ = self.cache[int(I[0][0])]
            return cached_response

        return None

    def save_cache(self) -> None:
        with open(self.cache_file, 'wb') as f:
            pickle.dump((self.cache, self.index), f)

    def load_cache(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache, self.index = pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar cache: {e}")