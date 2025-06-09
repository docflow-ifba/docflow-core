import numpy as np
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger("query-engine")

class SemanticCache:
    def __init__(self, embedder, similarity_threshold=0.85, max_cache_size=100):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict[str, Tuple[np.ndarray, str, List[dict]]]] = {}
        self.query_history: Dict[str, List[str]] = {}

    def _compute_embedding(self, text: str) -> np.ndarray:
        return self.embedder.embed_query(text)
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def _get_query_hash(self, query: str) -> str:
        return str(hash(query))
    
    def get(self, query: str, document_id: str) -> Optional[Tuple[str, List[dict]]]:
        if document_id not in self.cache:
            return None
        
        query_embedding = self._compute_embedding(query)
        
        best_match = None
        best_similarity = 0
        
        for query_hash, (cached_embedding, response, context_docs) in self.cache[document_id].items():
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (response, context_docs)
        
        if best_match:
            logger.info(f"Cache hit for document {document_id} with similarity {best_similarity:.4f}")
            return best_match
        
        return None
    
    def put(self, query: str, document_id: str, response: str, context_docs: List[dict]):
        if document_id not in self.cache:
            self.cache[document_id] = {}
            self.query_history[document_id] = []
        
        query_hash = self._get_query_hash(query)
        query_embedding = self._compute_embedding(query)
        
        self.cache[document_id][query_hash] = (query_embedding, response, context_docs)
        self.query_history[document_id].append(query_hash)
        
        if len(self.query_history[document_id]) > self.max_cache_size:
            oldest_query_hash = self.query_history[document_id].pop(0)
            if oldest_query_hash in self.cache[document_id]:
                del self.cache[document_id][oldest_query_hash]
                logger.info(f"Removed oldest entry from cache for document {document_id}")
