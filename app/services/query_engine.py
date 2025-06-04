# services/query_engine.py
import os
import json
import logging
import requests
import numpy as np
from typing import Generator, List, Optional, Dict, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.config import (
    EMBEDDING_DEVICE,
    EMBEDDING_INDEX_PATH,
    EMBEDDING_MODEL,
    LLM_API_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from app.dto.message_request import MessageRequestDTO

logger = logging.getLogger("query-engine")

class SemanticCache:
    def __init__(self, embedder, similarity_threshold=0.85, max_cache_size=100):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict[str, Tuple[np.ndarray, str, List[dict]]]] = {}  # document_id -> {query_hash: (embedding, response, context_docs)}
        self.query_history: Dict[str, List[str]] = {}  # document_id -> [query_hash1, query_hash2, ...]

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
        
        # Add to cache
        self.cache[document_id][query_hash] = (query_embedding, response, context_docs)
        self.query_history[document_id].append(query_hash)
        
        # Enforce cache size limit
        if len(self.query_history[document_id]) > self.max_cache_size:
            oldest_query_hash = self.query_history[document_id].pop(0)
            if oldest_query_hash in self.cache[document_id]:
                del self.cache[document_id][oldest_query_hash]
                logger.info(f"Removed oldest entry from cache for document {document_id}")

class QueryEngine:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )
        self.semantic_cache = SemanticCache(self.embedder)

    def query_embedding_stream(self, prompt: str, document_id: str, messages: List[MessageRequestDTO] = None) -> Generator[str, None, None]:
        logger.info(f"Iniciando consulta (stream) para o documento: {document_id}")

        if messages is None:
            messages = []

        # Check cache first
        cache_result = self.semantic_cache.get(prompt, document_id)
        if cache_result:
            cached_response, cached_docs = cache_result
            logger.info(f"Usando resposta em cache para documento {document_id}")
            yield cached_response
            return

        # Cache miss, proceed with normal flow
        vector_store = self._load_vector_store(document_id)
        if vector_store is None:
            yield f"Índice FAISS para o documento {document_id} não encontrado."
            return

        docs = vector_store.similarity_search(prompt, k=3)
        context = self._build_context(docs)
        chat_messages = self._build_messages(prompt, context, messages)

        # Collect the full response to cache it
        full_response = ""
        for chunk in self._stream_llm_response(chat_messages):
            full_response += chunk
            yield chunk
        
        # Cache the response
        self.semantic_cache.put(prompt, document_id, full_response, [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ])

    def _load_vector_store(self, document_id: str) -> Optional[FAISS]:
        index_path = os.path.join(EMBEDDING_INDEX_PATH, document_id)
        if not os.path.exists(index_path):
            logger.warning(f"Índice FAISS não encontrado para o documento: {document_id}")
            return None

        return FAISS.load_local(index_path, self.embedder, allow_dangerous_deserialization=True)

    def _build_context(self, docs: list) -> str:
        return "\n".join(
            f"Seção: {doc.metadata.get('section', 'N/A')}\nConteúdo: {doc.page_content}"
            for doc in docs
        )

    def _build_messages(self, prompt: str, context: str, messages: List[MessageRequestDTO]) -> List[dict]:
        system_message = {
            "role": "system",
            "content": f"Você é um assistente que responde perguntas com base no contexto abaixo.\n\nContexto:\n{context}"
        }

        if messages:
            user_messages = [{"role": m.role, "content": m.content} for m in messages]
            user_messages.append({"role": "user", "content": prompt})
            return [system_message] + user_messages

        return [
            {"role": "system", "content": "Você é um assistente que responde perguntas com base no contexto abaixo."},
            {"role": "user", "content": f"{prompt}\nContexto:\n{context}"}
        ]

    def _stream_llm_response(self, chat_messages: List[dict]) -> Generator[str, None, None]:
        payload = {
            "model": LLM_MODEL,
            "messages": chat_messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": True
        }

        try:
            response = requests.post(
                LLM_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Erro na requisição à LLM: {e}")
            yield "Erro ao obter resposta da IA."
            return

        response.encoding = 'utf-8'
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                try:
                    data = json.loads(line.replace("data: ", ""))
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

def query_embedding_stream(prompt: str, document_id: str, messages: List[MessageRequestDTO] = None) -> Generator[str, None, None]:
    engine = QueryEngine()
    yield from engine.query_embedding_stream(prompt, document_id, messages)