import os
import json
import logging
import requests
from typing import Generator, List, Optional
from functools import lru_cache
from app.services.semantic_cache import SemanticCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.config import (
    EMBEDDING_DEVICE, EMBEDDING_INDEX_PATH, EMBEDDING_MODEL,
    LLM_API_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE,
)
from app.dto.message_request import MessageRequestDTO

logger = logging.getLogger("query-engine")

class QueryEngine:
    def __init__(self):
        self.embedder = self._get_embedder()
        self.semantic_cache = SemanticCache(self.embedder)
        self.vector_stores = {}

    @lru_cache(maxsize=1)
    def _get_embedder(self):
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )

    def query_embedding_stream(self, prompt: str, document_id: str, messages: List[MessageRequestDTO] = None) -> Generator[str, None, None]:
        logger.info(f"Iniciando consulta (stream) para o documento: {document_id}")
        messages = messages or []

        cache_result = self.semantic_cache.get(prompt, document_id)
        if cache_result:
            cached_response, _ = cache_result
            logger.info(f"Usando resposta em cache para documento {document_id}")
            yield cached_response
            return

        vector_store = self._load_vector_store(document_id)
        if vector_store is None:
            yield f"Índice FAISS para o documento {document_id} não encontrado."
            return

        docs = vector_store.similarity_search(prompt, k=3)
        context = self._build_context(docs)
        chat_messages = self._build_messages(prompt, context, messages)

        full_response = ""
        try:
            for chunk in self._stream_llm_response(chat_messages):
                full_response += chunk
                yield chunk
            
            self.semantic_cache.put(prompt, document_id, full_response, [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ])
        except Exception:
            logger.exception(f"Erro ao processar consulta para documento {document_id}")
            yield "Erro ao processar sua consulta."

    def _load_vector_store(self, document_id: str) -> Optional[FAISS]:
        if document_id in self.vector_stores:
            return self.vector_stores[document_id]

        index_path = os.path.join(EMBEDDING_INDEX_PATH, document_id)
        if not os.path.exists(index_path):
            logger.warning(f"Índice FAISS não encontrado para o documento: {document_id}")
            return None

        vector_store = FAISS.load_local(index_path, self.embedder, allow_dangerous_deserialization=True)
        self.vector_stores[document_id] = vector_store
        return vector_store

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
            with requests.post(
                LLM_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=30
            ) as response:
                response.raise_for_status()
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
        except requests.RequestException as e:
            logger.error(f"Erro na requisição à LLM: {e}")
            yield "Erro ao obter resposta da IA."

_engine_instance = None

def query_embedding_stream(prompt: str, document_id: str, messages: List[MessageRequestDTO] = None) -> Generator[str, None, None]:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = QueryEngine()
    yield from _engine_instance.query_embedding_stream(prompt, document_id, messages)