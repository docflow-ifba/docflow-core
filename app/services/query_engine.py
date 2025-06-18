import os
import json
import logging
import requests
import time
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
            yield from self._simulate_stream(cached_response)
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
            
            if full_response.strip():
                self.semantic_cache.put(prompt, document_id, full_response, [
                    {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
                ])
                
        except Exception as e:
            logger.exception(f"Erro ao processar consulta para documento {document_id}: {e}")
            yield "Erro ao processar sua consulta."

    def _simulate_stream(self, text: str, chunk_size: int = 10, delay: float = 0.05) -> Generator[str, None, None]:
        words = []
        for line in text.split('\n'):
            words.extend(line.split())
            words.append('\n')
        if words and words[-1] == '\n':
            words.pop()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i > 0:
                chunk = " " + chunk
            yield chunk
            time.sleep(delay)

    def _load_vector_store(self, document_id: str) -> Optional[FAISS]:
        if document_id in self.vector_stores:
            return self.vector_stores[document_id]

        index_path = os.path.join(EMBEDDING_INDEX_PATH, document_id)
        if not os.path.exists(index_path):
            logger.warning(f"Índice FAISS não encontrado para o documento: {document_id}")
            return None

        try:
            vector_store = FAISS.load_local(index_path, self.embedder, allow_dangerous_deserialization=True)
            self.vector_stores[document_id] = vector_store
            return vector_store
        except Exception as e:
            logger.error(f"Erro ao carregar vector store para {document_id}: {e}")
            return None

    def _build_context(self, docs: list) -> str:
        return "\\n\\n".join(
            f"Seção: {doc.metadata.get('section', 'N/A')}\\nConteúdo: {doc.page_content}"
            for doc in docs
        )

    def _build_messages(self, prompt: str, context: str, messages: List[MessageRequestDTO]) -> List[dict]:
        system_message = {
            "role": "system",
            "content": f"""
                Você é um assistente especialista em responder perguntas sobre documentos oficiais como editais e PDFs.

                🔥 Regras obrigatórias:
                1. Toda sua comunicação — pensamento, raciocínio e resposta — deve ser 100% em PORTUGUÊS.
                2. Nunca use palavras ou estruturas em inglês.
                3. Use exclusivamente as informações do CONTEXTO abaixo.
                4. Se a resposta não estiver no contexto, responda claramente que não foi possível encontrar.

                🧠 Lembre-se: pense, raciocine e fale em português.

                Contexto:
                {context}
            """
        }

        if messages:
            user_messages = [{"role": m.role, "content": m.content} for m in messages]
            user_messages.append({"role": "user", "content": prompt})
            return [system_message] + user_messages

        return [
            system_message,
            {"role": "user", "content": prompt}
        ]

    def _stream_llm_response(self, chat_messages: List[dict]) -> Generator[str, None, None]:
        payload = {
            "model": LLM_MODEL,
            "messages": chat_messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": True
        }

        logger.info(f"Enviando requisição para LLM: {LLM_API_URL}, payload: {payload}")

        try:
            with requests.post(
                LLM_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                        
                    line_data = line.replace("data: ", "").strip()
                    if line_data == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(line_data)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                        
        except requests.RequestException as e:
            logger.error(f"Erro na requisição à LLM: {e}")
            yield "Erro ao obter resposta da IA."

_engine_instance = None
_engine_lock = None

def get_query_engine() -> QueryEngine:
    global _engine_instance, _engine_lock
    if _engine_instance is None:
        if _engine_lock is None:
            import threading
            _engine_lock = threading.Lock()
        
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = QueryEngine()
    return _engine_instance

def query_embedding_stream(prompt: str, document_id: str, messages: List[MessageRequestDTO] = None) -> Generator[str, None, None]:
    engine = get_query_engine()
    yield from engine.query_embedding_stream(prompt, document_id, messages)