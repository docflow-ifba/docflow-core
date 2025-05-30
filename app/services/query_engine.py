import os
from dataclasses import dataclass
from typing import Literal
import json
import logging
import requests
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

logger = logging.getLogger("query-engine")

@dataclass
class MessageRequestDTO:
    role: Literal["user", "system"]
    content: str

def query_embedding_stream(prompt: str, document_id: str, messages: list[MessageRequestDTO] = []):
    logger.info(f"Iniciando consulta (stream) para o documento: {document_id}")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE}
    )

    index_path = f"{EMBEDDING_INDEX_PATH}/{document_id}"
    if not os.path.exists(index_path):
        yield f"Índice FAISS para o documento {document_id} não encontrado."
        return

    vs = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    docs = vs.similarity_search(prompt, k=3)

    context = "\n".join([
        f"Seção: {d.metadata.get('section', 'N/A')}\nConteúdo: {d.page_content}" for d in docs
    ])

    headers = {"Content-Type": "application/json"}

    if messages:
        chat_messages = [{"role": "system", "content": f"Você é um assistente que responde perguntas com base no contexto abaixo.\n\nContexto:\n{context}"}]
        chat_messages.extend([{"role": m.role, "content": m.content} for m in messages])
        chat_messages.append({"role": "user", "content": prompt})
    else:
        chat_messages = [
            {"role": "system", "content": "Você é um assistente que responde perguntas com base no contexto abaixo."},
            {"role": "user", "content": f"{prompt}\nContexto:\n{context}"}
        ]

    payload = {
        "model": LLM_MODEL,
        "messages": chat_messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": True
    }

    response = requests.post(LLM_API_URL, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        yield "Erro ao obter resposta da IA."
        return

    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            try:
                data = json.loads(line.replace("data: ", ""))
                content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield content
            except json.JSONDecodeError:
                continue
