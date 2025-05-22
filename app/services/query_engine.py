import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config.config import EMBEDDING_DEVICE, EMBEDDING_INDEX_PATH, EMBEDDING_MODEL, LLM_API_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

def query_embedding(prompt: str, document_id: str) -> str:
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE}
    )

    index_path = f"{EMBEDDING_INDEX_PATH}/{document_id}"
    if not os.path.exists(index_path):
        return f"Índice FAISS para o documento {document_id} não encontrado."

    vs = FAISS.load_local(
        index_path, embedder, allow_dangerous_deserialization=True
    )
    docs = vs.similarity_search(prompt, k=3)
    context = "\n".join([f"Seção: {d.metadata['section']}\nConteúdo: {d.page_content}" for d in docs])

    print(f"\n\nContexto encontrado: {context}\n\n")

    res = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Você é um assistente que ajuda a responder perguntas em português com base nas informações fornecidas."},
                {"role": "user", "content": f"{prompt}\nContexto:\n{context}"}
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": False
        }
    )
    return res.json()["choices"][0]["message"]["content"]
