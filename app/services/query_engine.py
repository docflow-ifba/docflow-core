import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def query_embedding(prompt: str, document_id: str) -> str:
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    index_path = f"./output/faiss_indexes/{document_id}"
    if not os.path.exists(index_path):
        return f"Índice FAISS para o documento {document_id} não encontrado."

    vs = FAISS.load_local(
        index_path, embedder, allow_dangerous_deserialization=True
    )
    docs = vs.similarity_search(prompt, k=3)
    context = "\n".join([f"Seção: {d.metadata['section']}\nConteúdo: {d.page_content}" for d in docs])

    print(f"\n\nContexto encontrado: {context}\n\n")

    res = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "deepseek-r1-distill-qwen-14b",
            "messages": [
                {"role": "system", "content": "Você é um assistente que ajuda a responder perguntas em português com base nas informações fornecidas."},
                {"role": "user", "content": f"{prompt}\nContexto:\n{context}"}
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": False
        }
    )
    return res.json()["choices"][0]["message"]["content"]
