import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def query_embedding(prompt: str):
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vs = FAISS.load_local(
        "./output/document_faiss_index", embedder, allow_dangerous_deserialization=True
    )
    docs = vs.similarity_search(prompt, k=3)
    context = "\n".join([f"Seção: {d.metadata['section']}\nConteúdo: {d.page_content}" for d in docs])

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
