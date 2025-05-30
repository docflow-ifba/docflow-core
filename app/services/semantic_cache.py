from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import threading

class SemanticCache:
    def __init__(self, embedder: HuggingFaceEmbeddings):
        self.embedder = embedder
        self.lock = threading.Lock()
        self.prompts = []
        self.responses = []
        self.vectorstore = None  # inicial vazio

    def get(self, prompt: str, threshold: float = 0.90) -> str | None:
        with self.lock:
            if not self.prompts or self.vectorstore is None:
                return None

            results = self.vectorstore.similarity_search_with_score(prompt, k=1)
            if results and results[0][1] >= threshold:
                idx = self.prompts.index(results[0][0].page_content)
                return self.responses[idx]

            return None

    def add(self, prompt: str, response: str):
        with self.lock:
            self.prompts.append(prompt)
            self.responses.append(response)
            if self.vectorstore is None:
                # Cria o índice na primeira adição
                self.vectorstore = FAISS.from_texts([prompt], self.embedder)
            else:
                self.vectorstore.add_texts([prompt])

