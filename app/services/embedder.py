import logging
import os
from typing import List
from functools import lru_cache
from app.config.config import EMBEDDING_DEVICE, EMBEDDING_INDEX_PATH, EMBEDDING_MODEL
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter

logger = logging.getLogger("embedder")

class DocumentEmbedder:
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_device: str = EMBEDDING_DEVICE,
        index_path: str = EMBEDDING_INDEX_PATH
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.index_path = index_path
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.embedder = self._get_embedder()

    @lru_cache(maxsize=1)
    def _get_embedder(self):
        logger.info(f"Inicializando modelo de embeddings: {self.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.embedding_device}
        )

    def embed_document(self, content: str, doc_id: str) -> List[Document]:
        logger.info(f"Iniciando embedding do documento: {doc_id}")

        docs = self._split_into_chunks(content, doc_id)

        self._create_and_save_index(docs, doc_id)

        logger.info(f"Embedding do documento {doc_id} concluído com sucesso")

        return docs

    def _split_into_chunks(self, content: str, doc_id: str) -> List[Document]:
        logger.debug("Dividindo conteúdo em chunks para embedding...")

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        header_chunks = header_splitter.split_text(content)

        text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        documents = text_splitter.split_documents(header_chunks)

        docs = []
        for i, doc in enumerate(documents):
            metadata = {
                "document_id": doc_id,
                "source": doc_id,
                "chunk_id": i,
                **doc.metadata
            }

            docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))

        logger.info(f"{len(docs)} chunks gerados a partir do conteúdo")
        
        return docs

    def _create_and_save_index(self, docs: List[Document], doc_id: str) -> None:
        if not docs:
            logger.warning(f"Nenhum documento para criar embeddings para {doc_id}")
            return

        logger.debug(f"Gerando vetores para {len(docs)} chunks...")

        vector_store = FAISS.from_documents(docs, self.embedder)

        index_path = f"{self.index_path}/{doc_id}"

        vector_store.save_local(index_path)
        logger.info(f"Índice FAISS salvo em: {index_path}")