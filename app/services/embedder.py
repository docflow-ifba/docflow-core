import logging
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from app.config.config import EMBEDDING_DEVICE, EMBEDDING_INDEX_PATH, EMBEDDING_MODEL

logger = logging.getLogger("embedder")

class DocumentEmbedder:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )

    def embed_document(self, content: str, docflow_notice_id: str):
        """Main method to process and embed a document"""
        logger.info(f"Iniciando embedding do documento: {docflow_notice_id}")
        sections = self._extract_sections(content)
        docs = self._split_into_chunks(sections, docflow_notice_id)
        self._create_and_save_index(docs, docflow_notice_id)

    def _extract_sections(self, md_content: str) -> List[Dict]:
        """Extract sections from markdown content"""
        logger.debug("Extraindo seções do conteúdo Markdown...")
        sections = []
        current = None
        for line in md_content.split('\n'):
            if line.startswith('## '):
                if current: sections.append(current)
                current = {'title': line[3:].strip(), 'content': []}
            elif current:
                current['content'].append(line)
        if current: sections.append(current)
        for sec in sections:
            sec['content'] = '\n'.join(sec['content']).strip()
        logger.info(f"{len(sections)} seções extraídas do Markdown")
        return sections

    def _split_into_chunks(self, sections: List[Dict], doc_id: str) -> List[Document]:
        """Split sections into chunks for embedding"""
        logger.debug("Dividindo seções em chunks para embedding...")
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = []
        for sec in sections:
            chunks = splitter.create_documents([sec['content']])
            for chunk in chunks:
                docs.append(Document(page_content=chunk.page_content, metadata={
                    'source': doc_id,
                    'section': sec['title']
                }))
        logger.info(f"{len(docs)} chunks gerados a partir das seções")
        return docs

    def _create_and_save_index(self, docs: List[Document], docflow_notice_id: str):
        """Create and save FAISS index"""
        logger.debug("Gerando vetor e salvando índice FAISS...")
        vs = FAISS.from_documents(docs, self.embedder)
        index_path = f"{EMBEDDING_INDEX_PATH}/{docflow_notice_id}"
        vs.save_local(index_path)
        logger.info(f"Índice FAISS salvo em: {index_path}")