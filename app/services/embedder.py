import os
import logging
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter

logger = logging.getLogger("embedder")

def extract_sections(md_content: str) -> List[Dict]:
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

def split_into_chunks(sections: List[Dict], doc_id: str) -> List[Document]:
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

def embed_document(content: str, doc_id: str):
    logger.info(f"Iniciando embedding do documento: {doc_id}")
    sections = extract_sections(content)
    docs = split_into_chunks(sections, doc_id)

    logger.debug("Carregando modelo de embeddings...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    logger.debug("Gerando vetor e salvando índice FAISS...")
    vs = FAISS.from_documents(docs, embedder)
    index_path = f"./output/faiss_indexes/{doc_id}"
    vs.save_local(index_path)
    logger.info(f"Índice FAISS salvo em: {index_path}")
