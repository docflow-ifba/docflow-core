import os
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter

def extract_sections(md_content: str) -> List[Dict]:
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
    return sections

def split_into_chunks(sections: List[Dict]) -> List[Document]:
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for sec in sections:
        chunks = splitter.create_documents([sec['content']])
        for chunk in chunks:
            docs.append(Document(page_content=chunk.page_content, metadata={
                'source': './output/processed_document.md',
                'section': sec['title']
            }))
    return docs

def embed_document(md_path: str):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = extract_sections(content)
    docs = split_into_chunks(sections)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local("./output/document_faiss_index")
