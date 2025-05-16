import os
import tempfile
import logging
from app.models.notice_model import NoticeModel
from app.config.database import SessionLocal
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from app.services.embedder import embed_document
from app.utils.file_utils import process_markdown

logger = logging.getLogger("pdf-processor")

def process_pdf_and_convert(name: str, deadline: str, pdf_bytes: bytes):
    logger.info(f"Iniciando processamento do PDF: {name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name
        logger.debug(f"PDF temporário criado em: {temp_pdf_path}")

    try:
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption()}
        )
        result = doc_converter.convert(temp_pdf_path)
        md_content = result.document.export_to_markdown()
        logger.info(f"Conversão do PDF '{name}' para Markdown concluída")
    finally:
        os.remove(temp_pdf_path)
        logger.debug(f"PDF temporário removido: {temp_pdf_path}")

    processed_md, tables_md = process_markdown(md_content)
    logger.info(f"Markdown processado para '{name}'")

    db = SessionLocal()
    new_doc = NoticeModel(
        name=name,
        deadline=deadline,
        pdf_bytes=pdf_bytes,
        content_markdown=md_content,
        clean_markdown=processed_md,
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    logger.info(f"Documento '{name}' persistido no banco com ID: {new_doc.notice_id}")

    embed_document(processed_md, str(new_doc.notice_id))
    logger.info(f"Embeddings criados para o documento: {new_doc.notice_id}")

    return str(new_doc.notice_id)
