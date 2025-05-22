import os
import tempfile
import logging
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from app.services.embedder import embed_document
from app.utils.file_utils import process_markdown

logger = logging.getLogger("pdf-processor")

def process_pdf_and_convert(pdf_bytes: bytes, docflow_notice_id: str):
    logger.info(f"Iniciando processamento do PDF: {docflow_notice_id}")

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
        logger.info(f"Conversão do PDF '{docflow_notice_id}' para Markdown concluída")
    finally:
        os.remove(temp_pdf_path)
        logger.debug(f"PDF temporário removido: {temp_pdf_path}")

    processed_md, tables_md = process_markdown(md_content)
    logger.info(f"Markdown processado para '{docflow_notice_id}'")

    embed_document(processed_md, docflow_notice_id)
    logger.info(f"Embeddings criados para o documento: {docflow_notice_id}")
