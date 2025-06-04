import os
import tempfile
import logging
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from app.services.embedder import DocumentEmbedder
from app.utils.file_utils import process_markdown

logger = logging.getLogger("pdf-processor")

class PdfProcessor:
    def __init__(self):
        self.embedder = DocumentEmbedder()

    def process_pdf_and_convert(self, pdf_bytes: bytes, docflow_notice_id: str):
        logger.info(f"Iniciando processamento do PDF: {docflow_notice_id}")
        content_md = self._convert_pdf_to_markdown(pdf_bytes)
        clean_md, tables_md = process_markdown(content_md)

        self.embedder.embed_document(clean_md, docflow_notice_id)
        logger.info(f"Embeddings criados para o documento: {docflow_notice_id}")

        return content_md, clean_md, tables_md

    def _convert_pdf_to_markdown(self, pdf_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
            logger.debug(f"PDF temporário criado em: {temp_pdf_path}")

        try:
            doc_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption()}
            )
            result = doc_converter.convert(temp_pdf_path)
            content_md = result.document.export_to_markdown()
            logger.info("Conversão do PDF para Markdown concluída")
            return content_md
        finally:
            os.remove(temp_pdf_path)
            logger.debug(f"PDF temporário removido: {temp_pdf_path}")

def process_pdf_and_convert(pdf_bytes: bytes, docflow_notice_id: str):
    processor = PdfProcessor()
    return processor.process_pdf_and_convert(pdf_bytes, docflow_notice_id)