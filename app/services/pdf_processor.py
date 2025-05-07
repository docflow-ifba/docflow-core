import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from app.services.embedder import embed_document
from app.utils.file_utils import extract_tables_and_replace, process_markdown

def process_pdf_and_convert(pdf_path: str):
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    converted_md = os.path.join(output_dir, "converted_document.md")
    processed_md = os.path.join(output_dir, "processed_document.md")

    if not os.path.exists(converted_md):
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption()}
        )
        result = doc_converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()

        with open(converted_md, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    process_markdown(converted_md, processed_md)
    embed_document(processed_md)
