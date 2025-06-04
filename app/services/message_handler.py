import base64
import logging
from app.config.config import KAFKA_ANSWER_TOPIC, KAFKA_EMBED_RESULT_TOPIC
from app.kafka.producer import send_message
from app.services.pdf_processor import process_pdf_and_convert
from app.services.query_engine import query_embedding_stream

logger = logging.getLogger("message-handler")

def handle_embedding_message(data):
    pdf_base64 = data.get("pdf_base64")
    docflow_notice_id = data.get("docflow_notice_id")

    if not pdf_base64:
        logging.error("❗ Mensagem sem pdf_base64")
        return
    if not docflow_notice_id:
        logging.error("❗ Mensagem sem docflow_notice_id")
        return

    logging.info(f"📄 Recebido PDF: {docflow_notice_id}")
    pdf_bytes = base64.b64decode(pdf_base64)
    content_md, clean_md, tables_md = process_pdf_and_convert(pdf_bytes, docflow_notice_id)

    response = {
        "docflow_notice_id": docflow_notice_id,
        "content_md": content_md,
        "clean_md": clean_md,
        "tables_md": tables_md
    }
    send_message(KAFKA_EMBED_RESULT_TOPIC, response)

def handle_question_message(data):
    question = data.get("prompt")
    docflow_notice_id = data.get("docflow_notice_id")
    user_id = data.get("user_id")
    answer_conversation_id = data.get("answer_conversation_id")
    messages = data.get("messages")

    if not question or not docflow_notice_id or not user_id or not answer_conversation_id:
        logging.error("❗ Mensagem incompleta")
        return

    logger.info(f"📄 Pergunta recebida: {question}")

    full_response = ""
    for chunk in query_embedding_stream(question, docflow_notice_id, messages):
        full_response += chunk
        partial_response = {
            "docflow_notice_id": docflow_notice_id,
            "user_id": user_id,
            "answer_conversation_id": answer_conversation_id,
            "answer": chunk,
            "done": False
        }
        send_message(KAFKA_ANSWER_TOPIC, partial_response)

    send_message(KAFKA_ANSWER_TOPIC, {
        "docflow_notice_id": docflow_notice_id,
        "user_id": user_id,
        "answer_conversation_id": answer_conversation_id,
        "answer": full_response,
        "done": True
    })