import base64
import logging
from app.config.config import KAFKA_ANSWER_TOPIC, KAFKA_EMBED_RESULT_TOPIC
from app.kafka.producer import send_message
from app.services.pdf_processor import process_pdf_and_convert
from app.services.query_engine import query_embedding_stream

logger = logging.getLogger("message-handler")

def handle_embedding_message(data: dict):
    pdf_base64 = data.get("pdf_base64")
    docflow_notice_id = data.get("docflow_notice_id")

    if not pdf_base64:
        logger.error("❗ Mensagem sem pdf_base64")
        return
    if not docflow_notice_id:
        logger.error("❗ Mensagem sem docflow_notice_id")
        return

    logger.info(f"📄 Recebido PDF: {docflow_notice_id}")
    
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        content_md, clean_md = process_pdf_and_convert(pdf_bytes, docflow_notice_id)

        response = {
            "docflow_notice_id": docflow_notice_id,
            "content_md": content_md,
            "clean_md": clean_md
        }
        send_message(KAFKA_EMBED_RESULT_TOPIC, response)
    except Exception:
        send_message(KAFKA_EMBED_RESULT_TOPIC, { "docflow_notice_id": docflow_notice_id, "error": "Erro ao processar PDF" })
        logger.exception(f"Erro ao processar PDF {docflow_notice_id}")

def handle_question_message(data: dict):
    required_fields = ["prompt", "docflow_notice_id", "user_id", "answer_conversation_id"]
    
    if not all(data.get(field) for field in required_fields):
        logger.error("❗ Mensagem incompleta")
        return

    question = data["prompt"]
    docflow_notice_id = data["docflow_notice_id"]
    user_id = data["user_id"]
    answer_conversation_id = data["answer_conversation_id"]

    logger.info(f"📄 Pergunta recebida: {question}")

    try:
        full_response = ""
        for chunk in query_embedding_stream(question, docflow_notice_id):
            full_response += chunk
            
            partial_response = {
                "docflow_notice_id": docflow_notice_id,
                "user_id": user_id,
                "answer_conversation_id": answer_conversation_id,
                "answer": chunk,
                "done": False
            }
            send_message(KAFKA_ANSWER_TOPIC, partial_response)

        final_response = {
            "docflow_notice_id": docflow_notice_id,
            "user_id": user_id,
            "answer_conversation_id": answer_conversation_id,
            "answer": full_response,
            "done": True
        }
        send_message(KAFKA_ANSWER_TOPIC, final_response)
    except Exception:
        logger.exception(f"Erro ao processar pergunta para documento {docflow_notice_id}")