from kafka import KafkaConsumer
import json
import base64
import logging
from app.config.config import KAFKA_ANSWER_TOPIC, KAFKA_EMBED_RESULT_TOPIC, KAFKA_EMBED_TOPIC, KAFKA_QUESTION_TOPIC, KAFKA_URL
from app.kafka.producer import send_result
from app.services.pdf_processor import process_pdf_and_convert
from app.config.logging import setup_logging
from app.services.query_engine import query_embedding_stream

setup_logging()
logger = logging.getLogger("kafka-consumer")

def start_embedding_consumer():
    consumer = KafkaConsumer(
        KAFKA_EMBED_TOPIC,
        bootstrap_servers=KAFKA_URL,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='docflow-core-group'
    )

    for message in consumer:
        try:
            data = message.value
            pdf_base64 = data.get("pdf_base64")
            docflow_notice_id = data.get("docflow_notice_id")

            if not pdf_base64:
                logging.error("❗ Mensagem sem pdf_base64")
                continue
            if not docflow_notice_id:
                logging.error("❗ Mensagem sem docflow_notice_id")
                continue

            logging.info(f"📄 Recebido PDF: {docflow_notice_id}")
            pdf_bytes = base64.b64decode(pdf_base64)
            content_md, clean_md, tables_md = process_pdf_and_convert(pdf_bytes, docflow_notice_id)

            response = {
                "docflow_notice_id": docflow_notice_id,
                "content_md": content_md,
                "clean_md": clean_md,
                "tables_md": tables_md
            }
            send_result(KAFKA_EMBED_RESULT_TOPIC, response)

        except Exception as e:
            logging.exception("❗ Erro ao processar mensagem Kafka")

def start_question_consumer():
    consumer = KafkaConsumer(
        KAFKA_QUESTION_TOPIC,
        bootstrap_servers=KAFKA_URL,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='docflow-core-group'
    )

    for message in consumer:
        try:
            data = message.value
            question = data.get("question")
            docflow_notice_id = data.get("docflow_notice_id")
            conversation_id = data.get("conversation_id")
            messages = data.get("messages")

            if not question or not docflow_notice_id or not conversation_id:
                logging.error("❗ Mensagem incompleta")
                continue

            logger.info(f"📄 Pergunta recebida: {question}")

            full_response = ""
            for chunk in query_embedding_stream(question, docflow_notice_id):
                full_response += chunk
                # partial_response = {
                #     "docflow_notice_id": docflow_notice_id,
                #     "conversation_id": conversation_id,
                #     "answer_chunk": chunk,
                #     "done": False
                # }
                # send_result(KAFKA_ANSWER_TOPIC, partial_response)

            send_result(KAFKA_ANSWER_TOPIC, {
                "docflow_notice_id": docflow_notice_id,
                "conversation_id": conversation_id,
                "answer": full_response,
                "done": True
            })

        except Exception as e:
            logging.exception("❗ Erro ao processar pergunta no Kafka")
