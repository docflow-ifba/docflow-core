from kafka import KafkaConsumer
import json
import base64
import os
import logging
from app.config.config import KAFKA_TOPIC, KAFKA_URL
from app.services.pdf_processor import process_pdf_and_convert
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger("kafka-consumer")

def start_consumer():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_URL,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='pdf-processor-group'
    )

    for message in consumer:
        try:
            data = message.value
            pdf_base64 = data.get("pdf_base64")
            docflow_notice_id = data.get("docflow_notice_id")

            if pdf_base64 and docflow_notice_id:
                logging.info(f"📄 Recebido PDF: {docflow_notice_id}")
                pdf_bytes = base64.b64decode(pdf_base64)
                process_pdf_and_convert(pdf_bytes, docflow_notice_id)
            else:
                logging.warning("❌ Mensagem Kafka incompleta (name, deadline ou pdf_base64 ausente)")

        except Exception as e:
            logging.exception("❗ Erro ao processar mensagem Kafka")
