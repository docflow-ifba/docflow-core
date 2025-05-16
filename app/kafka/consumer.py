from kafka import KafkaConsumer
import json
import base64
import os
import logging
from app.services.pdf_processor import process_pdf_and_convert
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger("kafka-consumer")

def start_consumer():
    consumer = KafkaConsumer(
        'pdf-topic',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='pdf-processor-group'
    )

    for message in consumer:
        try:
            data = message.value
            name = data.get("name")
            deadline = data.get("deadline")
            pdf_base64 = data.get("pdf_base64")

            if name and pdf_base64 and deadline:
                logging.info(f"📄 Recebido PDF: {name} | deadline: {deadline}")
                pdf_bytes = base64.b64decode(pdf_base64)
                process_pdf_and_convert(name, deadline, pdf_bytes)
            else:
                logging.warning("❌ Mensagem Kafka incompleta (name, deadline ou pdf_base64 ausente)")

        except Exception as e:
            logging.exception("❗ Erro ao processar mensagem Kafka")
