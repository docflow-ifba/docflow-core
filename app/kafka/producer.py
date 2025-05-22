import logging
from kafka import KafkaProducer
import json
from app.config.config import KAFKA_EMBED_RESULT_TOPIC, KAFKA_URL
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger("kafka-producer")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_URL,
    client_id='docflow-core-producer',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_result(message):
    producer.send(KAFKA_EMBED_RESULT_TOPIC, value=message)
    producer.flush()

    logger.info(f"✅ Enviado resultado para o tópico Kafka")