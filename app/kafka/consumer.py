import json
import logging
from contextlib import contextmanager
from kafka import KafkaConsumer
from app.config.config import KAFKA_EMBED_TOPIC, KAFKA_QUESTION_TOPIC, KAFKA_URL
from app.services.message_handler import handle_embedding_message, handle_question_message

logger = logging.getLogger("kafka-consumer")

@contextmanager
def create_consumer(topic: str):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_URL,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='docflow-core-consumer',
    )
    try:
        yield consumer
    finally:
        consumer.close()

def start_embedding_consumer():
    with create_consumer(KAFKA_EMBED_TOPIC) as consumer:
        for message in consumer:
            try:
                handle_embedding_message(message.value)
            except Exception:
                logger.exception("❗ Erro ao processar mensagem Kafka")

def start_question_consumer():
    with create_consumer(KAFKA_QUESTION_TOPIC) as consumer:
        for message in consumer:
            try:
                handle_question_message(message.value)
            except Exception:
                logger.exception("❗ Erro ao processar pergunta no Kafka")