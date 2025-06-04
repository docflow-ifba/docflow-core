import json
import logging
from kafka import KafkaConsumer
from app.config.config import KAFKA_EMBED_TOPIC, KAFKA_QUESTION_TOPIC, KAFKA_URL
from app.services.message_handler import handle_embedding_message, handle_question_message

logger = logging.getLogger("kafka-consumer")

def __create_consumer(topic, group_id='docflow-core-group'):
    return KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_URL,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id=group_id
    )

def start_embedding_consumer():
    consumer = __create_consumer(KAFKA_EMBED_TOPIC)
    for message in consumer:
        try:
            handle_embedding_message(message.value)
        except Exception as e:
            logging.exception("❗ Erro ao processar mensagem Kafka")

def start_question_consumer():
    consumer = __create_consumer(KAFKA_QUESTION_TOPIC)
    for message in consumer:
        try:
            handle_question_message(message.value)
        except Exception as e:
            logging.exception("❗ Erro ao processar pergunta no Kafka")