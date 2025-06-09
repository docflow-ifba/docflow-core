import json
import logging
import atexit
from kafka import KafkaProducer
from app.config.config import KAFKA_URL
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger("kafka-producer")

class KafkaProducerSingleton:
    _instance = None
    _producer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._producer = KafkaProducer(
                bootstrap_servers=KAFKA_URL,
                client_id='docflow-core-producer',
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10,
                compression_type='gzip'
            )
            atexit.register(cls._cleanup)
        return cls._instance

    @classmethod
    def _cleanup(cls):
        if cls._producer:
            cls._producer.close()

    def send_message(self, topic: str, message: dict):
        future = self._producer.send(topic, value=message)
        self._producer.flush()
        logger.info(f"✅ Enviado resultado para o tópico {topic}")
        return future

producer_instance = KafkaProducerSingleton()

def send_message(topic: str, message: dict):
    return producer_instance.send_message(topic, message)