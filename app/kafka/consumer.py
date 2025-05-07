from kafka import KafkaConsumer
import json
from app.services.pdf_processor import process_pdf_and_convert

def start_consumer():
    consumer = KafkaConsumer(
        'pdf-topic',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id='pdf-processor-group'
    )
    for message in consumer:
        pdf_path = message.value.get("pdf_path")
        if pdf_path:
            print(f"📄 Recebido PDF: {pdf_path}")
            process_pdf_and_convert(pdf_path)
