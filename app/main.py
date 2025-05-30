from fastapi import FastAPI

app = FastAPI(title="PDF Processor with Embedding")

from app.kafka.consumer import start_embedding_consumer, start_question_consumer
import threading
threading.Thread(target=start_embedding_consumer, daemon=True).start()
threading.Thread(target=start_question_consumer, daemon=True).start()
