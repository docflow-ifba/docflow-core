from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="PDF Processor with Embedding")
app.include_router(api_router)

from app.kafka.consumer import start_consumer
import threading
threading.Thread(target=start_consumer, daemon=True).start()
