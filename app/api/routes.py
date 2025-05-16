from fastapi import APIRouter, Query
from app.services.query_engine import query_embedding

router = APIRouter()

@router.get("/query")
def search_in_embedding(
    prompt: str = Query(..., description="Prompt da consulta"),
    document_id: str = Query(..., description="ID do documento para consulta")
):
    return query_embedding(prompt, document_id)
