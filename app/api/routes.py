from fastapi import APIRouter, Query
from app.services.query_engine import query_embedding

router = APIRouter()

@router.get("/query")
def search_in_embedding(prompt: str = Query(..., description="Prompt da consulta")):
    return query_embedding(prompt)
