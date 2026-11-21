from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.rag.service import rag_service

router = APIRouter(tags=["rag"])

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

@router.post("/rag/ingest")
def ingest_documents():
    try:
        return rag_service.ingest_directory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

@router.post("/rag/ask")
def ask(request: AskRequest):
    try:
        return rag_service.answer(request.question, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}")
