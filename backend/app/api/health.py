from fastapi import APIRouter
from backend.app.ml.inference import classifier

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status": "ok", "service": "intelligent-healthcare-assistant"}

@router.get("/model/status")
def model_status():
    return {"loaded": classifier.is_loaded, "checkpoint": classifier.checkpoint_path}
