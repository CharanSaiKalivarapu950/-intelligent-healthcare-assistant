from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.image import router as image_router
from backend.app.api.rag import router as rag_router

app = FastAPI(
    title="Intelligent Healthcare Assistant",
    version="1.0.0",
    description="Educational project only; not a medical diagnosis tool.",
)
app.include_router(health_router, prefix="/api")
app.include_router(image_router, prefix="/api")
app.include_router(rag_router, prefix="/api")

@app.get("/")
def root():
    return {
        "name": "Intelligent Healthcare Assistant",
        "status": "running",
        "disclaimer": "Educational use only. Not for clinical decisions.",
    }
