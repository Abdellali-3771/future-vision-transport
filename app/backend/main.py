from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import uvicorn

# 🔹 Import sans "app."
from app.routers import segmentation
from app.backend.config.settings import settings

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Cityscapes Segmentation API",
    description="API pour la segmentation sémantique d'images urbaines",
    version="1.0.0"
)

# Configuration CORS
origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    os.getenv("FRONTEND_URL", "*"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(
    segmentation.router,
    prefix="/api/v1/segmentation",
    tags=["segmentation"]
)

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API de segmentation sémantique Cityscapes",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/api/v1/segmentation/predict",
            "health": "/api/v1/segmentation/health",
            "model_info": "/api/v1/segmentation/model/info"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'application"""
    logger.info("Démarrage de l'API...")
    logger.info(f"Port: {settings.PORT}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=port, reload=False)
