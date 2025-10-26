from fastapi import APIRouter, UploadFile, File

# 🔹 Import direct (sans "app.") car tout est au même niveau
from models.predictor import predict_segmentation
from schemas.prediction import PredictResponse, ModelInfo
from utils.image_processing import preprocess_image, mask_to_base64

router = APIRouter()

@router.get("/health")
async def health():
    """Endpoint pour vérifier l’état de l’API"""
    return {"status": "ok", "message": "Segmentation API running"}

@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Retourne les informations du modèle"""
    return {
        "model_name": "Cityscapes Segmentation Model",
        "framework": "TensorFlow",
        "input_shape": [256, 512, 3],
    }

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint principal de prédiction"""
    image, arr, (w, h) = preprocess_image(await file.read())
    mask = predict_segmentation(arr[None, ...])
    mask_png = mask_to_base64(mask)
    return {"width": w, "height": h, "mask_png_base64": mask_png}
