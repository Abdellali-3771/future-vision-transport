from fastapi import APIRouter, UploadFile, File

from app.models.predictor import predict_segmentation
from app.schemas.prediction import PredictResponse, ModelInfo
from app.utils.image_processing import preprocess_image, mask_to_base64

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
    # 🔹 Lecture & prétraitement
    image, arr, (w, h) = preprocess_image(await file.read())

    # 🔹 Prédiction modèle
    mask = predict_segmentation(arr[None, ...], original_size=(w, h))

    # 🔹 Génération du masque coloré redimensionné à la taille d'origine
    mask_png = mask_to_base64(mask, original_size=(w, h))

    # 🔹 Retour JSON à l’interface
    return {"width": w, "height": h, "mask_png_base64": mask_png}
