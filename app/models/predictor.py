import tensorflow as tf
import numpy as np
from backend.config.settings import settings  # 🔹 sans app.
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement du modèle
try:
    model = tf.keras.models.load_model(settings.MODEL_PATH, compile=False)
    logger.info("✅ Modèle chargé avec succès !")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

def predict_segmentation(image_array: np.ndarray):
    """Effectue une prédiction de segmentation"""
    if model is None:
        raise RuntimeError("Le modèle n'est pas chargé.")
    prediction = model.predict(image_array)
    mask = np.argmax(prediction, axis=-1)[0]
    return mask
