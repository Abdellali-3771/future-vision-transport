import logging
import numpy as np
import tensorflow as tf
import cv2

from app.backend.config.settings import settings

# 🔹 Logger local
logger = logging.getLogger(__name__)

# 🔹 Palette officielle Cityscapes (19 classes)
CITYSCAPES_PALETTE = [
    (128, 64,128),  # Road
    (244, 35,232),  # Sidewalk
    (70,  70, 70),  # Building
    (102,102,156),  # Wall
    (190,153,153),  # Fence
    (153,153,153),  # Pole
    (250,170, 30),  # Traffic Light
    (220,220,  0),  # Traffic Sign
    (107,142, 35),  # Vegetation
    (152,251,152),  # Terrain
    (70,130,180),   # Sky
    (220, 20, 60),  # Person
    (255,  0,  0),  # Rider
    (0,   0,142),   # Car
    (0,   0, 70),   # Truck
    (0,  60,100),   # Bus
    (0,  80,100),   # Train
    (0,   0,230),   # Motorcycle
    (119, 11, 32),  # Bicycle
]

def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque d'indices (H,W) en image RGB (H,W,3) avec la palette Cityscapes"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(CITYSCAPES_PALETTE):
        rgb[mask == i] = color
    return rgb

# 🔹 Chargement du modèle au démarrage
try:
    model = tf.keras.models.load_model(settings.MODEL_PATH, compile=False)
    logger.info("✅ Modèle chargé avec succès !")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

def predict_segmentation(image_array: np.ndarray, original_size=None, return_colored=True):
    """Effectue une prédiction de segmentation et retourne un masque d’indices ou RGB"""
    if model is None:
        raise RuntimeError("Le modèle n'est pas chargé.")

    prediction = model.predict(image_array)[0]  # (H, W, C)
    mask = np.argmax(prediction, axis=-1).astype(np.uint8)  # (H, W)

    print("➡️ Classes détectées dans le masque:", np.unique(mask))

    if original_size:
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    return decode_segmentation_mask(mask) if return_colored else mask
