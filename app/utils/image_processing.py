import io, base64
import numpy as np
from PIL import Image

CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70),
    (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

def preprocess_image(file_bytes: bytes, target_size=(224, 224)):
    """Prépare une image pour la prédiction (redimension automatique vers 224x224)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Sauvegarde taille originale (pour potentielle re-projection du masque)
    w, h = img.size

    # 🔹 Redimensionnement automatique pour correspondre au modèle
    resized = img.resize(target_size[::-1])  # (width, height)
    arr = np.array(resized, dtype=np.float32) / 255.0

    return img, arr, (w, h)


def mask_to_base64(mask: np.ndarray):
    """Convertit le masque numpy en image RGB encodée base64."""
    # Squeeze si le masque a une dimension supplémentaire (ex: (1,224,224))
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # 🔹 Convertit les indices en couleurs
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(CITYSCAPES_PALETTE):
        rgb[mask == i] = color

    # 🔹 Encode en base64
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
