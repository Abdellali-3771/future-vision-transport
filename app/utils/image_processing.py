import io, base64
import numpy as np
from PIL import Image
import cv2  # pip install opencv-python

# ============================================================
# 🎨 Palette simplifiée Cityscapes (8 super-classes)
# ============================================================
CITYSCAPES_PALETTE = [
    (128, 64, 128),   # 0 - Flat : routes, trottoirs
    (220, 20, 60),    # 1 - Human : piétons, cyclistes
    (0, 0, 142),      # 2 - Vehicle : voitures, bus, motos
    (70, 70, 70),     # 3 - Construction : bâtiments, murs
    (250, 170, 30),   # 4 - Object : panneaux, poteaux
    (107, 142, 35),   # 5 - Nature : arbres, végétation
    (70, 130, 180),   # 6 - Sky : ciel
    (0, 0, 0)         # 7 - Void : indéfini
]


# ============================================================
# 🧩 Prétraitement de l'image
# ============================================================
def preprocess_image(file_bytes: bytes, target_size=(224, 224)):
    """
    Prépare une image pour la prédiction (redimension automatique vers 224x224).
    Retourne :
      - Image PIL originale
      - Tableau numpy normalisé pour le modèle
      - Taille d’origine (w, h)
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Sauvegarde taille originale
    w, h = img.size

    # 🔹 Redimensionnement automatique pour correspondre au modèle
    resized = img.resize(target_size[::-1])  # (width, height)
    arr = np.array(resized, dtype=np.float32) / 255.0

    return img, arr, (w, h)


# ============================================================
# 🎨 Conversion du masque vers image encodée base64
# ============================================================
def mask_to_base64(mask: np.ndarray, original_size=None):
    """
    Convertit un masque numpy (H,W) ou (H,W,3) en image PNG encodée base64.
    Gère les formats avec batch et redimension optionnel.
    """
    # 🔹 Supprime les dimensions inutiles (batch, channel)
    mask = np.squeeze(mask)

    # 🔹 Si la sortie du modèle est une probabilité (float32), on prend l'indice max
    if mask.ndim == 3 and mask.shape[-1] > 3:
        mask = np.argmax(mask, axis=-1)

    # 🔹 Si c’est un masque RGB déjà prêt
    if mask.ndim == 3 and mask.shape[-1] == 3:
        rgb = mask.astype(np.uint8)
    else:
        # Sinon, on convertit les indices en RGB via la palette simplifiée
        mask = mask.astype(np.uint8)
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i, color in enumerate(CITYSCAPES_PALETTE):
            rgb[mask == i] = color

    # 🔹 Redimension si nécessaire (sans interpolation entre classes)
    if original_size is not None:
        rgb = cv2.resize(rgb, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)

    # 🔹 Sécurité : s'assurer que l'image est bien (H,W,3)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Format d'image invalide pour l'encodage : {rgb.shape}")

    # 🔹 Encode en base64 (PNG)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
