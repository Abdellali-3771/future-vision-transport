from pydantic_settings import BaseSettings
from pathlib import Path
from typing import ClassVar


class Settings(BaseSettings):
    # 🧩 Attribut interne (non interprété comme un champ d'env)
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    # 🔹 Variables de configuration réelles
    MODEL_PATH: str = str(BASE_DIR / "model_weights" / "cityscapes_segmentation_model.keras")
    CLASS_MAPPING_PATH: str = str(BASE_DIR / "model_weights" / "class_mapping.json")
    ENVIRONMENT: str = "local"
    PORT: int = 8000


# Instance globale
settings = Settings()
