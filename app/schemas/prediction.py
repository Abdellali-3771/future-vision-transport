from pydantic import BaseModel

class PredictResponse(BaseModel):
    width: int
    height: int
    mask_png_base64: str

class ModelInfo(BaseModel):
    model_name: str
    framework: str
    input_shape: list
