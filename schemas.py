from pydantic import BaseModel

class WasteClassificationResult(BaseModel):
    category: str
    confidence: float
    disposal_method: str
    fine_info: str

class WasteResponse(BaseModel):
    status: str
    message: str
    data: WasteClassificationResult
