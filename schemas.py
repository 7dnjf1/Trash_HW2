from pydantic import BaseModel
from typing import List, Dict, Any

class DetectedItem(BaseModel):
    label: str
    category: str
    confidence: float
    guideline: str
    box: List[float]  # [xmin, ymin, xmax, ymax]

class ComplianceReport(BaseModel):
    total_detected: int
    epr_items: int

class WasteResponseV2(BaseModel):
    status: str
    message: str
    items: List[DetectedItem]
    compliance_report: ComplianceReport
    annotated_image_base64: str
