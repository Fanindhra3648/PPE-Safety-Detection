"""Typed response schemas for the API."""
from typing import Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float

class ComplianceSummary(BaseModel):
    status: str
    helmet_detected: bool
    no_helmet_detected: bool
    vest_detected: bool
    no_vest_detected: bool
    violations: list[str]
    message: str

class ImageAnalysisResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    detections: list[DetectionItem]
    compliance: ComplianceSummary
    annotated_image_url: str
