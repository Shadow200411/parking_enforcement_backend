from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from typing import Optional
import base64
from pathlib import Path

from app.core.plates import normalize_registration_no
from app.models.domain import FlagType

BASE_DIR = Path(__file__).resolve().parent.parent

#1.Incoming data (from the AI camera)
class DetectionCreate(BaseModel):
    """The JSON structure we expect from the AI model to send us"""
    car_registration_no: str = Field(..., descritpion="The detected licence plate")
    parking_id: int = Field(..., description="The ID of the parking lot the police is currently in")
    
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="The confidence score of the detection")
    
    evidence_image_url: Optional[str] = Field(None, description="Path to the saved evidence image")

    @field_validator("car_registration_no")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        normalized = normalize_registration_no(value)
        if not normalized:
            raise ValueError("car_registration_no must contain letters or digits")
        return normalized


class RawDetectionCreate(BaseModel):
    """Raw evidence captured by the officer device before OCR is applied."""
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG of the vehicle")
    parking_id: int = Field(..., description="The ID of the parking lot the officer is currently in")
    latitude: Optional[float] = Field(None, description="GPS latitude from the officer's device")
    longitude: Optional[float] = Field(None, description="GPS longitude from the officer's device")
    timestamp: Optional[str] = Field(None, description="ISO-8601 capture time")
    officer_id: Optional[str] = Field(None, description="Officer identifier from the capture device")
    device_id: Optional[str] = Field(None, description="Device identifier from the capture device")


class RawDetectionResponse(BaseModel):
    """Combined AI analysis and enforcement outcome."""
    status: str
    flag_id: Optional[int] = None
    type: Optional[FlagType] = None
    message: Optional[str] = None
    request_id: str
    timestamp: str
    parking_id: int
    detected_plate: str
    confidence_score: float
    requires_human_verification: bool
    plate_obscured: bool
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    evidence_image_url: str
    analysis_notes: str
    model_version: str

    @field_validator("evidence_image_url")
    @classmethod
    def convert_url_to_base64(cls, v: str) -> str:
        if not v or v.startswith("data:"):
            return v
        filepath = BASE_DIR / v.lstrip("/")
        if filepath.exists():
            with open(filepath, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
        return v


class ParkingResponse(BaseModel):
    """Parking lot metadata used by the capture client."""
    id: int
    name: str
    location: str
    capacity: int

    model_config = ConfigDict(from_attributes=True)
    
#2.Outgoing data (to the frontend dashboard)
class FlaggedCarResponse(BaseModel):
    """The JSON structure we will send back to the frontend"""
    id: int
    type: FlagType
    car_registration_no: str
    parking_id: int
    detected_at: datetime
    
    confidence_score: Optional[float]
    evidence_image_url: Optional[str]
    
    requires_human_verification: bool
    verified_by_human: bool
    verification_notes: Optional[str]
    
    #Pydantic v2 cofigur