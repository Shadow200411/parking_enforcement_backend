from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional

from app.models.domain import FlagType


#1.Incoming data (from the AI camera)
class DetectionCreate(BaseModel):
    """The JSON structure we expect from the AI model to send us"""
    car_registration_no: str = Field(..., descritpion="The detected licence plate")
    parking_id: int = Field(..., description="The ID of the parking lot the police is currently in")
    
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="The confidence score of the detection")
    
    evidence_image_url: Optional[str] = Field(None, description="Path to the saved evidence image")


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
    
    #Pydantic v2 cofiguration to read SQLAlchemy database models directly
    model_config = ConfigDict(from_attributes=True)
    
#3.Verification update (from the human operator)
class FlagVerificationUpdate(BaseModel):
    """When a human reviews a low-confidence flag, they will send this."""
    is_valid_violation: bool = Field(...,description="Did the human confirm this is a real violation?")
    notes: Optional[str] = Field(None, description="Optional notes from the officer")
    corrected_plate: Optional[str] = Field(None, description="If the AI misread the plate, the human can type the real one here")
