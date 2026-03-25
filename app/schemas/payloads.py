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