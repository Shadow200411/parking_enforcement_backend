from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.schemas.payloads import DetectionCreate, FlaggedCarResponse, FlagVerificationUpdate
from app.services.decision_engine import process_detection
from app.models.domain import FlaggedCar, Parking

router = APIRouter(tags=["Enforcement"])

@router.post("/detections", status_code=201)
async def receive_detection(detection: DetectionCreate, db: AsyncSession = Depends(get_db)):
    """
    Receives a plate detection from the AI camera.
    Passes it to the Decision Engine to determine if it's a violation.
    """
    new_flag = await process_detection(db, detection)
    
    if new_flag:
        #Added these 2 lines so we get the data after the database has commited it and generated the id
        await db.commit()
        await db.refresh(new_flag)
        return {"status": "flagged", "flag_id": new_flag.id, "type": new_flag.type}
    
    return {"status": "ignored", "message": "Legally parked or dupilcate detection."}

@router.get("/flags", response_model=List[FlaggedCarResponse])
async def get_all_flags(db: AsyncSession = Depends(get_db)):
    """
    Used by the Frontend Dashboard to get a list of all the flagged cars,
    sorted by the oldest detections first.
    """
    stmt = select(FlaggedCar).order_by(FlaggedCar.detected_at.asc())
    result = await db.execute(stmt)
    flags = result.scalars().all()
    return flags

@router.patch("/flags/{flag_id}/verify", response_model=FlaggedCarResponse)
async def verify_flag(flag_id: int, update_data: FlagVerificationUpdate, db: AsyncSession = Depends(get_db)):
    """
    Used by a human officer to approve or reject a lo-confidence flag.
    """
    flag = await db.get(FlaggedCar, flag_id)
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")
    
    #Update the flag based on human input
    flag.verified_by_human = True
    flag.requires_human_verification = False
    flag.verification_notes = update_data.notes
    
    if not update_data.is_valid_violation:
        flag.verification_notes = f"[REJECTED BY OFFICER] {update_data.notes or ''}"
    
    await db.commit()
    await db.refresh(flag)
    return flag

@router.post("/seed", tags=["System"])
async def seed_database(db: AsyncSession = Depends(get_db)):
    """Temporary endpoint to create a test parking lot."""
    # Check if it already exists
    lot = await db.get(Parking, 1)
    if not lot:
        new_lot = Parking(
            name="Downtown Central",
            location="123 Main St",
            capacity=500
        )
        db.add(new_lot)
        await db.commit()
        return {"message": "Parking Lot #1 created successfully!"}
    
    return {"message": "Parking Lot already exists."}