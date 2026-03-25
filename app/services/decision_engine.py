from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.domain import Car, Subscription, FlaggedCar, FlagType
from app.schemas.payloads import DetectionCreate
from app.services.cache import detection_cache

#TO-DO: Need to move later to a config.py file
CONFIDENCE_THRESHOLD = 0.85

async def process_detection(db: AsyncSession, detection: DetectionCreate) -> FlaggedCar | None:
    """
    Evalueates and AI detection. Returns a FlaggedCar object if a violation occurrred,
    or None if the car is legally parked and high-confidence.
    """

    #1. Check the daily cahce to prevent spam
    if not detection_cache.should_process_detection(
        detection.car_registration_no,
        detection.parking_id,
        detection.confidence_score
    ):
        return None #We ignore the duplicate. TO-DO: Discuss if this is the right method of handling this case.
    
    #2.Check for the "Ghost Car" (if the car is not in the database, explanation bellow)
    car = await db.get(Car, detection.car_registration_no)
    if not car:
        #We must create a dummy car so our flagged_cars Foreign Key dosen't crash
        car = Car(
            registration_no=detection.car_registration_no,
            make="Unknown",
            model="Unknown",
            color="Unknown"
        )
        db.add(car)
        await db.flush() #Flushes to Db so we can use it immediately without commiting yet
    
    #3. Determine if confidence is high enough for auto-flagging
    is_confident = detection.confidence_score >= CONFIDENCE_THRESHOLD
    requires_human = not is_confident
    
    #4. Fetch all subscriptions for this car
    stmt = select(Subscription).where(Subscription.car_registration_no == car.registration_no)
    result = await db.execute(stmt)
    subs = result.scalars().all()
    
    #5.Business Logic Routing 
    today = datetime.now().date()
    flag_type = None
    
    if not subs:
        #Case D: Car has never had a subscription anywhere
        flag_type = FlagType.no_subscription
    else:
        #Check if they have a valid sub for THIS specific parking lot right now
        has_valid_local_sub = any(
            sub.parking_id == detection.parking_id and
            sub.begin_date <= today and
            sub.expiration_date >= today
            for sub in subs
        )
        
        if has_valid_local_sub:
            #The car is completly legal, however we check the AI confidence and decide if we flagg it for review
            #in case of a misread plate of a car that is legaly parked
            if requires_human:
                flag_type = FlagType.no_subscription #Default to no_subscription for review
            else:
                return None #Legal car, high confidence score -> we ignore it.
    
        else:
            #They have subs, but none are valid for this lot today.
            #We chack to see if they have a subscription for this spot that expired or (*****)
            has_expired_local_sub = any(
                sub.parking_id == detection.parking_id and
                sub.expiration_date < today
                for sub in subs
            )
            
            if has_expired_local_sub:
                flag_type = FlagType.subscription_expired
            else:
                #(*****) They have and active sub, but for a different lot
                flag_type = FlagType.car_in_wrong_parking
        
    #6. Create the Violation Flag
    new_flag = FlaggedCar(
        type=flag_type,
        car_registration_no=car.registration_no,
        parking_id=detection.parking_id,
        detected_at=datetime.now(),
        confidence_score=detection.confidence_score,
        evidence_image_url=detection.evidence_image_url,
        requires_human_verification=requires_human,
        verified_by_human=False
    )
    
    db.add(new_flag)
    
    #7.Update Cache
    detection_cache.mark_as_flagged(
        plate=car.registration_no,
        parking_id=detection.parking_id,
        confidence=detection.confidence_score,
        was_auto_flagged=not requires_human
    )
    
    return new_flag