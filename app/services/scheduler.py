from datetime import datetime, timedelta
from sqlalchemy import select, and_
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.database import AsyncSessionLocal
from app.models.domain import Subscription, FlaggedCar, FlagType

_days_before_exp = 5

async def check_expiring_subscriptions():
    
    #Calculate the target date
    target_date = (datetime.now() + timedelta(days=_days_before_exp)).date()

    #Open a fresh database session
    async with AsyncSessionLocal() as db:
        #1. Find all subscriptions expiring exacly on the target date
        stmt = select(Subscription).where(Subscription.expiration_date == target_date)
        result = await db.execute(stmt)
        exp_subs = result.scalars().all()
    
    if not exp_subs:
        print(f"No subscriptions expiring in {_days_before_exp} days.")
        
    #2. Flag each car
    flags_created = 0
    for sub in exp_subs:
        existing_flag_stmt = select(FlaggedCar).where(
           and_(FlaggedCar.car_registration_no == sub.car_registration_no,
                FlaggedCar.type == FlagType.subscription_close_to_expiration,
                FlaggedCar.detected_at >= datetime.now() - timedelta(days=_days_before_exp + 1)
                )
        )
        existing_flag_result = await db.execute(existing_flag_stmt)

        if not existing_flag_result.scalars().first():
            #Create the warning flag
            warning_flag = FlaggedCar(
                type=FlagType.subscription_close_to_expiration,
                car_registration_no=sub.car_registration_no,
                parking_id=sub.parking_id,
                detected_at=datetime.now(),
                requires_human_verification=False,
                verified_by_human=False,
                verification_notes=f"Automated warning: Subscription expires in {_days_before_exp} days."
            )
            db.add(warning_flag)
            flags_created += 1
        
        await db.commit()
        print(f"Expiration check completed. Created {flags_created} warning flags.")
        
scheduler = AsyncIOScheduler()

def start_scheduler():
    #For testing we run this every minute 
    #Later we change it to days=1
    scheduler.add_job(check_expiring_subscriptions, 'interval', minutes=2)
    scheduler.start()