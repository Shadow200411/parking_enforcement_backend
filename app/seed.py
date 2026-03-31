"""
This was generated using ai.
Seed script — populates the database with realistic test data covering
every violation scenario the decision engine needs to handle.

Run with:
    docker compose run --rm api python seed.py

Scenarios covered:
    A) Car with valid subscription for the correct lot         → no flag
    B) Car with subscription for a DIFFERENT lot               → car_in_wrong_parking
    C) Car with expired subscription for the correct lot       → subscription_expired
    D) Car with no subscription at all                         → no_subscription
    E) Car with subscription expiring in 5 days                → subscription_close_to_expiration (scheduler)
    F) Car with valid sub but low AI confidence                → requires_human_verification
"""

import asyncio
from datetime import date, timedelta

from sqlalchemy import text
from app.core.database import AsyncSessionLocal
from app.models.domain import Car, Parking, Subscription, User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

today = date.today()


# ── Data definitions ──────────────────────────────────────────────────────────

PARKINGS = [
    {"id": 1, "name": "Central Plaza",    "location": "12 Main St",    "capacity": 200},
    {"id": 2, "name": "North Park",       "location": "88 North Ave",  "capacity": 150},
    {"id": 3, "name": "Mall Underground", "location": "5 Shopping Blvd","capacity": 300},
]

CARS = [
    # Scenario A — valid subscription for lot 1
    {"registration_no": "CJ-01-AAA", "make": "Dacia",      "model": "Logan",   "color": "White"},
    # Scenario B — has subscription but for lot 2, will be detected in lot 1
    {"registration_no": "CJ-02-BBB", "make": "Volkswagen", "model": "Golf",    "color": "Black"},
    # Scenario C — expired subscription for lot 1
    {"registration_no": "CJ-03-CCC", "make": "Renault",    "model": "Clio",    "color": "Red"},
    # Scenario D — no subscription at all
    {"registration_no": "CJ-04-DDD", "make": "Ford",       "model": "Focus",   "color": "Blue"},
    # Scenario E — subscription expiring in 5 days (scheduler warning)
    {"registration_no": "CJ-05-EEE", "make": "Toyota",     "model": "Yaris",   "color": "Silver"},
    # Scenario F — valid subscription but will be sent with low confidence score
    {"registration_no": "CJ-06-FFF", "make": "Skoda",      "model": "Octavia", "color": "Grey"},
    # Bonus — car with subscription for all 3 lots (should never be flagged)
    {"registration_no": "CJ-07-GGG", "make": "BMW",        "model": "320i",    "color": "Black"},
]

SUBSCRIPTIONS = [
    # A — valid, correct lot
    {
        "car_registration_no": "CJ-01-AAA",
        "parking_id": 1,
        "begin_date": today - timedelta(days=30),
        "expiration_date": today + timedelta(days=60),
    },
    # B — valid, but for lot 2 (not lot 1 where car will be detected)
    {
        "car_registration_no": "CJ-02-BBB",
        "parking_id": 2,
        "begin_date": today - timedelta(days=10),
        "expiration_date": today + timedelta(days=20),
    },
    # C — expired for lot 1
    {
        "car_registration_no": "CJ-03-CCC",
        "parking_id": 1,
        "begin_date": today - timedelta(days=90),
        "expiration_date": today - timedelta(days=5),  # expired 5 days ago
    },
    # D — no subscription row at all
    # E — expiring in exactly 5 days (scheduler will pick this up)
    {
        "car_registration_no": "CJ-05-EEE",
        "parking_id": 1,
        "begin_date": today - timedelta(days=25),
        "expiration_date": today + timedelta(days=5),
    },
    # F — valid subscription (the low confidence comes from the AI, not the data)
    {
        "car_registration_no": "CJ-06-FFF",
        "parking_id": 1,
        "begin_date": today - timedelta(days=5),
        "expiration_date": today + timedelta(days=25),
    },
    # Bonus GGG — subscriptions for all 3 lots
    {
        "car_registration_no": "CJ-07-GGG",
        "parking_id": 1,
        "begin_date": today - timedelta(days=15),
        "expiration_date": today + timedelta(days=45),
    },
    {
        "car_registration_no": "CJ-07-GGG",
        "parking_id": 2,
        "begin_date": today - timedelta(days=15),
        "expiration_date": today + timedelta(days=45),
    },
    {
        "car_registration_no": "CJ-07-GGG",
        "parking_id": 3,
        "begin_date": today - timedelta(days=15),
        "expiration_date": today + timedelta(days=45),
    },
]

USERS = [
    {"fullname": "admin",   "password": "admin123"},
    {"fullname": "officer1","password": "officer123"},
    {"fullname": "officer2","password": "officer123"},
]


# ── Seed logic ────────────────────────────────────────────────────────────────

async def seed():
    async with AsyncSessionLocal() as db:

        # Wipe existing data in correct FK order
        await db.execute(text("DELETE FROM flagged_cars"))
        await db.execute(text("DELETE FROM subscriptions"))
        await db.execute(text("DELETE FROM cars"))
        await db.execute(text("DELETE FROM parkings"))
        await db.execute(text("DELETE FROM users"))
        await db.commit()
        print("Cleared existing data.")

        # Parkings
        for p in PARKINGS:
            db.add(Parking(**p))
        await db.commit()
        print(f"  Inserted {len(PARKINGS)} parking lots.")

        # Cars
        for c in CARS:
            db.add(Car(**c))
        await db.commit()
        print(f"  Inserted {len(CARS)} cars.")

        # Subscriptions
        for s in SUBSCRIPTIONS:
            db.add(Subscription(**s))
        await db.commit()
        print(f"  Inserted {len(SUBSCRIPTIONS)} subscriptions.")

        # Users (with hashed passwords)
        for u in USERS:
            db.add(User(
                fullname=u["fullname"],
                password=pwd_context.hash(u["password"]),
            ))
        await db.commit()
        print(f"  Inserted {len(USERS)} users.")

        print("\nSeed complete. Test scenarios ready:")
        print("  CJ-01-AAA  in lot 1 → no violation (valid sub)")
        print("  CJ-02-BBB  in lot 1 → car_in_wrong_parking (sub is for lot 2)")
        print("  CJ-03-CCC  in lot 1 → subscription_expired")
        print("  CJ-04-DDD  in lot 1 → no_subscription")
        print("  CJ-05-EEE  in lot 1 → subscription_close_to_expiration (via scheduler)")
        print("  CJ-06-FFF  in lot 1 → requires_human_verification (send confidence < 0.85)")
        print("  CJ-07-GGG  in any lot → no violation (valid sub everywhere)")
        print("\nLogin credentials:")
        for u in USERS:
            print(f"  {u['fullname']} / {u['password']}")


if __name__ == "__main__":
    asyncio.run(seed())