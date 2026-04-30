"""
Seed the demo database with parking lots, cars, subscriptions, and users.

Run with:
    docker compose exec web python -m app.seed

Optional overrides:
    Create app/seed_overrides.json based on app/seed_overrides.example.json
    to add or replace demo data with presentation-specific vehicles.
"""

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from passlib.context import CryptContext
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.core.plates import normalize_registration_no
from app.models.domain import Car, Parking, Subscription, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

today = date.today()
OVERRIDES_PATH = Path(__file__).with_name("seed_overrides.json")


PARKINGS = [
    {"id": 1, "name": "Central Plaza", "location": "12 Main St", "capacity": 200},
    {"id": 2, "name": "North Park", "location": "88 North Ave", "capacity": 150},
    {"id": 3, "name": "Mall Underground", "location": "5 Shopping Blvd", "capacity": 300},
    {"id": 4, "name": "Arena South", "location": "17 Stadium Way", "capacity": 260},
    {"id": 5, "name": "Airport Long Stay", "location": "1 Terminal Rd", "capacity": 500},
]

CARS = [
    {"registration_no": "CJ01AAA", "make": "Dacia", "model": "Logan", "color": "White"},
    {"registration_no": "CJ02BBB", "make": "Volkswagen", "model": "Golf", "color": "Black"},
    {"registration_no": "B999XZA", "make": "Renault", "model": "Clio", "color": "Red"},
    {"registration_no": "CJ04DDD", "make": "Ford", "model": "Focus", "color": "Blue"},
    {"registration_no": "CJ05EEE", "make": "Toyota", "model": "Yaris", "color": "Silver"},
    {"registration_no": "CJ82RST", "make": "Skoda", "model": "Octavia", "color": "Grey"},
    {"registration_no": "CJ07GGG", "make": "BMW", "model": "320i", "color": "Black"},
    {"registration_no": "B101AAA", "make": "Audi", "model": "A4", "color": "Gray"},
    {"registration_no": "B202BBB", "make": "Mercedes-Benz", "model": "C200", "color": "Black"},
    {"registration_no": "B303CCC", "make": "Tesla", "model": "Model 3", "color": "White"},
    {"registration_no": "TM404DDD", "make": "Ford", "model": "Transit", "color": "White"},
    {"registration_no": "IS505EEE", "make": "Hyundai", "model": "Tucson", "color": "Blue"},
    {"registration_no": "PH606FFF", "make": "Peugeot", "model": "3008", "color": "Green"},
    {"registration_no": "IF707GGG", "make": "Kia", "model": "Sportage", "color": "Silver"},
]

SUBSCRIPTIONS = [
    {"car_registration_no": "CJ01AAA", "parking_id": 1, "begin_date": today - timedelta(days=30), "expiration_date": today + timedelta(days=60)},
    {"car_registration_no": "CJ02BBB", "parking_id": 2, "begin_date": today - timedelta(days=10), "expiration_date": today + timedelta(days=20)},
    {"car_registration_no": "B999XZA", "parking_id": 4, "begin_date": today - timedelta(days=90), "expiration_date": today - timedelta(days=5)},
    {"car_registration_no": "CJ05EEE", "parking_id": 1, "begin_date": today - timedelta(days=25), "expiration_date": today + timedelta(days=5)},
    {"car_registration_no": "CJ82RST", "parking_id": 3, "begin_date": today - timedelta(days=5), "expiration_date": today + timedelta(days=25)},
    {"car_registration_no": "CJ07GGG", "parking_id": 1, "begin_date": today - timedelta(days=15), "expiration_date": today + timedelta(days=45)},
    {"car_registration_no": "CJ07GGG", "parking_id": 2, "begin_date": today - timedelta(days=15), "expiration_date": today + timedelta(days=45)},
    {"car_registration_no": "CJ07GGG", "parking_id": 3, "begin_date": today - timedelta(days=15), "expiration_date": today + timedelta(days=45)},
    {"car_registration_no": "B101AAA", "parking_id": 4, "begin_date": today - timedelta(days=7), "expiration_date": today + timedelta(days=30)},
    {"car_registration_no": "B202BBB", "parking_id": 5, "begin_date": today - timedelta(days=2), "expiration_date": today + timedelta(days=90)},
    {"car_registration_no": "B303CCC", "parking_id": 1, "begin_date": today - timedelta(days=60), "expiration_date": today - timedelta(days=1)},
    {"car_registration_no": "IS505EEE", "parking_id": 3, "begin_date": today - timedelta(days=14), "expiration_date": today + timedelta(days=14)},
    {"car_registration_no": "PH606FFF", "parking_id": 2, "begin_date": today - timedelta(days=20), "expiration_date": today + timedelta(days=20)},
    {"car_registration_no": "IF707GGG", "parking_id": 5, "begin_date": today - timedelta(days=1), "expiration_date": today + timedelta(days=5)},
]

USERS = [
    {"fullname": "admin", "password": "admin123"},
    {"fullname": "officer1", "password": "officer123"},
    {"fullname": "officer2", "password": "officer123"},
    {"fullname": "demochief", "password": "demo123"},
]


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Unsupported date value: {value!r}")


def _normalize_car(car: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(car)
    normalized["registration_no"] = normalize_registration_no(normalized["registration_no"])
    return normalized


def _normalize_subscription(subscription: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(subscription)
    normalized["car_registration_no"] = normalize_registration_no(normalized["car_registration_no"])
    normalized["begin_date"] = _parse_date(normalized["begin_date"])
    normalized["expiration_date"] = _parse_date(normalized["expiration_date"])
    return normalized


def _load_overrides() -> dict[str, list[dict[str, Any]]]:
    if not OVERRIDES_PATH.exists():
        return {"parkings": [], "cars": [], "subscriptions": [], "users": []}

    raw = json.loads(OVERRIDES_PATH.read_text())
    return {
        "parkings": raw.get("parkings", []),
        "cars": [_normalize_car(car) for car in raw.get("cars", [])],
        "subscriptions": [_normalize_subscription(sub) for sub in raw.get("subscriptions", [])],
        "users": raw.get("users", []),
    }


def _merge_rows(
    base_rows: list[dict[str, Any]],
    override_rows: list[dict[str, Any]],
    key_fn,
) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {key_fn(row): row for row in base_rows}
    for row in override_rows:
        merged[key_fn(row)] = row
    return list(merged.values())


async def seed():
    overrides = _load_overrides()
    parkings = _merge_rows(PARKINGS, overrides["parkings"], lambda row: row["id"])
    cars = _merge_rows([_normalize_car(car) for car in CARS], overrides["cars"], lambda row: row["registration_no"])
    subscriptions = _merge_rows(
        [_normalize_subscription(sub) for sub in SUBSCRIPTIONS],
        overrides["subscriptions"],
        lambda row: (row["car_registration_no"], row["parking_id"], row["begin_date"]),
    )
    users = _merge_rows(USERS, overrides["users"], lambda row: row["fullname"])

    async with AsyncSessionLocal() as db:
        await db.execute(text("DELETE FROM flagged_cars"))
        await db.execute(text("DELETE FROM subscriptions"))
        await db.execute(text("DELETE FROM cars"))
        await db.execute(text("DELETE FROM parkings"))
        await db.execute(text("DELETE FROM users"))
        await db.commit()
        print("Cleared existing data.")

        for parking in parkings:
            db.add(Parking(**parking))
        await db.commit()
        print(f"  Inserted {len(parkings)} parking lots.")

        for car in cars:
            db.add(Car(**car))
        await db.commit()
        print(f"  Inserted {len(cars)} cars.")

        for subscription in subscriptions:
            db.add(Subscription(**subscription))
        await db.commit()
        print(f"  Inserted {len(subscriptions)} subscriptions.")

        for user in users:
            db.add(User(fullname=user["fullname"], password=pwd_context.hash(user["password"])))
        await db.commit()
        print(f"  Inserted {len(users)} users.")

        print("\nSeed complete. Demo cases ready:")
        print("  CJ01AAA   in lot 1 → no violation (valid sub)")
        print("  CJ02BBB   in lot 1 → car_in_wrong_parking")
        print("  CJ03CCC   in lot 1 → subscription_expired")
        print("  CJ04DDD   in lot 1 → no_subscription")
        print("  CJ05EEE   in lot 1 → subscription_close_to_expiration (scheduler)")
        print("  CJ06FFF   in lot 1 → requires_human_verification if confidence < 0.85")
        print("  CJ07GGG   in lots 1/2/3 → no violation")
        print("  TM404DDD  in any lot → no_subscription")
        print("  B303CCC   in lot 1 → subscription_expired")
        print("\nLogin credentials:")
        for user in users:
            print(f"  {user['fullname']} / {user['password']}")
        if overrides["cars"] or overrides["subscriptions"] or overrides["parkings"] or overrides["users"]:
            print(f"\nLoaded local overrides from {OVERRIDES_PATH.name}.")


if __name__ == "__main__":
    asyncio.run(seed())
