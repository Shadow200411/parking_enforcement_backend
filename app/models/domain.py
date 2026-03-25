import enum
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import String, Integer, ForeignKey, Boolean, Float ,Text, Enum as SQLEnum, Index, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    """All of our models will inherit from this Base class."""
    pass

class FlagType(str, enum.Enum):
    subscription_close_to_expiration = "subscription_close_to_expiration"
    subscription_expired = "subscription_expired"
    car_in_wrong_parking = "car_in_wrong_parking"
    no_subscription = "no_subscription"

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fullname: Mapped[str] = mapped_column(String, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
class Parking(Base):
    __tablename__ = "parkings"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    subscriptions: Mapped[List["Subscription"]] = relationship(back_populates="parking")
    flagged_cars: Mapped[List["FlaggedCar"]] = relationship(back_populates="parking")
    
class Car(Base):
    __tablename__ = "cars"
    
    registration_no: Mapped[str] = mapped_column(String, primary_key=True)
    make: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    color: Mapped[str] = mapped_column(String, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    subscriptions: Mapped[List["Subscription"]] = relationship(back_populates="car")
    flags: Mapped[List["FlaggedCar"]] = relationship(back_populates="car")
    
class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        UniqueConstraint('car_registration_no', 'parking_id', 'begin_date', name='uix_car_parking_date'),
        Index('idx_sub_car_reg', 'car_registration_no'),
        Index('idx_sub_parking', 'parking_id')
    )
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    car_registration_no: Mapped[str] = mapped_column(ForeignKey("cars.registration_no"), nullable=False)
    parking_id: Mapped[str] = mapped_column(ForeignKey("parkings.id"), nullable=False)
    begin_date: Mapped[date] = mapped_column(nullable=False)
    expiration_date: Mapped[date] = mapped_column(nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    car: Mapped["Car"] = relationship(back_populates="subscriptions")
    parking: Mapped["Parking"] = relationship(back_populates="subscriptions")
    
class FlaggedCar(Base):
    __tablename__ = "flagged_cars"
    __table_args__ = (
        Index('idx_flag_car_reg', 'car_registration_no'),
        Index('idx_flag_parking', 'parking_id'),
        Index('idx_flag_type', 'type'),
        Index('idx_flag_req_human', 'requires_human_verification'),
        Index('idx_flag_verified', 'verified_by_human')
    ) 
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[FlagType] = mapped_column(SQLEnum(FlagType), nullable=False)
    car_registration_no: Mapped[str] = mapped_column(ForeignKey("cars.registration_no"), nullable=False)
    parking_id: Mapped[int] = mapped_column(ForeignKey("parkings.id"), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(nullable=False)

    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    evidence_image_url: Mapped[Optional[str]] = mapped_column(String)
    
    requires_human_verification: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verified_by_human: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verification_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    car: Mapped["Car"] = relationship(back_populates="flags")
    parking: Mapped["Parking"] = relationship(back_populates="flagged_cars")
    


    