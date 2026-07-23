from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime
)

from sqlalchemy.orm import declarative_base

from datetime import datetime


Base = declarative_base()


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    disease = Column(
        String,
        nullable=False
    )

    confidence = Column(
        String,
        nullable=False
    )

    report = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )