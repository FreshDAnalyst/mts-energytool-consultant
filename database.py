
import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# =========================================
# DATABASE CONFIGURATION
# =========================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./mts_energytool.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()

def get_db():
    """Provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def utc_now():
    return datetime.now(timezone.utc)

# =========================================
# CONVERSATION SESSIONS
# =========================================

class ChatSession(Base):

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    status = Column(String(30), default="active")
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    leads = relationship(
        "CustomerLead",
        back_populates="session",
    )

# =========================================
# CHAT MESSAGES
# =========================================

class ChatMessage(Base):

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        Integer,
        ForeignKey("sessions.id"),
        nullable=False,
    )

    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship(
        "ChatSession",
        back_populates="messages",
    )

# =========================================
# CUSTOMER LEADS
# =========================================

class CustomerLead(Base):

    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        Integer,
        ForeignKey("sessions.id"),
        nullable=False,
    )

    full_name = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=False)
    company = Column(String(255), nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship(
        "ChatSession",
        back_populates="leads",
    )

# =========================================
# CREATE DATABASE TABLES
# =========================================

def init_db():
    """Create all database tables."""

    Base.metadata.create_all(bind=engine)

    print("MTS Energytool database initialized!")
    print("Tables: sessions, messages, leads")