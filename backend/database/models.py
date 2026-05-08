from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Boolean, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id            = Column(Integer, primary_key=True, index=True)
    title         = Column(String, nullable=False)
    category      = Column(String, default="admin")
    priority      = Column(Integer, default=2)      # 1=low 2=medium 3=high
    effort        = Column(Integer, default=2)       # Pomodoro slots needed
    deadline_days = Column(Integer, default=7)
    status        = Column(String, default="pending")  # pending / done
    created_at    = Column(DateTime, default=datetime.utcnow)


class TimeLog(Base):
    __tablename__ = "timelogs"

    id               = Column(Integer, primary_key=True, index=True)
    hostname         = Column(String)
    duration_seconds = Column(Integer)
    category         = Column(String)   # productive / distracting / neutral
    start_timestamp  = Column(Float)
    logged_at        = Column(DateTime, default=datetime.utcnow)


class EnergyLog(Base):
    __tablename__ = "energylogs"

    id           = Column(Integer, primary_key=True, index=True)
    energy_level = Column(Integer)   # 1 to 5
    logged_at    = Column(DateTime, default=datetime.utcnow)


# Creates tasks.db file automatically on first run
engine = create_engine(
    "sqlite:///tasks.db",
    connect_args={"check_same_thread": False}
)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Provides a database session for each API request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
