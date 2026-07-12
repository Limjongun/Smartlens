import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class InferenceSession(Base):
    __tablename__ = 'inference_sessions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(50), unique=True, nullable=False) # e.g. inf_001
    start_time = Column(DateTime, default=datetime.utcnow)
    source = Column(String(200))
    
    logs = relationship("VehicleLog", back_populates="session", cascade="all, delete-orphan")
    violations = relationship("ViolationLog", back_populates="session", cascade="all, delete-orphan")

class VehicleLog(Base):
    __tablename__ = 'vehicle_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(50), ForeignKey('inference_sessions.session_name'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    track_id = Column(String(50), nullable=False)
    vehicle_class = Column(String(50), nullable=False)
    color = Column(String(50), nullable=True)
    speed_kmh = Column(Integer, nullable=True)
    
    session = relationship("InferenceSession", back_populates="logs")

class ViolationLog(Base):
    __tablename__ = 'violations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(50), ForeignKey('inference_sessions.session_name'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    track_id = Column(String(50), nullable=False)
    vehicle_class = Column(String(50), nullable=False)
    region_name = Column(String(100), nullable=False)
    image_path = Column(String(255), nullable=False)
    
    session = relationship("InferenceSession", back_populates="violations")

def init_db(db_path="sqlite:///traffic.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print("Database initialized successfully.")
