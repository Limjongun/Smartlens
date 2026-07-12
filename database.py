import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class VehicleLog(Base):
    __tablename__ = 'vehicle_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(50), nullable=False)
    color = Column(String(50), nullable=True)
    speed_kmh = Column(Integer, nullable=True)
    
def init_db(db_path="sqlite:///traffic.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print("Database initialized successfully.")
