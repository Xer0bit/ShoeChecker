from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ShoeRequest(Base):
    __tablename__ = "shoe_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(String)
    image_path = Column(String)
    image_data = Column(LargeBinary)
    suggested_price = Column(Float, nullable=True)
    damage_type = Column(String)
    material_type = Column(String)
    confidence = Column(Float)
    repair_cost_min = Column(Float)
    repair_cost_max = Column(Float)
