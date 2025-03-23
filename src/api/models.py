from pydantic import BaseModel
from typing import Optional

class ShoeAnalysisRequest(BaseModel):
    description: str
    suggested_price: Optional[float] = None

class ShoeAnalysisResponse(BaseModel):
    damage_type: str
    material_type: str
    shoe_type: str
    confidence: float
    description: str
    repair_cost_min: float
    repair_cost_max: float
    analysis_details: str
