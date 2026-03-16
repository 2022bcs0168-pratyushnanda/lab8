from pydantic import BaseModel
from typing import Optional

class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: Optional[float] = None
    population: float
    households: float
    median_income: float
    ocean_proximity: str

