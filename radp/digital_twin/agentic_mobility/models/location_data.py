"""Location data model."""
from typing import Tuple

from pydantic import BaseModel


class LocationData(BaseModel):
    """Geographic location information."""

    center: Tuple[float, float]  # (latitude, longitude)
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    area_type: str  # "urban", "suburban", "rural", etc.
