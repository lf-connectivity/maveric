"""Generated mobility parameters model."""
from typing import Dict

from pydantic import BaseModel, Field


class GenParams(BaseModel):
    """Generated mobility parameters."""

    alpha: float = Field(ge=0.0, le=1.0, description="Levy walk alpha exponent")
    variance: float = Field(ge=0.0, description="Step size variance")
    ue_class_distribution: Dict[str, float] = Field(description="UE class distribution percentages")
    velocity_adjustments: Dict[str, Dict[str, float]] = Field(description="Velocity and variance per UE class")
