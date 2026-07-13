"""Topology parameters model."""
from typing import List

from pydantic import BaseModel, Field


class TopologyParams(BaseModel):
    """LLM-generated topology parameters."""

    num_cells: int = Field(ge=1, description="Number of cell towers to generate")
    cell_carrier_freq_mhz: int = Field(
        ge=700, le=6000, description="Carrier frequency in MHz (e.g., 1800, 2100, 2600, 3500)"
    )
    azimuth_strategy: str = Field(
        description="Azimuth generation strategy: 'sectored' (3-sector sites with 0,120,240), 'omnidirectional' (single 0), or 'random'"
    )
    cell_placement_strategy: str = Field(
        description="Cell placement strategy: 'grid' (uniform grid), 'cluster' (clustered around hotspots), or 'boundary' (perimeter coverage)"
    )
    reasoning: str = Field(description="Brief explanation of why these parameters were chosen based on the scenario")
