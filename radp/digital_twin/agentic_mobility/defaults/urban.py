"""Urban scenario preset.

Dense city environment: significant pedestrian traffic, moderate vehicles,
balanced mobility classes, and Levy-walk parameters that favor moderate
randomness with relatively high step variance.
"""

URBAN_PRESET = {
    "name": "urban",
    "ue_density_per_km2": 1000,
    "default_distribution": {
        "stationary": 0.20,
        "pedestrian": 0.45,
        "cyclist": 0.10,
        "car": 0.25,
    },
    "alpha": 0.5,
    "variance": 0.8,
    "recommended_ticks": 300,
    "recommended_cells": 5,
}
