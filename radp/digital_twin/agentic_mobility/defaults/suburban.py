"""Suburban scenario preset.

Lower-density residential/commercial mix: more cars and fewer pedestrians than
urban, with Levy-walk parameters that favor more directional movement.
"""

SUBURBAN_PRESET = {
    "name": "suburban",
    "ue_density_per_km2": 400,
    "default_distribution": {
        "stationary": 0.15,
        "pedestrian": 0.25,
        "cyclist": 0.10,
        "car": 0.50,
    },
    "alpha": 0.7,
    "variance": 0.6,
    "recommended_ticks": 300,
    "recommended_cells": 4,
}
