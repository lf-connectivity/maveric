"""Rural scenario preset.

Sparse environment dominated by vehicles: mostly straight-line vehicular
movement, very few pedestrians/cyclists, low step variance.
"""

RURAL_PRESET = {
    "name": "rural",
    "ue_density_per_km2": 50,
    "default_distribution": {
        "stationary": 0.10,
        "pedestrian": 0.05,
        "cyclist": 0.05,
        "car": 0.80,
    },
    "alpha": 0.9,
    "variance": 0.2,
    "recommended_ticks": 300,
    "recommended_cells": 3,
}
