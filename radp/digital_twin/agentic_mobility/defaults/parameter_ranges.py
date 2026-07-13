"""Valid parameter ranges for validation."""

# Alpha parameter (Levy walk exponent)
ALPHA_RANGE = {"min": 0.0, "max": 1.0, "description": "Levy walk alpha exponent (0=Brownian, 1=ballistic)"}

# Variance parameter
VARIANCE_RANGE = {"min": 0.0, "max": float("inf"), "description": "Step size variance (must be non-negative)"}

# Velocity ranges by mobility class (m/s)
VELOCITY_RANGES = {
    "stationary": {"min": 0.0, "max": 0.0},
    "pedestrian": {"min": 0.5, "max": 2.0},
    "cyclist": {"min": 3.0, "max": 8.0},
    "car": {"min": 5.0, "max": 30.0},
}

# Area constraints
AREA_CONSTRAINTS = {
    "min_side_length_m": 100,  # Minimum 100m x 100m
    "max_side_length_m": 10000,  # Maximum 10km x 10km
    "max_area_km2": 100,  # Maximum 100 km²
}

# Simulation constraints
SIMULATION_CONSTRAINTS = {"max_ues": 10000, "min_ues": 30, "max_ticks": 1000, "min_ticks": 50}
