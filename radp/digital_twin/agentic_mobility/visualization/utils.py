import numpy as np


def calculate_total_distance(ue_data):
    """Calculate total distance traveled by a UE."""
    if len(ue_data) < 2:
        return 0

    ue_data = ue_data.sort_values("tick")
    lats = ue_data["lat"].values
    lons = ue_data["lon"].values

    # Euclidean distance (simplified)
    distances = np.sqrt(np.diff(lats) ** 2 + np.diff(lons) ** 2)
    return distances.sum()
