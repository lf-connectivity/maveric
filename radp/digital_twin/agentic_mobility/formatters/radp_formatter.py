"""Convert generated parameters to RADP UETracksGenerationParams format."""
from typing import Dict


class RADPFormatter:
    """Format generated parameters for RADP integration."""

    @staticmethod
    def format_to_radp(gen_params: Dict, location_data: Dict, query_intent: Dict) -> Dict:
        """Convert generated parameters to RADP JSON schema.

        Args:
            gen_params: Generated mobility parameters
            location_data: Geographic bounds data
            query_intent: Original query intent

        Returns:
            Dictionary in RADP UETracksGenerationParams format
        """
        # Extract parameters
        alpha = gen_params["alpha"]
        variance = gen_params["variance"]
        ue_class_dist = gen_params["ue_class_distribution"]
        velocity_adj = gen_params["velocity_adjustments"]
        num_ues = query_intent["num_ues"]
        num_ticks = query_intent["num_ticks"]

        # Calculate counts per class based on distribution
        # Use rounding to nearest integer, then adjust for any discrepancy
        ue_class_counts = {}
        fractional_parts = {}

        for ue_class, percentage in ue_class_dist.items():
            exact_count = num_ues * percentage
            ue_class_counts[ue_class] = int(exact_count)
            fractional_parts[ue_class] = exact_count - int(exact_count)

        # Ensure total equals num_ues (handle rounding)
        total_allocated = sum(ue_class_counts.values())
        remaining = num_ues - total_allocated

        if remaining > 0:
            # Distribute remaining UEs to classes with highest fractional parts
            sorted_classes = sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)
            for i in range(remaining):
                ue_class_counts[sorted_classes[i][0]] += 1
        elif remaining < 0:
            # Remove excess UEs from classes with lowest fractional parts
            sorted_classes = sorted(fractional_parts.items(), key=lambda x: x[1])
            for i in range(abs(remaining)):
                if ue_class_counts[sorted_classes[i][0]] > 0:
                    ue_class_counts[sorted_classes[i][0]] -= 1

        # Build UE class distribution in RADP format
        radp_ue_class_dist = {}
        for ue_class in ["stationary", "pedestrian", "cyclist", "car"]:
            vel_adj = velocity_adj.get(ue_class, {"velocity": 0, "velocity_variance": 0})
            radp_ue_class_dist[ue_class] = {
                "count": ue_class_counts.get(ue_class, 0),
                "velocity": vel_adj.get("velocity", 0),
                "velocity_variance": vel_adj.get("velocity_variance", 0),
            }

        # Build lat/lon boundaries
        lat_lon_boundaries = {
            "min_lat": location_data["min_lat"],
            "max_lat": location_data["max_lat"],
            "min_lon": location_data["min_lon"],
            "max_lon": location_data["max_lon"],
        }

        # Build Gauss-Markov parameters
        gauss_markov_params = {
            "alpha": alpha,
            "variance": variance,
            "rng_seed": 42,  # Default random seed for reproducibility
            "lon_x_dims": 100,  # Default grid x-dimension (meters)
            "lon_y_dims": 100,  # Default grid y-dimension (meters)
        }

        # Construct full RADP format
        radp_params = {
            "ue_tracks_generation": {
                "params": {
                    "simulation_duration": 3600,  # Default 1 hour
                    "simulation_time_interval_seconds": 0.01,  # Default 10ms (note: key has _seconds suffix)
                    "num_ticks": num_ticks,
                    "num_batches": 1,  # Default
                    "ue_class_distribution": radp_ue_class_dist,
                    "lat_lon_boundaries": lat_lon_boundaries,
                    "gauss_markov_params": gauss_markov_params,
                }
            }
        }

        return radp_params
