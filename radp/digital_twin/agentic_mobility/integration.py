"""Integration module for end-to-end natural language to mobility simulation."""
from typing import Dict, Tuple

import pandas as pd

from radp.digital_twin.agentic_mobility.api import generate_mobility_params
from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
from radp.digital_twin.mobility.ue_tracks_params import UETracksGenerationParams


class AgenticMobilityIntegration:
    """Bridge between agentic NLP generation and existing RADP mobility simulation."""

    @staticmethod
    def generate_from_natural_language(query: str) -> Tuple[pd.DataFrame, Dict]:
        """Complete pipeline: Natural language → Parameters → Simulation → DataFrame.

        This method orchestrates the full workflow:
        1. Parse natural language query into structured parameters
        2. Generate mobility parameters using LLM-based agentic approach
        3. Validate and format parameters for RADP compatibility
        4. Execute mobility simulation using UETracksGenerator
        5. Return mobility tracks as pandas DataFrame

        Args:
            query: Natural language query (e.g., "Generate 100 UEs in urban Tokyo")

        Returns:
            Tuple containing:
            - pd.DataFrame: UE mobility tracks with columns [ue_id, longitude, latitude, tick]
            - Dict: Metadata including:
                - query_intent: Parsed query information
                - retry_count: Number of generation retries
                - validation_warnings: Any validation issues
                - spatial_bounds: Requested and actual lat/lon boundaries

        Example:
            >>> df, metadata = AgenticMobilityIntegration.generate_from_natural_language(
            ...     "Generate 100 UEs in urban Tokyo during morning rush hour"
            ... )
            >>> print(f"Generated {len(df)} position points")
            >>> print(df.head())
        """
        # Step 1: Convert natural language to RADP-compatible parameters
        result = generate_mobility_params(query)

        if result["status"] not in ["success", "success_with_warnings"]:
            raise ValueError(f"Parameter generation failed: {result}")

        radp_params = result["radp_params"]
        metadata = result["metadata"]

        # Step 2: Create UETracksGenerationParams object (validates schema)
        ue_tracks_params = UETracksGenerationParams(radp_params)

        # Step 3: Generate mobility tracks using RADP UETracksGenerator
        ue_tracks_df = pd.DataFrame()  # Initialize empty DataFrame

        for ue_tracks_generation_batch in UETracksGenerator.generate_as_lon_lat_points(
            rng_seed=ue_tracks_params.rng_seed,
            lon_x_dims=ue_tracks_params.lon_x_dims,
            lon_y_dims=ue_tracks_params.lon_y_dims,
            num_ticks=ue_tracks_params.num_ticks,
            num_UEs=ue_tracks_params.num_UEs,
            num_batches=ue_tracks_params.num_batches,
            alpha=ue_tracks_params.alpha,
            variance=ue_tracks_params.variance,
            min_lat=ue_tracks_params.min_lat,
            max_lat=ue_tracks_params.max_lat,
            min_lon=ue_tracks_params.min_lon,
            max_lon=ue_tracks_params.max_lon,
            mobility_class_distribution=ue_tracks_params.mobility_class_distribution,
            mobility_class_velocities=ue_tracks_params.mobility_class_velocities,
            mobility_class_velocity_variances=ue_tracks_params.mobility_class_velocity_variances,
        ):
            # Append each batch to the main DataFrame
            ue_tracks_df = pd.concat([ue_tracks_df, ue_tracks_generation_batch], ignore_index=True)

        # Step 4: Calculate spatial bounds and add to metadata
        spatial_bounds = {
            "requested": {
                "min_lat": float(ue_tracks_params.min_lat),
                "max_lat": float(ue_tracks_params.max_lat),
                "min_lon": float(ue_tracks_params.min_lon),
                "max_lon": float(ue_tracks_params.max_lon),
            },
            "actual": None,
        }

        # Calculate actual bounds from generated data
        if not ue_tracks_df.empty:
            spatial_bounds["actual"] = {
                "min_lat": float(ue_tracks_df["lat"].min()),
                "max_lat": float(ue_tracks_df["lat"].max()),
                "min_lon": float(ue_tracks_df["lon"].min()),
                "max_lon": float(ue_tracks_df["lon"].max()),
            }

        metadata["spatial_bounds"] = spatial_bounds

        return ue_tracks_df, metadata

    @staticmethod
    def generate_params_only(query: str) -> Tuple[Dict, Dict]:
        """Generate only the parameters without running simulation.

        Useful for testing, debugging, or when you want to inspect/modify
        parameters before running the actual simulation.

        Args:
            query: Natural language query

        Returns:
            Tuple containing:
            - Dict: RADP-compatible parameters
            - Dict: Metadata (query_intent, retry_count, etc.)

        Example:
            >>> params, metadata = AgenticMobilityIntegration.generate_params_only(
            ...     "Generate 50 UEs in suburban Chicago"
            ... )
            >>> print(params['ue_tracks_generation']['params']['num_ticks'])
        """
        result = generate_mobility_params(query)

        if result["status"] not in ["success", "success_with_warnings"]:
            raise ValueError(f"Parameter generation failed: {result}")

        return result["radp_params"], result["metadata"]
