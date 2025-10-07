# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Coverage and Capacity Optimization (CCO) Environment Module.

This module defines the CCO environment based on the dGPCO algorithm,
providing methods for metric calculation and configuration management.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from apps.coverage_capacity_optimization import constants
from apps.coverage_capacity_optimization.cco_engine import CcoEngine, CcoMetric
from radp.digital_twin.utils.cell_selection import perform_attachment

load_dotenv()

from radp.client.client import RADPClient  # noqa: E402
from radp.client.helper import RADPHelper, SimulationStatus  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CCOEnvironment:
    """
    CCO Environment based on dGPCO algorithm.
    
    This class provides the core functionality for running CCO simulations
    and calculating metrics, similar to the original DgpcoCCO class.
    """
    
    def __init__(self, topology: pd.DataFrame, ue_data: pd.DataFrame, 
                 config: pd.DataFrame, bayesian_digital_twin_id: str,
                 valid_configuration_values: Dict[str, List[float]]):
        """
        Initialize CCO Environment.
        
        Args:
            topology: Cell topology DataFrame
            ue_data: UE data DataFrame
            config: Configuration DataFrame
            bayesian_digital_twin_id: Trained model ID
            valid_configuration_values: Valid parameter values
        """
        self.topology = topology
        self.num_cells = len(self.topology)
        self.valid_configuration_values = valid_configuration_values
        self.ue_data = ue_data
        self.config = config.copy()
        
        # Initialize RADP client
        self.radp_client = RADPClient()
        self.radp_helper = RADPHelper(self.radp_client)
        
        # Set up simulation event
        self.simulation_event: Dict[str, Any] = {
            "simulation_time_interval_seconds": 1,
            "ue_tracks": {"ue_data_id": "ue_data_1"},
            "rf_prediction": {"model_id": bayesian_digital_twin_id, "config_id": 1},
        }
        
        logger.info(f"Initialized CCO Environment with {self.num_cells} cells")
    
    def calc_metric(self, lambda_: float = 0.5, weak_coverage_threshold: float = -90,
                   over_coverage_threshold: float = 0) -> tuple:
        """
        Calculate the CCO metric given the current state of the config.
        
        Args:
            lambda_: Weight parameter for weak vs over coverage
            weak_coverage_threshold: Threshold for weak coverage
            over_coverage_threshold: Threshold for over coverage
            
        Returns:
            Tuple of (rf_dataframe, coverage_dataframe, cco_objective)
        """
        # Update the simulation event
        self.simulation_event["rf_prediction"]["config_id"] += 1
        
        # Run simulation
        simulation_id = self.radp_client.simulation(
            simulation_event=self.simulation_event,
            ue_data=self.ue_data,
            config=self.config,
        )["simulation_id"]
        
        simulation_status: SimulationStatus = self.radp_helper.resolve_simulation_status(
            simulation_id,
            wait_interval=1,
            max_attempts=100,
            verbose=False,
        )
        
        if not simulation_status.success:
            raise Exception(
                f"Exception occurred while running simulation '{simulation_id}': {simulation_status.error_message}"
            )
        
        # Consume simulation results
        rf_dataframe = self.radp_client.consume_simulation_output(simulation_id)
        
        # Run cell attachment
        cell_selected_rf_dataframe = perform_attachment(rf_dataframe, self.topology)
        
        # Get CCO coverage dataframe
        coverage_dataframe = CcoEngine.rf_to_coverage_dataframe(
            rf_dataframe=cell_selected_rf_dataframe,
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )
        
        # Get CCO objective
        cco_objective = CcoEngine.get_cco_objective_value(
            coverage_dataframe=coverage_dataframe,
            active_ids_list=coverage_dataframe[constants.CELL_ID].unique(),
            cco_metric=CcoMetric.PIXEL,
        )
        
        return cell_selected_rf_dataframe, coverage_dataframe, cco_objective
    
    def update_config(self, cell_id: str, parameter: str, value: float) -> None:
        """
        Update configuration for a specific cell.
        
        Args:
            cell_id: Cell identifier
            parameter: Parameter name to update
            value: New parameter value
        """
        cell_config_index = self.config.index[self.config["cell_id"] == cell_id][0]
        self.config.loc[cell_config_index, parameter] = value
    
    def get_config(self, cell_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific cell.
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            Cell configuration dictionary
        """
        cell_data = self.config[self.config["cell_id"] == cell_id]
        if cell_data.empty:
            raise ValueError(f"Cell ID '{cell_id}' not found in configuration")
        
        return cell_data.iloc[0].to_dict()
    
    def get_valid_tilt_values(self) -> List[float]:
        """
        Get valid tilt values for optimization.
        
        Returns:
            List of valid tilt values
        """
        return self.valid_configuration_values.get(constants.CELL_EL_DEG, [])
    
    def get_cell_neighbors(self, cell_id: str, max_distance_km: float = 5.0) -> List[str]:
        """
        Get neighboring cells within specified distance.
        
        Args:
            cell_id: Reference cell ID
            max_distance_km: Maximum distance in kilometers
            
        Returns:
            List of neighboring cell IDs
        """
        # Get reference cell coordinates
        ref_cell = self.topology[self.topology[constants.CELL_ID] == cell_id]
        if ref_cell.empty:
            raise ValueError(f"Cell ID '{cell_id}' not found")
        
        ref_lat = ref_cell[constants.CELL_LAT].iloc[0]
        ref_lon = ref_cell[constants.CELL_LON].iloc[0]
        
        # Calculate distances and find neighbors
        neighbors = []
        for _, cell in self.topology.iterrows():
            if cell[constants.CELL_ID] != cell_id:
                distance = self._calculate_distance_km(
                    ref_lat, ref_lon, cell[constants.CELL_LAT], cell[constants.CELL_LON]
                )
                if distance <= max_distance_km:
                    neighbors.append(cell[constants.CELL_ID])
        
        return neighbors
    
    def _calculate_distance_km(self, lat1: float, lon1: float, 
                              lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        radius_km = 6371
        return c * radius_km
    
    def reset_config(self) -> None:
        """Reset configuration to initial state."""
        # This would reset to the original configuration
        # Implementation depends on how initial config is stored
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'num_cells': self.num_cells,
            'num_ue_points': len(self.ue_data),
            'topology_bounds': {
                'lat_range': (self.topology[constants.CELL_LAT].min(), self.topology[constants.CELL_LAT].max()),
                'lon_range': (self.topology[constants.CELL_LON].min(), self.topology[constants.CELL_LON].max())
            }
        }
        
        return stats