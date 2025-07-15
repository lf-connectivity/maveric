# maveric/radp/digital_twin/utils/traffic_demand_simulation.py

import os
import sys
import json
import logging
import math
from typing import Dict, List, Tuple, Any, Union

import pandas as pd
import numpy as np

try:
    from scipy.spatial import Voronoi
except ImportError:
    print("Warning: scipy.spatial.Voronoi not found. Voronoi calculation will fail.")
    Voronoi = None

try:
    from shapely.geometry import Polygon, box, MultiPolygon, Point
    from shapely.validation import make_valid
    from shapely.errors import GEOSException
except ImportError:
    print("Error: Shapely library not found. Please install it: pip install Shapely")
    sys.exit(1)

# An attempt is made to import from the RADP library.
try:
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
except ImportError:
    print("Critical Error: Failed to import RADP constants or GISTools.")
    print("Ensure RADP_ROOT is in PYTHONPATH and radp modules are accessible.")
    class c:
        CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"
        LAT = "lat"; LON = "lon"
    class GISTools:
        @staticmethod
        def dist(coord1, coord2):
             R = 6371.0; lat1, lon1 = map(np.radians, coord1); lat2, lon2 = map(np.radians, coord2)
             dlon = lon2 - lon1; dlat = lat2 - lat1
             a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
             return R * 2 * np.arcsin(np.sqrt(a))
    print("Warning: Using fallback definitions for constants and GISTools.")


logger = logging.getLogger(__name__)

class TrafficDemandModel:
    """
    The generation of User Equipment (UE) traffic demand over a simulated geographical area is performed by this class.
    A fixed spatial layout is created based on the Voronoi regions of cell towers,
    and UEs are distributed across this layout according to time-varying parameters.
    """

    def __init__(self):
        """The TrafficDemandModel is initialized by this constructor."""
        pass

    def _space_boundary(self, cell_topology_data: pd.DataFrame, buffer_percent: float = 0.3) -> Dict[str, float]:
        """
        The spatial boundary for the given cell topology data is calculated by this method, with an optional buffer applied.
        """
        if cell_topology_data.empty:
            raise ValueError("Cell topology data cannot be empty for space boundary calculation.")
        min_lat = cell_topology_data[c.CELL_LAT].min()
        max_lat = cell_topology_data[c.CELL_LAT].max()
        min_lon = cell_topology_data[c.CELL_LON].min()
        max_lon = cell_topology_data[c.CELL_LON].max()
        lat_range_val = max_lat - min_lat
        lon_range_val = max_lon - min_lon
        lat_buffer = lat_range_val * buffer_percent if lat_range_val > 1e-9 else 0.1
        lon_buffer = lon_range_val * buffer_percent if lon_range_val > 1e-9 else 0.1
        return {
            "min_lon_buffered": max(min_lon - lon_buffer, -180.0),
            "min_lat_buffered": max(min_lat - lat_buffer, -90.0),
            "max_lon_buffered": min(max_lon + lon_buffer, 180.0),
            "max_lat_buffered": min(max_lat + lat_buffer, 90.0),
        }

    def _assign_space_types_to_polygons(self, polygons: List[Polygon], spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Space types are deterministically assigned to a list of polygons based on the provided proportions.
        """
        space_types = spatial_params.get("types", [])
        proportions = spatial_params.get("proportions", [])
        num_polygons = len(polygons)

        if not space_types or len(space_types) != len(proportions) or num_polygons == 0:
            logger.error("Invalid 'types'/'proportions' in spatial_params or no polygons to assign. Returning empty list.")
            return []
        
        counts = np.round(np.array(proportions) * num_polygons).astype(int)
        diff = num_polygons - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff

        type_assignments = []
        for i, stype in enumerate(space_types):
            type_assignments.extend([stype] * counts[i])
        
        np.random.shuffle(polygons)

        spatial_cells_with_types = []
        for i, poly in enumerate(polygons):
            assigned_type = type_assignments[i]
            spatial_cells_with_types.append({
                "bounds": list(poly.exterior.coords), "type": assigned_type, "cell_id": i,
                "area_sq_km": GISTools.area_of_polygon(poly.exterior.coords) if hasattr(GISTools, 'area_of_polygon') else poly.area
            })
        
        logger.info(f"Assigned types to {num_polygons} polygons with counts: {dict(zip(space_types, counts))}")
        return spatial_cells_with_types

    # MODIFIED: This method is now more robust for handling few cell sites.
    def generate_spatial_layout(self,
                                 topology_df: pd.DataFrame,
                                 spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        A fixed spatial layout is generated using Voronoi tessellation. 
        This method has been made robust against a small number of sites by the addition of temporary bounding points, ensuring that all real sites produce a finite polygon.
        """
        logger.info("Generating fixed spatial layout...")
        if Voronoi is None:
            logger.error("Scipy Voronoi is not available. Cannot generate spatial layout.")
            return []
        if not {c.CELL_LAT, c.CELL_LON}.issubset(topology_df.columns):
            raise ValueError(f"Topology data must include '{c.CELL_LAT}' and '{c.CELL_LON}'.")

        points = topology_df[[c.CELL_LON, c.CELL_LAT]].values
        if len(points) < 4:
            logger.error(f"Need at least 4 cell sites for 2D Voronoi, found {len(points)}.")
            return []

        # Temporary "ghost" points are added to bound the diagram,
        # ensuring that all regions for the original points are finite.
        min_lon, max_lon = points[:, 0].min(), points[:, 0].max()
        min_lat, max_lat = points[:, 1].min(), points[:, 1].max()
        lon_range = max_lon - min_lon if max_lon > min_lon else 1.0
        lat_range = max_lat - min_lat if max_lat > min_lat else 1.0

        ghost_points = np.array([
            [min_lon - 3 * lon_range, min_lat - 3 * lat_range],
            [min_lon - 3 * lon_range, max_lat + 3 * lat_range],
            [max_lon + 3 * lon_range, max_lat + 3 * lat_range],
            [max_lon + 3 * lon_range, min_lat - 3 * lat_range]
        ])
        combined_points = np.vstack([points, ghost_points])
        num_original_points = len(points)
        # End of ghost points addition

        try:
            vor = Voronoi(combined_points)
        except Exception as e:
            logger.error(f"Voronoi diagram generation failed: {e}")
            return []

        space_bound_dict = self._space_boundary(topology_df)
        clipping_box = box(space_bound_dict["min_lon_buffered"], space_bound_dict["min_lat_buffered"],
                           space_bound_dict["max_lon_buffered"], space_bound_dict["max_lat_buffered"])

        # The original points are looped through and their (now finite) regions are obtained.
        valid_polygons = []
        for i in range(num_original_points):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]

            if -1 in region_vertices_indices:
                logger.warning(f"Region for original point {i} is unexpectedly infinite despite bounding points. Skipping.")
                continue

            try:
                polygon = Polygon(vor.vertices[region_vertices_indices])
                clipped_poly = polygon.intersection(clipping_box)
                
                if not clipped_poly.is_empty and clipped_poly.area > 1e-9:
                    valid_polygons.append(clipped_poly)
            except Exception as e:
                 logger.warning(f"Could not process polygon for point {i}: {e}")
        
        if not valid_polygons:
            logger.error("No valid spatial cells created after Voronoi processing.")
            return []

        spatial_cells = self._assign_space_types_to_polygons(valid_polygons, spatial_params)
        logger.info(f"Generated {len(spatial_cells)} spatial cells for the fixed layout.")
        return spatial_cells


    def distribute_ues_over_time(self,
                                  spatial_layout: List[Dict[str, Any]],
                                  time_params: Dict[str, Any],
                                  num_ues_per_tick: int) -> Dict[int, pd.DataFrame]:
        """
        UEs are distributed across a pre-generated spatial layout for each time tick by this method.
        """
        if not spatial_layout:
            logger.error("Spatial layout is empty. Cannot distribute UEs.")
            return {}

        total_ticks = time_params.get("total_ticks", 1)
        time_weights_all_types = time_params.get("time_weights", {})
        cells_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for cell_info in spatial_layout:
            cells_by_type.setdefault(cell_info['type'], []).append(cell_info)

        all_ue_data_per_tick: Dict[int, pd.DataFrame] = {}

        for tick in range(total_ticks):
            tick_ue_data_list = []
            current_ue_id_in_tick = 0
            
            tick_type_weights = {stype: time_weights_all_types.get(stype, [0]*total_ticks)[tick] for stype in cells_by_type.keys()}
            total_weight_this_tick = sum(tick_type_weights.values())

            if total_weight_this_tick <= 1e-9:
                all_ue_data_per_tick[tick] = pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
                continue
            
            # A temporary dictionary is used to handle rounding and to ensure that the total number of UEs matches the target.
            ue_counts = {}
            for space_type, weight in tick_type_weights.items():
                ue_counts[space_type] = (weight / total_weight_this_tick) * num_ues_per_tick
            
            # The counts are adjusted to integers, ensuring that the sum matches num_ues_per_tick.
            total_ues = 0
            int_ue_counts = {stype: int(count) for stype, count in ue_counts.items()}
            total_ues = sum(int_ue_counts.values())
            
            remainder = num_ues_per_tick - total_ues
            if remainder > 0:
                # The remainder is distributed based on fractional parts.
                sorted_types = sorted(ue_counts, key=lambda k: ue_counts[k] - int_ue_counts[k], reverse=True)
                for i in range(remainder):
                    int_ue_counts[sorted_types[i % len(sorted_types)]] += 1

            for space_type, num_ues_for_type in int_ue_counts.items():
                if num_ues_for_type == 0:
                    continue
                
                available_cells_for_this_type = cells_by_type.get(space_type, [])
                if not available_cells_for_this_type:
                    continue

                for _ in range(num_ues_for_type):
                    chosen_cell_info = np.random.choice(available_cells_for_this_type)
                    polygon_boundary = Polygon(chosen_cell_info['bounds'])
                    min_x, min_y, max_x, max_y = polygon_boundary.bounds
                    
                    attempts = 0
                    while attempts < 100:
                        ue_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
                        if polygon_boundary.contains(ue_point):
                            tick_ue_data_list.append({
                                'ue_id': current_ue_id_in_tick, c.LON: ue_point.x, c.LAT: ue_point.y,
                                'tick': tick, 'space_type': space_type
                            })
                            current_ue_id_in_tick += 1
                            break
                        attempts += 1
            
            all_ue_data_per_tick[tick] = pd.DataFrame(tick_ue_data_list) if tick_ue_data_list else pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
        
        logger.info(f"Finished UE distribution for {total_ticks} ticks.")
        return all_ue_data_per_tick

    def generate_traffic_demand(self,
                                topology_df: pd.DataFrame,
                                spatial_params: Dict[str, Any],
                                time_params: Dict[str, Any],
                                num_ues_per_tick: int) -> Tuple[Dict[int, pd.DataFrame], List[Dict[str, Any]]]:
        """
        Traffic demand is generated by creating a spatial layout and distributing UEs over time.
        """
        spatial_layout = self.generate_spatial_layout(topology_df, spatial_params)
        if not spatial_layout:
            logger.error("Failed to generate spatial layout. Aborting traffic demand generation.")
            return {}, []

        ue_data_per_tick = self.distribute_ues_over_time(spatial_layout, time_params, num_ues_per_tick)

        return ue_data_per_tick, spatial_layout
