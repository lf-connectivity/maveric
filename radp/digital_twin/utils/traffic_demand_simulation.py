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

# Attempt to import from RADP library.
# These imports assume that this file is part of the radp package and
# other radp modules are accessible in the Python path.
try:
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
except ImportError:
    print("Critical Error: Failed to import RADP constants or GISTools.")
    print("Ensure RADP_ROOT is in PYTHONPATH and radp modules are accessible.")
    # Fallback for basic script operation if run standalone for some reason,
    # but this is not ideal for a library component.
    class c:
        CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"
        LAT = "lat"; LON = "lon"
    class GISTools:
        @staticmethod
        def dist(coord1, coord2): # Simplified fallback
             R = 6371.0; lat1, lon1 = map(np.radians, coord1); lat2, lon2 = map(np.radians, coord2)
             dlon = lon2 - lon1; dlat = lat2 - lat1
             a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
             return R * 2 * np.arcsin(np.sqrt(a))
    print("Warning: Using fallback definitions for constants and GISTools. This is not recommended for production.")


logger = logging.getLogger(__name__)

class TrafficDemandModel:
    """
    Generates UE (User Equipment) traffic demand over a simulated geographical area.
    It creates a fixed spatial layout based on Voronoi regions of cell towers
    and distributes UEs across this layout according to time-varying parameters.
    """

    def __init__(self):
        """Initializes the TrafficDemandModel."""
        pass

    def _space_boundary(self, cell_topology_data: pd.DataFrame, buffer_percent: float = 0.3) -> Dict[str, float]:
        """
        Calculates the buffered bounding box around cell tower locations.

        Args:
            cell_topology_data: DataFrame with cell latitude and longitude.
                                Must contain columns c.CELL_LAT and c.CELL_LON.
            buffer_percent: Percentage to extend the boundary.

        Returns:
            Dictionary with min/max buffered latitude and longitude.
        """
        if cell_topology_data.empty:
            raise ValueError("Cell topology data cannot be empty for space boundary calculation.")
        if not {c.CELL_LAT, c.CELL_LON}.issubset(cell_topology_data.columns):
            raise ValueError(f"Topology data must include '{c.CELL_LAT}' and '{c.CELL_LON}'.")

        min_lat = cell_topology_data[c.CELL_LAT].min()
        max_lat = cell_topology_data[c.CELL_LAT].max()
        min_lon = cell_topology_data[c.CELL_LON].min()
        max_lon = cell_topology_data[c.CELL_LON].max()

        lat_range_val = max_lat - min_lat
        lon_range_val = max_lon - min_lon

        lat_buffer = lat_range_val * buffer_percent if lat_range_val > 1e-9 else 0.1
        lon_buffer = lon_range_val * buffer_percent if lon_range_val > 1e-9 else 0.1
        
        logger.debug(f"Original Bounds: LON=[{min_lon:.4f}, {max_lon:.4f}], LAT=[{min_lat:.4f}, {max_lat:.4f}]")
        logger.debug(f"Buffer Percent: {buffer_percent*100}%, Lon Buffer: {lon_buffer:.4f}, Lat Buffer: {lat_buffer:.4f}")

        return {
            "min_lon_buffered": max(min_lon - lon_buffer, -180.0),
            "min_lat_buffered": max(min_lat - lat_buffer, -90.0),
            "max_lon_buffered": min(max_lon + lon_buffer, 180.0),
            "max_lat_buffered": min(max_lat + lat_buffer, 90.0),
        }

    def _assign_space_types_to_polygons(self, polygons: List[Polygon], space_type_proportions: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Assigns space types to a list of polygons based on given proportions.

        Args:
            polygons: A list of Shapely Polygon objects.
            space_type_proportions: Dict mapping space type names to their proportions.

        Returns:
            A list of dictionaries, each representing a spatial cell with its
            bounds (polygon exterior coordinates) and assigned 'type'.
        """
        spatial_cells_with_types = []
        space_keys = list(space_type_proportions.keys())
        space_probs = list(space_type_proportions.values())

        if not np.isclose(sum(space_probs), 1.0):
            logger.warning("Proportions for space types do not sum to 1.0. Normalizing.")
            space_probs = np.array(space_probs) / sum(space_probs)

        for i, poly in enumerate(polygons):
            assigned_type = np.random.choice(space_keys, p=space_probs)
            spatial_cells_with_types.append({
                "bounds": list(poly.exterior.coords),
                "type": assigned_type,
                "cell_id": i, # Simple ID for this spatial cell
                "area_sq_km": GISTools.area_of_polygon(poly.exterior.coords) if hasattr(GISTools, 'area_of_polygon') else poly.area # poly.area is in squared units of coords
            })
        return spatial_cells_with_types


    def _generate_spatial_layout(self,
                                 topology_df: pd.DataFrame,
                                 spatial_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a fixed spatial layout using Voronoi tessellation based on cell towers,
        clips them to a boundary, and assigns area types.

        Args:
            topology_df: DataFrame with cell tower locations (c.CELL_LAT, c.CELL_LON).
            spatial_params: Dictionary with "types" (list of area type names)
                            and "proportions" (list of corresponding proportions).

        Returns:
            A list of dictionaries, each representing a spatial cell with its
            'bounds' (polygon exterior coordinates) and assigned 'type'.
            Returns an empty list if layout generation fails.
        """
        logger.info("Generating fixed spatial layout...")
        if Voronoi is None:
            logger.error("Scipy Voronoi is not available. Cannot generate spatial layout from Voronoi.")
            return []
        if not {c.CELL_LAT, c.CELL_LON}.issubset(topology_df.columns):
            raise ValueError(f"Topology data must include '{c.CELL_LAT}' and '{c.CELL_LON}'.")

        points = topology_df[[c.CELL_LON, c.CELL_LAT]].values
        if len(points) < 4: # Voronoi requires at least N+1 points in N-D
            logger.error(f"Need at least 4 cell sites for 2D Voronoi, found {len(points)}. Cannot generate spatial layout.")
            return []

        try:
            vor = Voronoi(points)
        except Exception as e:
            logger.error(f"Voronoi diagram generation failed: {e}")
            return []

        # Calculate bounding box for clipping
        try:
            space_bound_dict = self._space_boundary(topology_df)
            min_lon_b = space_bound_dict["min_lon_buffered"]
            min_lat_b = space_bound_dict["min_lat_buffered"]
            max_lon_b = space_bound_dict["max_lon_buffered"]
            max_lat_b = space_bound_dict["max_lat_buffered"]
            clipping_box = box(min_lon_b, min_lat_b, max_lon_b, max_lat_b)
        except Exception as e:
            logger.error(f"Failed to create clipping boundary: {e}")
            return []

        valid_clipped_polygons = []
        processed_regions = 0
        skipped_infinite = 0
        skipped_invalid_geom = 0
        
        for region_idx_list in vor.regions:
            if not region_idx_list or -1 in region_idx_list:
                skipped_infinite += 1
                continue # Skip infinite regions

            try:
                polygon_vertices = vor.vertices[region_idx_list]
                if np.isnan(polygon_vertices).any() or len(polygon_vertices) < 3:
                    skipped_invalid_geom +=1
                    continue

                region_polygon = Polygon(polygon_vertices)
                
                # Validate and attempt to fix invalid polygons
                if not region_polygon.is_valid:
                    region_polygon = make_valid(region_polygon) # Requires Shapely 1.8+
                    if not region_polygon.is_valid: # Still invalid
                        skipped_invalid_geom +=1
                        continue
                
                # Handle MultiPolygons that might result from make_valid or intersection
                if region_polygon.geom_type == 'MultiPolygon':
                    for poly_geom in region_polygon.geoms:
                        if poly_geom.geom_type == 'Polygon':
                            clipped_poly = poly_geom.intersection(clipping_box)
                            if not clipped_poly.is_empty and clipped_poly.is_valid and clipped_poly.geom_type == 'Polygon' and clipped_poly.area > 1e-9: # Area threshold
                                valid_clipped_polygons.append(clipped_poly)
                elif region_polygon.geom_type == 'Polygon':
                    clipped_poly = region_polygon.intersection(clipping_box)
                    if not clipped_poly.is_empty and clipped_poly.is_valid and clipped_poly.geom_type == 'Polygon' and clipped_poly.area > 1e-9:
                        valid_clipped_polygons.append(clipped_poly)
                else: # GeometryCollection, etc.
                    skipped_invalid_geom +=1
                    continue
                processed_regions +=1
            except GEOSException as e: # Catch specific Shapely errors
                logger.warning(f"Shapely GEOSException processing Voronoi region: {e}")
                skipped_invalid_geom +=1
            except Exception as e:
                logger.warning(f"Error processing Voronoi region: {e}")
                skipped_invalid_geom +=1

        logger.info(f"Voronoi processing: Regions processed={processed_regions}, Infinite skipped={skipped_infinite}, Invalid/Empty skipped={skipped_invalid_geom}")
        
        if not valid_clipped_polygons:
            logger.error("No valid spatial cells created after Voronoi processing and clipping.")
            return []

        # Assign space types to the valid clipped polygons
        space_types = spatial_params.get("types", [])
        proportions = spatial_params.get("proportions", [])

        if not space_types or not proportions or len(space_types) != len(proportions):
            logger.error("Invalid 'types' or 'proportions' in spatial_params. Using single default type.")
            # Fallback to a single type if params are incorrect
            default_type_name = "default_area"
            space_type_proportions = {default_type_name: 1.0}
        else:
            space_type_proportions = {stype: prop for stype, prop in zip(space_types, proportions)}
        
        spatial_cells = self._assign_space_types_to_polygons(valid_clipped_polygons, space_type_proportions)
        
        logger.info(f"Generated {len(spatial_cells)} spatial cells for the fixed layout.")
        return spatial_cells

    def _distribute_ues_in_layout(self,
                                  spatial_layout: List[Dict[str, Any]],
                                  time_params: Dict[str, Any],
                                  num_ues_per_tick: int) -> Dict[int, pd.DataFrame]:
        """
        Distributes UEs across the pre-generated spatial layout for each time tick.

        Args:
            spatial_layout: List of spatial cell dicts (output of _generate_spatial_layout).
            time_params: Dictionary with "total_ticks" and "time_weights"
                         (mapping area types to lists of density weights per tick).
            num_ues_per_tick: Target total number of UEs to generate per tick.

        Returns:
            A dictionary where keys are tick numbers and values are DataFrames
            of UE locations for that tick (columns: ue_id, lon, lat, tick, space_type).
        """
        if not spatial_layout:
            logger.error("Spatial layout is empty. Cannot distribute UEs.")
            return {}

        total_ticks = time_params.get("total_ticks", 1)
        time_weights_all_types = time_params.get("time_weights", {})
        
        # Pre-group spatial cells by their type for efficient lookup
        cells_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for cell_info in spatial_layout:
            cells_by_type.setdefault(cell_info['type'], []).append(cell_info)

        all_ue_data_per_tick: Dict[int, pd.DataFrame] = {}

        for tick in range(total_ticks):
            logger.debug(f"--- Generating UEs for Tick {tick} ---")
            tick_ue_data_list = []
            current_ue_id_in_tick = 0

            # Get weights for the current tick for types present in the layout
            tick_type_weights = {}
            for space_type in cells_by_type.keys(): # Iterate over types that actually exist in layout
                type_specific_weights = time_weights_all_types.get(space_type, [])
                if tick < len(type_specific_weights):
                    tick_type_weights[space_type] = type_specific_weights[tick]
                else:
                    tick_type_weights[space_type] = 0 # Default to 0 if tick exceeds weight list length
            
            total_weight_this_tick = sum(tick_type_weights.values())

            if total_weight_this_tick <= 1e-9: # Effectively zero
                logger.warning(f"Sum of weights is zero for tick {tick}. No UEs generated for this tick.")
                all_ue_data_per_tick[tick] = pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
                continue

            # Calculate target UE counts per space type for this tick
            target_counts_for_tick: Dict[str, int] = {}
            calculated_total_for_tick = 0
            for stype, weight in tick_type_weights.items():
                proportion = weight / total_weight_this_tick
                target_counts_for_tick[stype] = int(round(proportion * num_ues_per_tick))
                calculated_total_for_tick += target_counts_for_tick[stype]
            
            # Adjust counts if rounding caused mismatch with num_ues_per_tick
            difference = num_ues_per_tick - calculated_total_for_tick
            if difference != 0 and target_counts_for_tick:
                adjust_type = max(target_counts_for_tick, key=target_counts_for_tick.get)
                target_counts_for_tick[adjust_type] = max(0, target_counts_for_tick[adjust_type] + difference)

            logger.debug(f"Tick {tick} Target UE counts: {target_counts_for_tick}")

            # Place UEs
            for space_type, num_ues_for_type in target_counts_for_tick.items():
                if num_ues_for_type == 0:
                    continue

                available_cells_for_this_type = cells_by_type.get(space_type, [])
                if not available_cells_for_this_type:
                    logger.warning(f"Cannot place {num_ues_for_type} UEs for type '{space_type}' at tick {tick}: No spatial cells of this type in layout.")
                    continue
                
                # Distribute num_ues_for_type among available_cells_for_this_type
                # For simplicity, randomly choose a cell for each UE.
                # More sophisticated distribution (e.g., weighted by cell area) could be added.
                for _ in range(num_ues_for_type):
                    chosen_cell_info = np.random.choice(available_cells_for_this_type)
                    polygon_boundary = Polygon(chosen_cell_info['bounds'])
                    min_x, min_y, max_x, max_y = polygon_boundary.bounds

                    # Generate random point within the polygon's bounding box until it's inside the polygon
                    attempts = 0
                    max_attempts = 100
                    while attempts < max_attempts:
                        ue_lon_val = np.random.uniform(min_x, max_x)
                        ue_lat_val = np.random.uniform(min_y, max_y)
                        ue_point = Point(ue_lon_val, ue_lat_val)
                        if polygon_boundary.contains(ue_point):
                            tick_ue_data_list.append({
                                'ue_id': current_ue_id_in_tick,
                                c.LON: ue_lon_val,
                                c.LAT: ue_lat_val,
                                'tick': tick,
                                'space_type': space_type,
                                'voronoi_cell_id': chosen_cell_info['cell_id']
                            })
                            current_ue_id_in_tick += 1
                            break
                        attempts += 1
                    else: # Max attempts reached
                        logger.warning(f"Failed to place UE in cell {chosen_cell_info['cell_id']} (type '{space_type}') after {max_attempts} attempts for tick {tick}.")
            
            if tick_ue_data_list:
                all_ue_data_per_tick[tick] = pd.DataFrame(tick_ue_data_list)
            else:
                all_ue_data_per_tick[tick] = pd.DataFrame(columns=['ue_id', c.LON, c.LAT, 'tick', 'space_type'])
                logger.info(f"No UEs generated for tick {tick} based on current distribution logic.")
        
        logger.info(f"Finished UE distribution for {total_ticks} ticks.")
        return all_ue_data_per_tick

    def generate_traffic_demand(self,
                                topology_df: pd.DataFrame,
                                spatial_params: Dict[str, Any],
                                time_params: Dict[str, Any],
                                num_ues_per_tick: int) -> Tuple[Dict[int, pd.DataFrame], List[Dict[str, Any]]]:
        """
        Generates traffic demand by creating a spatial layout and distributing UEs over time.

        Args:
            topology_df: DataFrame with cell tower locations.
                         Expected columns: ecgi, site_id, cell_name, enodeb_id, 
                                           cell_az_deg, tac, cell_lat, cell_lon, 
                                           cell_id, cell_carrier_freq_mhz.
                                           (cell_txpwr_dbm is NOT expected by this model).
            spatial_params: Dictionary defining area "types" and "proportions".
            time_params: Dictionary defining "total_ticks" and "time_weights"
                         (mapping area types to lists of UE density weights per tick).
            num_ues_per_tick: The target total number of UEs to generate for each simulation tick.

        Returns:
            A tuple containing:
            1. ue_data_per_tick (Dict[int, pd.DataFrame]): Dictionary mapping tick number
               to a DataFrame of UE locations for that tick.
            2. spatial_layout (List[Dict[str, Any]]): List of dictionaries defining
               the generated spatial cells (polygons and their types).
        """
        logger.info("Starting traffic demand generation process...")

        # 1. Generate the fixed spatial layout
        spatial_layout = self._generate_spatial_layout(topology_df, spatial_params)
        if not spatial_layout:
            logger.error("Failed to generate spatial layout. Aborting traffic demand generation.")
            return {}, []

        # 2. Distribute UEs in the layout over time
        ue_data_per_tick = self._distribute_ues_in_layout(spatial_layout, time_params, num_ues_per_tick)

        logger.info("Traffic demand generation process completed.")
        return ue_data_per_tick, spatial_layout

if __name__ == '__main__':
    # This is a basic example of how to use TrafficDemandModel
    # For a full demonstration, see the traffic_demand_app.py or the Jupyter notebook.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running basic TrafficDemandModel example...")

    # Create dummy topology data
    example_topology_data = {
        c.CELL_ID: [f'cell_{i}' for i in range(5)],
        c.CELL_LAT: np.random.uniform(40.70, 40.75, 5),
        c.CELL_LON: np.random.uniform(-74.00, -73.95, 5)
    }
    example_topology_df = pd.DataFrame(example_topology_data)

    # Define example spatial and time parameters
    example_spatial_params = {
        "types": ["residential", "commercial", "park"],
        "proportions": [0.5, 0.3, 0.2]
    }
    example_time_params = {
        "total_ticks": 3, # Short simulation for example
        "time_weights": {
            "residential": [0.8, 0.7, 0.6],
            "commercial": [0.1, 0.2, 0.3],
            "park": [0.1, 0.1, 0.1]
        }
    }
    example_num_ues = 100

    # Instantiate and run the model
    traffic_model = TrafficDemandModel()
    try:
        ue_output, layout_output = traffic_model.generate_traffic_demand(
            topology_df=example_topology_df,
            spatial_params=example_spatial_params,
            time_params=example_time_params,
            num_ues_per_tick=example_num_ues
        )

        if ue_output:
            logger.info(f"Generated spatial layout with {len(layout_output)} cells.")
            # print("Example spatial layout (first cell):", layout_output[0] if layout_output else "N/A")
            
            for tick, df in ue_output.items():
                logger.info(f"Tick {tick}: Generated {len(df)} UEs.")
                # print(f"UEs for tick {tick} (first 5):\n{df.head()}\n")
        else:
            logger.warning("UE output was empty.")

    except Exception as e:
        logger.error(f"An error occurred during the example run: {e}", exc_info=True)
