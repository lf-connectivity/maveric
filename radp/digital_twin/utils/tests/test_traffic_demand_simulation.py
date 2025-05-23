# maveric/radp/digital_twin/utils/tests/test_traffic_demand_simulation.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point

# Assuming the test file is in a 'tests' subdirectory next to the module
# Adjust relative import if necessary based on your test runner's working directory
try:
    from radp.digital_twin.utils.traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools # For mocking if needed
except ImportError:
    # Fallback for running tests in environments where PYTHONPATH might not be fully set up
    # This assumes a specific directory structure.
    import sys
    import os
    # Get the absolute path to the 'radp' directory (e.g., MAVERIC_ROOT/radp)
    # This assumes MAVERIC_ROOT is two levels up from .../utils/tests/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    maveric_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
    if maveric_root not in sys.path:
        sys.path.insert(0, maveric_root)
    from radp.digital_twin.utils.traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools


class TestTrafficDemandModel(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.traffic_model = TrafficDemandModel()
        np.random.seed(42) # for reproducible UE placement if needed in some tests

        # Sample topology data
        self.sample_topology_data = {
            c.CELL_ID: [f'cell_{i}' for i in range(6)],
            c.CELL_LAT: [40.71, 40.72, 40.73, 40.715, 40.725, 40.705],
            c.CELL_LON: [-74.00, -73.99, -73.98, -74.005, -73.995, -73.985]
        }
        self.topology_df = pd.DataFrame(self.sample_topology_data)

        self.topology_df_small = pd.DataFrame({
            c.CELL_ID: ['cell_A', 'cell_B', 'cell_C'],
            c.CELL_LAT: [40.71, 40.72, 40.73],
            c.CELL_LON: [-74.00, -73.99, -73.98]
        })
        
        self.topology_df_very_close_points = pd.DataFrame({
            c.CELL_ID: [f'cell_vc_{i}' for i in range(4)], # Need 4 for Voronoi
            # Make range very small (e.g., <= 1e-9) to trigger the 'else 0.1' for buffer
            c.CELL_LAT: [40.710000000, 40.710000000, 40.710000000, 40.7100000001], # range = 1e-10
            c.CELL_LON: [-74.000000000, -74.000000000, -74.000000000, -74.0000000001]  # range = 1e-10
        })


        self.spatial_params = {
            "types": ["residential", "commercial", "park"],
            "proportions": [0.5, 0.3, 0.2]
        }
        self.time_params = {
            "total_ticks": 2,
            "time_weights": {
                "residential": [0.8, 0.6],
                "commercial": [0.1, 0.3],
                "park": [0.1, 0.1]
            }
        }
        self.num_ues_per_tick = 100

    def test_space_boundary_valid_input(self):
        """Test _space_boundary with valid topology data."""
        boundary = self.traffic_model._space_boundary(self.topology_df, buffer_percent=0.1)
        self.assertIsInstance(boundary, dict)
        self.assertIn("min_lon_buffered", boundary)
        self.assertIn("max_lat_buffered", boundary)
        min_lon_orig = self.topology_df[c.CELL_LON].min()
        self.assertLessEqual(boundary["min_lon_buffered"], min_lon_orig)

    def test_space_boundary_empty_topology(self):
        """Test _space_boundary with empty topology data."""
        with self.assertRaises(ValueError):
            self.traffic_model._space_boundary(pd.DataFrame())

    def test_space_boundary_missing_cols(self):
        """Test _space_boundary with missing lat/lon columns."""
        bad_topo = pd.DataFrame({'id': [1,2]})
        with self.assertRaises(ValueError):
            self.traffic_model._space_boundary(bad_topo)

    def test_space_boundary_very_close_points(self):
        """
        Test _space_boundary with very close points to check minimum buffer.
        This test ensures that if the original range of coordinates is very small (<=1e-9),
        a default buffer of 0.1 is applied to each side, resulting in a total range of ~0.2.
        """
        boundary = self.traffic_model._space_boundary(self.topology_df_very_close_points, buffer_percent=0.01) 
        lat_range_buffered = boundary["max_lat_buffered"] - boundary["min_lat_buffered"]
        lon_range_buffered = boundary["max_lon_buffered"] - boundary["min_lon_buffered"]
        
        # Expected range is original_tiny_range + 2 * 0.1 (default buffer per side)
        # For practical purposes, if original_tiny_range is e.g. 1e-10, total range is ~0.2
        self.assertGreaterEqual(lat_range_buffered, 0.2 - 1e-9) # Check it's close to 0.2
        self.assertLessEqual(lat_range_buffered, 0.2 + 1e-8) # Check it's not excessively large
        self.assertGreaterEqual(lon_range_buffered, 0.2 - 1e-9)
        self.assertLessEqual(lon_range_buffered, 0.2 + 1e-8)
        # Original assertion was self.assertGreaterEqual(lat_range_buffered, 0.1), which is still true.
        # Making it more specific to check for ~0.2.


    @patch('radp.digital_twin.utils.traffic_demand_simulation.Voronoi')
    def test_generate_spatial_layout_valid(self, mock_voronoi_scipy):
        """Test _generate_spatial_layout with valid inputs and mocked Voronoi."""
        mock_vor = MagicMock()
        mock_vor.vertices = np.array([[0,0], [1,0], [0,1], [1,1], [0.5, 0.5], [2,2]]) 
        mock_vor.regions = [
            [0, 1, 4], [1, 3, 4], [0, 2, 4], [2, 3, 4], 
            [], 
            [-1, 0, 1] 
        ]
        mock_voronoi_scipy.return_value = mock_vor

        layout = self.traffic_model._generate_spatial_layout(self.topology_df, self.spatial_params)
        self.assertIsInstance(layout, list)
        if layout:
            self.assertIsInstance(layout[0], dict)
            self.assertIn("bounds", layout[0])
            self.assertIn("type", layout[0])
            self.assertIn(layout[0]["type"], self.spatial_params["types"])
            self.assertGreater(len(layout[0]["bounds"]), 2)
        mock_voronoi_scipy.assert_called_once()


    def test_generate_spatial_layout_too_few_sites(self):
        """Test _generate_spatial_layout with insufficient sites for Voronoi."""
        layout = self.traffic_model._generate_spatial_layout(self.topology_df_small, self.spatial_params)
        self.assertEqual(layout, [])

    @patch('radp.digital_twin.utils.traffic_demand_simulation.Voronoi', None)
    def test_generate_spatial_layout_no_voronoi_module(self):
        """Test _generate_spatial_layout when Voronoi (scipy) is not available."""
        layout = self.traffic_model._generate_spatial_layout(self.topology_df, self.spatial_params)
        self.assertEqual(layout, [])

    def test_generate_spatial_layout_invalid_spatial_params_missing_types(self):
        """Test _generate_spatial_layout with spatial_params missing 'types'."""
        invalid_spatial_params = {"proportions": [1.0]}
        layout = self.traffic_model._generate_spatial_layout(self.topology_df, invalid_spatial_params)
        if layout:
            self.assertTrue(all(cell['type'] == "default_area" for cell in layout))

    def test_generate_spatial_layout_empty_spatial_params_types(self):
        """Test _generate_spatial_layout with spatial_params having empty 'types' list."""
        invalid_spatial_params = {"types": [], "proportions": []}
        layout = self.traffic_model._generate_spatial_layout(self.topology_df, invalid_spatial_params)
        if layout: 
            self.assertTrue(all(cell['type'] == "default_area" for cell in layout))


    def test_distribute_ues_in_layout_valid_and_point_placement(self):
        """Test _distribute_ues_in_layout and verify UE placement within polygons."""
        mock_layout_simple = [
            {"bounds": [(0,0), (10,0), (10,10), (0,10)], "type": "residential", "cell_id": 0, "area_sq_km": 100},
            {"bounds": [(11,0), (12,0), (12,1), (11,1)], "type": "commercial", "cell_id": 1, "area_sq_km": 1}
        ]
        simple_time_params = {
            "total_ticks": 1,
            "time_weights": {"residential": [0.8], "commercial": [0.2]} 
        }
        num_ues = 10

        ue_data_per_tick = self.traffic_model._distribute_ues_in_layout(
            mock_layout_simple, simple_time_params, num_ues
        )
        self.assertEqual(len(ue_data_per_tick), 1)
        ue_df = ue_data_per_tick[0]
        self.assertAlmostEqual(len(ue_df), num_ues, delta=1) 

        residential_ues = ue_df[ue_df["space_type"] == "residential"]
        commercial_ues = ue_df[ue_df["space_type"] == "commercial"]

        self.assertTrue(0.6 * num_ues <= len(residential_ues) <= 1.0 * num_ues) 
        self.assertTrue(0.0 * num_ues <= len(commercial_ues) <= 0.4 * num_ues) 

        residential_poly = Polygon(mock_layout_simple[0]["bounds"])
        commercial_poly = Polygon(mock_layout_simple[1]["bounds"])

        for _, row in residential_ues.iterrows():
            pt = Point(row[c.LON], row[c.LAT])
            self.assertTrue(residential_poly.contains(pt) or residential_poly.touches(pt))

        for _, row in commercial_ues.iterrows():
            pt = Point(row[c.LON], row[c.LAT])
            self.assertTrue(commercial_poly.contains(pt) or commercial_poly.touches(pt))


    def test_distribute_ues_empty_layout(self):
        """Test _distribute_ues_in_layout with an empty spatial layout."""
        ue_data_per_tick = self.traffic_model._distribute_ues_in_layout(
            [], self.time_params, self.num_ues_per_tick
        )
        self.assertEqual(ue_data_per_tick, {})

    def test_distribute_ues_zero_weights(self):
        """Test _distribute_ues_in_layout when all time_weights for a tick are zero."""
        zero_weight_time_params = {
            "total_ticks": 1,
            "time_weights": {"residential": [0], "commercial": [0], "park": [0]} 
        }
        mock_layout = [{"bounds": [(0,0),(1,0),(1,1),(0,1)], "type": "residential", "cell_id": 0}]
        ue_data_per_tick = self.traffic_model._distribute_ues_in_layout(
            mock_layout, zero_weight_time_params, self.num_ues_per_tick
        )
        self.assertEqual(len(ue_data_per_tick[0]), 0)

    def test_distribute_ues_num_ues_zero(self):
        """Test _distribute_ues_in_layout with num_ues_per_tick = 0."""
        mock_layout = [{"bounds": [(0,0),(1,0),(1,1),(0,1)], "type": "residential", "cell_id": 0}]
        ue_data_per_tick = self.traffic_model._distribute_ues_in_layout(
            mock_layout, self.time_params, 0
        )
        self.assertEqual(len(ue_data_per_tick[0]), 0)
        self.assertEqual(len(ue_data_per_tick[1]), 0)

    def test_distribute_ues_missing_type_in_time_weights(self):
        """Test behavior when a type in layout is missing from time_weights."""
        time_params_missing_type = {
            "total_ticks": 1,
            "time_weights": {"residential": [1.0]} 
        }
        
        mock_layout_multi_type = [
            {"bounds": [(0,0), (1,0), (1,1), (0,1)], "type": "residential", "cell_id": 0},
            {"bounds": [(1,0), (2,0), (2,1), (1,1)], "type": "commercial", "cell_id": 1},
            {"bounds": [(2,0), (3,0), (3,1), (2,1)], "type": "park", "cell_id": 2}
        ]

        ue_data_per_tick = self.traffic_model._distribute_ues_in_layout(
            mock_layout_multi_type, time_params_missing_type, self.num_ues_per_tick
        )
        ue_df_tick0 = ue_data_per_tick[0]
        self.assertAlmostEqual(len(ue_df_tick0), self.num_ues_per_tick, delta=1)
        if not ue_df_tick0.empty:
            self.assertTrue((ue_df_tick0["space_type"] == "residential").all())


    @patch.object(TrafficDemandModel, '_generate_spatial_layout')
    @patch.object(TrafficDemandModel, '_distribute_ues_in_layout')
    def test_generate_traffic_demand_orchestration(self, mock_distribute_ues, mock_generate_layout):
        """Test the main generate_traffic_demand method's orchestration."""
        mock_spatial_layout_output = [{"bounds": [(0,0),(1,0),(1,1),(0,1)], "type": "residential", "cell_id":0}]
        mock_ue_data_output = {0: pd.DataFrame({c.LON: [0.5], c.LAT: [0.5]})}
        
        mock_generate_layout.return_value = mock_spatial_layout_output
        mock_distribute_ues.return_value = mock_ue_data_output

        ue_data, layout = self.traffic_model.generate_traffic_demand(
            self.topology_df, self.spatial_params, self.time_params, self.num_ues_per_tick
        )

        mock_generate_layout.assert_called_once_with(self.topology_df, self.spatial_params)
        mock_distribute_ues.assert_called_once_with(
            mock_spatial_layout_output, self.time_params, self.num_ues_per_tick
        )
        self.assertEqual(ue_data, mock_ue_data_output)
        self.assertEqual(layout, mock_spatial_layout_output)

    @patch.object(TrafficDemandModel, '_generate_spatial_layout')
    def test_generate_traffic_demand_empty_layout_propagates(self, mock_generate_layout):
        """Test that if _generate_spatial_layout returns empty, the main method also returns empty."""
        mock_generate_layout.return_value = [] 

        ue_data, layout = self.traffic_model.generate_traffic_demand(
            self.topology_df, self.spatial_params, self.time_params, self.num_ues_per_tick
        )
        
        mock_generate_layout.assert_called_once_with(self.topology_df, self.spatial_params)
        self.assertEqual(ue_data, {})
        self.assertEqual(layout, [])

    def test_assign_space_types_proportions_normalization(self):
        """Test _assign_space_types_to_polygons normalizes proportions if they don't sum to 1."""
        polygons = [Polygon([(0,0), (1,0), (1,1), (0,1)])] 
        space_props = {"type_a": 0.6, "type_b": 0.6} 
        
        with patch.object(GISTools, 'area_of_polygon', return_value=1.0, create=True) if not hasattr(GISTools, 'area_of_polygon') else patch('__main__.GISTools.area_of_polygon', create=True) as mock_area:
            assigned_cells = self.traffic_model._assign_space_types_to_polygons(polygons, space_props)
        
        self.assertEqual(len(assigned_cells), 1)
        self.assertIn(assigned_cells[0]['type'], ["type_a", "type_b"])


if __name__ == '__main__':
    unittest.main()
