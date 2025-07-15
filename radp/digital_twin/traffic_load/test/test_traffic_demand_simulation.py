import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Conditional import for shapely and scipy
try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Adjust relative import based on test runner's CWD
try:
    from radp.digital_twin.traffic_load.traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c
except ImportError:
    # Fallback for environments where PYTHONPATH is not fully set up
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    maveric_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
    if maveric_root not in sys.path:
        sys.path.insert(0, maveric_root)
    from radp.digital_twin.traffic_load.traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c

# Suppress logging for cleaner test output
logging.disable(logging.CRITICAL)


@unittest.skipIf(not SHAPELY_AVAILABLE, "Shapely library not installed, skipping tests.")
class TestTrafficDemandModel(unittest.TestCase):
    def setUp(self):
        """Set up common test data and initialize the model."""
        self.traffic_model = TrafficDemandModel()
        np.random.seed(42)  # for reproducible tests

        # Sample topology data with sufficient points for Voronoi
        self.topology_df = pd.DataFrame(
            {
                c.CELL_ID: [f"cell_{i}" for i in range(6)],
                c.CELL_LAT: [40.71, 40.72, 40.73, 40.715, 40.725, 40.705],
                c.CELL_LON: [-74.00, -73.99, -73.98, -74.005, -73.995, -73.985],
            }
        )

        # Topology data with too few points for Voronoi
        self.topology_df_small = pd.DataFrame(
            {
                c.CELL_ID: ["cell_A", "cell_B", "cell_C"],
                c.CELL_LAT: [40.71, 40.72, 40.73],
                c.CELL_LON: [-74.00, -73.99, -73.98],
            }
        )

        # Topology data with very close points to test buffer logic
        self.topology_df_very_close = pd.DataFrame(
            {
                c.CELL_ID: [f"cell_vc_{i}" for i in range(4)],
                c.CELL_LAT: [40.71, 40.710000001, 40.71, 40.710000001],
                c.CELL_LON: [-74.00, -74.00, -74.000000001, -74.000000001],
            }
        )

        # Standard simulation parameters
        self.spatial_params = {"types": ["residential", "commercial", "park"], "proportions": [0.5, 0.3, 0.2]}
        self.time_params = {
            "total_ticks": 2,
            "time_weights": {"residential": [0.8, 0.6], "commercial": [0.1, 0.3], "park": [0.1, 0.1]},
        }
        self.num_ues_per_tick = 100

    def tearDown(self):
        """Re-enable logging after tests."""
        logging.disable(logging.NOTSET)

    # --- Tests for _space_boundary ---
    def test_space_boundary_valid(self):
        """Test _space_boundary with valid topology data."""
        boundary = self.traffic_model._space_boundary(self.topology_df, buffer_percent=0.1)
        self.assertIsInstance(boundary, dict)
        self.assertIn("min_lon_buffered", boundary)
        self.assertLess(boundary["min_lon_buffered"], self.topology_df[c.CELL_LON].min())
        self.assertGreater(boundary["max_lat_buffered"], self.topology_df[c.CELL_LAT].max())

    def test_space_boundary_empty_df(self):
        """Test _space_boundary with an empty DataFrame raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Cell topology data cannot be empty"):
            self.traffic_model._space_boundary(pd.DataFrame())

    def test_space_boundary_missing_columns(self):
        """Test _space_boundary with missing lat/lon columns raises KeyError."""
        bad_df = pd.DataFrame({"id": [1, 2]})
        with self.assertRaises(KeyError):
            self.traffic_model._space_boundary(bad_df)

    # --- Tests for _assign_space_types_to_polygons ---
    def test_assign_space_types_valid(self):
        """Test _assign_space_types_to_polygons correctly assigns types based on proportions."""
        polygons = [Polygon([(i, i), (i + 1, i), (i + 1, i + 1)]) for i in range(10)]
        params = {"types": ["A", "B"], "proportions": [0.7, 0.3]}
        result = self.traffic_model._assign_space_types_to_polygons(polygons, params)
        self.assertEqual(len(result), 10)
        type_counts = pd.Series([r["type"] for r in result]).value_counts()
        self.assertEqual(type_counts["A"], 7)
        self.assertEqual(type_counts["B"], 3)

    def test_assign_space_types_rounding_adjustment(self):
        """Test that type counts are adjusted correctly to sum to the total number of polygons."""
        polygons = [Polygon([(0, 0), (1, 0), (1, 1)])] * 3  # 3 polygons
        params = {"types": ["A", "B", "C"], "proportions": [0.33, 0.33, 0.33]}  # sums to 0.99
        result = self.traffic_model._assign_space_types_to_polygons(polygons, params)
        # np.round([0.99, 0.99, 0.99]) -> [1, 1, 1]. Sum is 3. No diff.
        self.assertEqual(len(result), 3)
        type_counts = pd.Series([r["type"] for r in result]).value_counts()
        # Expect one of each due to rounding and adjustment
        self.assertEqual(type_counts.sum(), 3)
        self.assertIn(1, type_counts.values)

    def test_assign_space_types_invalid_params(self):
        """Test _assign_space_types returns empty for mismatched or empty params."""
        polygons = [Polygon([(0, 0), (1, 0), (1, 1)])]
        # Mismatched lengths
        self.assertEqual(
            self.traffic_model._assign_space_types_to_polygons(polygons, {"types": ["A"], "proportions": [0.5, 0.5]}),
            [],
        )
        # Empty types/proportions
        self.assertEqual(
            self.traffic_model._assign_space_types_to_polygons(polygons, {"types": [], "proportions": []}), []
        )

    # --- Tests for generate_spatial_layout ---
    @unittest.skipIf(not SCIPY_AVAILABLE, "scipy not available, skipping integration test.")
    def test_generate_spatial_layout_integration(self):
        """Perform an integration test of generate_spatial_layout with real Voronoi."""
        layout = self.traffic_model.generate_spatial_layout(self.topology_df, self.spatial_params)
        self.assertIsInstance(layout, list)
        self.assertGreater(len(layout), 0)
        # Check that the number of generated cells matches the number of input sites.
        self.assertEqual(len(layout), len(self.topology_df))
        cell = layout[0]
        self.assertIn("bounds", cell)
        self.assertIn("type", cell)
        self.assertIn("cell_id", cell)
        self.assertIn(cell["type"], self.spatial_params["types"])
        # Check that a valid polygon was created
        self.assertGreater(len(cell["bounds"]), 3)

    def test_generate_spatial_layout_too_few_sites(self):
        """Test generate_spatial_layout returns empty with insufficient sites."""
        layout = self.traffic_model.generate_spatial_layout(self.topology_df_small, self.spatial_params)
        self.assertEqual(layout, [])

    @patch("radp.digital_twin.traffic_load.traffic_demand_simulation.Voronoi", None)
    def test_generate_spatial_layout_no_scipy(self):
        """Test generate_spatial_layout returns empty when scipy is not available."""
        # Re-initialize model inside this test to re-trigger the module-level import check
        model = TrafficDemandModel()
        layout = model.generate_spatial_layout(self.topology_df, self.spatial_params)
        self.assertEqual(layout, [])

    # --- Tests for distribute_ues_over_time ---
    def test_distribute_ues_valid(self):
        """Test UE distribution for a single tick and verify counts and placement."""
        layout = [{"bounds": [(0, 0), (10, 0), (10, 10), (0, 10)], "type": "A", "cell_id": 0}]
        time_params = {"total_ticks": 1, "time_weights": {"A": [1.0]}}
        ue_data = self.traffic_model.distribute_ues_over_time(layout, time_params, 50)

        self.assertIn(0, ue_data)
        ue_df = ue_data[0]
        self.assertEqual(len(ue_df), 50)
        self.assertTrue((ue_df["space_type"] == "A").all())

        # Verify all points are within the polygon
        poly = Polygon(layout[0]["bounds"])
        for _, row in ue_df.iterrows():
            self.assertTrue(poly.contains(Point(row[c.LON], row[c.LAT])))

    def test_distribute_ues_count_adjustment_logic(self):
        """Test that UE counts are correctly distributed, including remainders."""
        layout = [
            {"bounds": [(0, 0), (1, 1), (0, 1)], "type": "res", "cell_id": 0},
            {"bounds": [(2, 2), (3, 3), (2, 3)], "type": "com", "cell_id": 1},
        ]
        time_params = {"total_ticks": 1, "time_weights": {"res": [0.805], "com": [0.195]}}
        num_ues = 100  # Expect 81 for res (80.5 rounded up), 19 for com (19.5 rounded down)

        ue_data = self.traffic_model.distribute_ues_over_time(layout, time_params, num_ues)
        ue_df = ue_data[0]

        self.assertEqual(len(ue_df), num_ues)
        counts = ue_df["space_type"].value_counts()
        self.assertEqual(counts["res"], 81)  # 80 + 1 from remainder
        self.assertEqual(counts["com"], 19)

    def test_distribute_ues_empty_layout(self):
        """Test UE distribution with an empty spatial layout returns empty dict."""
        result = self.traffic_model.distribute_ues_over_time([], self.time_params, 100)
        self.assertEqual(result, {})

    def test_distribute_ues_zero_weights(self):
        """Test UE distribution for a tick where all weights are zero."""
        layout = [{"bounds": [(0, 0), (1, 1), (0, 1)], "type": "A", "cell_id": 0}]
        time_params = {"total_ticks": 1, "time_weights": {"A": [0.0]}}
        ue_data = self.traffic_model.distribute_ues_over_time(layout, time_params, 100)
        self.assertEqual(len(ue_data[0]), 0)  # Should have a dataframe, but it's empty

    def test_distribute_ues_missing_type_in_weights(self):
        """Test behavior when a space type is missing from time_weights config."""
        layout = [
            {"bounds": [(0, 0), (1, 1), (0, 1)], "type": "A", "cell_id": 0},
            {"bounds": [(2, 2), (3, 3), (2, 3)], "type": "B", "cell_id": 1},
        ]
        time_params = {"total_ticks": 1, "time_weights": {"A": [1.0]}}  # 'B' is missing
        ue_data = self.traffic_model.distribute_ues_over_time(layout, time_params, 100)
        ue_df = ue_data[0]
        self.assertEqual(len(ue_df), 100)
        self.assertTrue((ue_df["space_type"] == "A").all())  # All UEs go to type 'A'
        self.assertNotIn("B", ue_df["space_type"].unique())

    # --- Tests for the main orchestration method ---
    @patch.object(TrafficDemandModel, "distribute_ues_over_time")
    @patch.object(TrafficDemandModel, "generate_spatial_layout")
    def test_generate_traffic_demand_orchestration(self, mock_gen_layout, mock_dist_ues):
        """Test the main method orchestrates calls to layout and distribution methods."""
        mock_layout = [{"id": 1}]
        mock_ues = {0: pd.DataFrame()}
        mock_gen_layout.return_value = mock_layout
        mock_dist_ues.return_value = mock_ues

        ue_data, layout = self.traffic_model.generate_traffic_demand(
            self.topology_df, self.spatial_params, self.time_params, self.num_ues_per_tick
        )

        mock_gen_layout.assert_called_once_with(self.topology_df, self.spatial_params)
        mock_dist_ues.assert_called_once_with(mock_layout, self.time_params, self.num_ues_per_tick)
        self.assertEqual(ue_data, mock_ues)
        self.assertEqual(layout, mock_layout)

    @patch.object(TrafficDemandModel, "generate_spatial_layout")
    def test_generate_traffic_demand_handles_empty_layout(self, mock_gen_layout):
        """Test that an empty layout from generation results in empty final output."""
        mock_gen_layout.return_value = []

        ue_data, layout = self.traffic_model.generate_traffic_demand(
            self.topology_df, self.spatial_params, self.time_params, self.num_ues_per_tick
        )

        self.assertEqual(ue_data, {})
        self.assertEqual(layout, [])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
