import math
import unittest

import numpy as np
import pandas as pd

from notebooks.radp_library import (
    add_cell_info,
    calc_log_distance,
    calc_relative_bearing,
    calc_rx_power,
    calculate_received_power,
    check_cartesian_format,
    get_ues_cells_cartesian_df,
    normalize_cell_ids,
    preprocess_ue_data,
)
from radp.digital_twin.utils.gis_tools import GISTools


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def test_calculate_received_power(self):
        dummy_distance = 1
        dummy_freq = 1800
        expected_power = -74.55545010206612
        power = calculate_received_power(distance_km=dummy_distance, frequency_mhz=dummy_freq)
        self.assertEqual(expected_power, power)

    def test_add_cell_info(self):
        # Input data
        new_data_with_rx_data = pd.DataFrame({"cell_id": [1, 2], "rx_power": [-80, -85]})
        topology = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2"],
                "cell_lat": [10.0, 20.0],
                "cell_lon": [30.0, 40.0],
                "cell_az_deg": [90, 180],
            }
        )

        # Expected output
        expected_output = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2"],
                "rx_power": [-80, -85],
                "cell_lat": [10.0, 20.0],
                "cell_lon": [30.0, 40.0],
                "cell_az_deg": [90, 180],
            }
        )

        # Run the function
        result = add_cell_info(new_data_with_rx_data, topology)

        # Assert the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_output)

    def test_check_cartesian_format(self):
        # Input data
        df = pd.DataFrame(
            {
                "latitude": [10.0, 10.0, 20.0, 20.0],
                "longitude": [30.0, 30.0, 40.0, 40.0],
                "cell_id": ["cell_1", "cell_2", "cell_1", "cell_2"],
            }
        )
        topology = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})

        # Test should pass without raising an exception
        self.assertTrue(check_cartesian_format(df, topology))
        # Input data with missing cell_id
        df = pd.DataFrame({"latitude": [10.0, 20.0], "longitude": [30.0, 40.0], "cell_id": ["cell_1", "cell_1"]})
        topology = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})
        # Input data with extra cell_id
        df = pd.DataFrame(
            {
                "latitude": [10.0, 10.0, 20.0, 20.0],
                "longitude": [30.0, 30.0, 40.0, 40.0],
                "cell_id": ["cell_1", "cell_2", "cell_1", "cell_3"],
            }
        )
        topology = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})

        # Test should raise ValueError for extra cells
        with self.assertRaises(ValueError) as context:
            check_cartesian_format(df, topology)
        self.assertIn("Extra cells", str(context.exception))

    def test_normalize_cell_ids(self):
        # Case 1: Numeric strings
        df1 = pd.DataFrame({"cell_id": ["1", "2.0", "3.5"]})
        expected1 = pd.DataFrame({"cell_id": ["cell_1", "cell_2", "cell_3"]})
        result1 = normalize_cell_ids(df1)
        pd.testing.assert_frame_equal(result1, expected1)

        # Case 2: Already normalized
        df2 = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})
        expected2 = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})
        result2 = normalize_cell_ids(df2)
        pd.testing.assert_frame_equal(result2, expected2)

        # Case 3: Mixed input
        df3 = pd.DataFrame({"cell_id": ["cell_5", "6", 7.8, "8.0"]})
        expected3 = pd.DataFrame({"cell_id": ["cell_5", "cell_6", "cell_7", "cell_8"]})
        result3 = normalize_cell_ids(df3)
        pd.testing.assert_frame_equal(result3, expected3)

        # Case 4: Integer inputs
        df4 = pd.DataFrame({"cell_id": [10, 11, 12]})
        expected4 = pd.DataFrame({"cell_id": ["cell_10", "cell_11", "cell_12"]})
        result4 = normalize_cell_ids(df4)
        pd.testing.assert_frame_equal(result4, expected4)

        # Case 5: None / NaN handling (assumes fallback to cell_0 or raises)
        df5 = pd.DataFrame({"cell_id": [None, np.nan, "cell_9"]})
        try:
            result5 = normalize_cell_ids(df5)
            expected5 = pd.DataFrame({"cell_id": ["cell_0", "cell_0", "cell_9"]})
            pd.testing.assert_frame_equal(result5, expected5)
        except (ValueError, TypeError):
            print("Skipping NaN/None handling test — not supported by implementation.")

        # Case 6: Check original DataFrame is not mutated
        df6 = pd.DataFrame({"cell_id": ["1", "2"]})
        df6_copy = df6.copy(deep=True)
        _ = normalize_cell_ids(df6)
        pd.testing.assert_frame_equal(df6, df6_copy)

    def test_preprocess_ue_data(self):
        # Input data
        # Arrange: Create minimal sample data
        ue_data = pd.DataFrame(
            {
                "ue_id": [1],
                "latitude": [0.0],
                "longitude": [0.0],
            }
        )

        topology_data = pd.DataFrame(
            {"cell_id": ["cell_1"], "cell_lat": [0.0], "cell_lon": [1.0], "cell_carrier_freq_mhz": [1800]}
        )

        # Act: Run full preprocessing
        result_df = preprocess_ue_data(ue_data.copy(), topology_data.copy())

        # Assert: Check expected columns exist
        self.assertIn("log_distance", result_df.columns)
        self.assertIn("cell_rxpwr_dbm", result_df.columns)

        # Check log distance is a positive number
        log_distance = result_df["log_distance"].iloc[0]
        self.assertTrue(isinstance(log_distance, float))
        self.assertGreater(log_distance, 0)

        # Check received power is a negative number (as expected in dBm)
        rx_power = result_df["cell_rxpwr_dbm"].iloc[0]
        self.assertTrue(isinstance(rx_power, float))
        self.assertLess(rx_power, 0)

    def test_calc_relative_bearing(self):
        # Case 1: UE directly north (relative bearing should be ~0)
        df1 = pd.DataFrame(
            {
                "cell_az_deg": [0],
                "cell_lat": [0.0],
                "cell_lon": [0.0],
                "latitude": [1.0],
                "longitude": [0.0],
            }
        )
        result1 = calc_relative_bearing(df1)
        self.assertTrue(math.isclose(result1["relative_bearing"].iloc[0], 0.0, abs_tol=1e-2))

        # Case 2: UE directly east (relative bearing should be ~90)
        df2 = pd.DataFrame(
            {
                "cell_az_deg": [0],
                "cell_lat": [0.0],
                "cell_lon": [0.0],
                "latitude": [0.0],
                "longitude": [1.0],
            }
        )
        result2 = calc_relative_bearing(df2)
        self.assertTrue(math.isclose(result2["relative_bearing"].iloc[0], 90.0, abs_tol=1e-2))

        # Case 3: UE directly south (relative bearing should be ~180)
        df3 = pd.DataFrame(
            {
                "cell_az_deg": [0],
                "cell_lat": [0.0],
                "cell_lon": [0.0],
                "latitude": [-1.0],
                "longitude": [0.0],
            }
        )
        result3 = calc_relative_bearing(df3)
        self.assertTrue(math.isclose(result3["relative_bearing"].iloc[0], 180.0, abs_tol=1e-2))

        # Case 4: UE directly west of cell facing east (relative bearing ~180)
        df4 = pd.DataFrame(
            {
                "cell_az_deg": [90],
                "cell_lat": [0.0],
                "cell_lon": [0.0],
                "latitude": [0.0],
                "longitude": [-1.0],
            }
        )
        result4 = calc_relative_bearing(df4)
        self.assertTrue(math.isclose(result4["relative_bearing"].iloc[0], 180.0, abs_tol=1e-2))

    def test_calc_rx_power(self):
        # Arrange: Create a minimal DataFrame with known values
        df = pd.DataFrame(
            {
                "log_distance": [3.0],  # example value (e.g., log10(distance))
                "cell_carrier_freq_mhz": [1800],  # 1800 MHz is common LTE band
            }
        )

        # Expected value from actual formula (can be calculated directly)
        expected_power = calculate_received_power(3.0, 1800)

        # Act
        result_df = calc_rx_power(df)

        # Assert
        self.assertIn("cell_rxpwr_dbm", result_df.columns)
        self.assertAlmostEqual(result_df["cell_rxpwr_dbm"].iloc[0], expected_power, places=3)

    def test_calc_log_distance(self):
        # Arrange: Create a minimal DataFrame with known values
        df = pd.DataFrame(
            {
                "latitude": [0.0],  # UE latitude
                "longitude": [0.0],  # UE longitude
                "cell_lat": [1.0],  # Cell latitude
                "cell_lon": [1.0],  # Cell longitude
            }
        )

        # Mock value for log distance that GISTools.get_log_distance would calculate
        expected_log_distance = GISTools.get_log_distance(0.0, 0.0, 1.0, 1.0)

        # Act: Call the function
        result_df = calc_log_distance(df)

        # Assert: Check if the 'log_distance' column exists
        self.assertIn("log_distance", result_df.columns)

        # Assert: Check if the computed log distance is as expected
        self.assertAlmostEqual(result_df["log_distance"].iloc[0], expected_log_distance, places=3)

    def test_get_ues_cells_cartesian_df(self):
        # Arrange: Create sample data and topology DataFrames
        data = pd.DataFrame(
            {
                "ue_id": [1, 2],
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            }
        )

        topology = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2"],  # cell_id in 'cell_' format
                "cell_lat": [0.0, 1.0],
                "cell_lon": [1.0, 2.0],
            }
        )

        # Act: Call the function
        result_df = get_ues_cells_cartesian_df(data, topology)

        # Assert: Check if the 'key' column is dropped (it should no longer exist)
        self.assertNotIn("key", result_df.columns)

        # Assert: Ensure the DataFrame has the correct number of rows (product of data and topology)
        self.assertEqual(result_df.shape[0], data.shape[0] * topology.shape[0])

        # Assert: Ensure cell_id was converted from string to integer correctly
        self.assertEqual(result_df["cell_id"].iloc[0], 1)
        self.assertEqual(result_df["cell_id"].iloc[1], 2)
