import unittest
import pandas as pd
import numpy as np
import pytest
from apps.mobility_robustness_optimization.mobility_robustness_optimization import MobilityRobustnessOptimization


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def setUp(self):
        # Sample UE data
        self.ue_data = pd.DataFrame(
            {
                "mock_ue_id": [1, 2, 1, 2, 3],
                "longitude": [10.0, 11.0, 12.0, 11.5, 9.5],
                "latitude": [20.0, 21.0, 20.5, 21.5, 19.5],
                "tick": [0, 0, 1, 1, 2],
            }
        )

        # Sample topology data
        self.topology = pd.DataFrame(
            {
                "cell_lat": [20.1, 21.1, 19.9],
                "cell_lon": [10.1, 11.1, 9.9],
                "cell_id": [1, 2, 3],
                "cell_az_deg": [45, 90, 180],
                "cell_carrier_freq_mhz": [1800, 2100, 900],
            }
        )

        # Sample prediction data
        self.prediction_data = pd.DataFrame(
            {
                "mock_ue_id": [1, 2, 1, 2, 3],
                "longitude": [10.0, 11.0, 12.0, 11.5, 9.5],
                "latitude": [20.0, 21.0, 20.5, 21.5, 19.5],
                "tick": [0, 0, 1, 1, 2],
            }
        )

        # Initialize the MobilityRobustnessOptimization class
        self.mro = MobilityRobustnessOptimization(
            ue_data=self.ue_data,
            topology=self.topology,
            prediction_data=self.prediction_data,
            tx_power_dbm=23,
        )

    def test_connect_ue_to_all_cells(self):
        # Test connecting UE data to all cells
        combined_df = self.mro._connect_ue_to_all_cells()

        # Check if the combined dataframe has the correct shape
        expected_rows = len(self.ue_data) * len(self.topology)  # Cartesian product
        self.assertEqual(len(combined_df), expected_rows)

        # Check if necessary columns are present
        self.assertIn("cell_lat", combined_df.columns)
        self.assertIn("cell_lon", combined_df.columns)
        self.assertIn("mock_ue_id", combined_df.columns)

    def test_calculate_received_power(self):
        # Test the received power calculation
        distance_km = 1.0  # Example distance in km
        frequency_mhz = 1800  # Example frequency in MHz

        received_power = self.mro._calculate_received_power(distance_km, frequency_mhz)

        # Manually calculate expected result
        distance_m = distance_km * 1000
        fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55
        expected_received_power = self.mro.tx_power_dbm - fspl_db

        self.assertAlmostEqual(received_power, expected_received_power, places=2)

    def test_preprocess_ue_topology_data(self):
        # Test the preprocessing of UE and topology data
        processed_data = self.mro._preprocess_ue_topology_data()

        # Check if the processed dataframe has the expected columns
        expected_columns = [
            "mock_ue_id",
            "longitude",
            "latitude",
            "cell_lat",
            "cell_lon",
            "cell_id",
            "log_distance",
            "cell_rxpwr_dbm",
        ]
        for col in expected_columns:
            self.assertIn(col, processed_data.columns)

        # Verify if the log distance calculation is performed correctly
        self.assertFalse(processed_data["log_distance"].isnull().any())

    def test_preprocess_ue_training_data(self):
        # Test the preprocessing of training data
        training_data = self.mro._preprocess_ue_training_data()

        # Check if training data is returned in a dictionary format per cell
        self.assertIsInstance(training_data, dict)

        # Verify that training data contains entries for each cell_id
        for cell_id in self.topology["cell_id"]:
            self.assertIn(cell_id, training_data)

    def test_preprocess_prediction_data(self):
        # Test the preprocessing of prediction data
        processed_prediction_data = self.mro._preprocess_prediction_data()

        # Check if the processed prediction data contains expected columns
        expected_columns = [
            "mock_ue_id",
            "longitude",
            "latitude",
            "cell_lat",
            "cell_lon",
            "cell_id",
            "log_distance",
            "cell_rxpwr_dbm",
            "relative_bearing",
        ]
        for col in expected_columns:
            self.assertIn(col, processed_prediction_data.columns)

        # Verify if the relative bearing calculation is performed correctly
        self.assertFalse(processed_prediction_data["relative_bearing"].isnull().any())

    def test_training(self):
        # Test the training function
        maxiter = 5
        loss_vs_iters = self.mro.training(maxiter=maxiter)

        # Check if training produces a loss for each cell
        self.assertEqual(len(loss_vs_iters), len(self.topology))

    def test_predictions(self):
        self.mro.training(maxiter=5)
        predicted, full_prediction_df = self.mro.predictions()
        self.assertIn("pred_means", full_prediction_df.columns)
        self.assertIn("cell_id", full_prediction_df.columns)
        self.assertIn("loc_x", predicted.columns)
        self.assertIn("loc_y", predicted.columns)
