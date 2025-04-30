import unittest
import pandas as pd
import numpy as np
from apps.mobility_robustness_optimization.mobility_robustness_optimization import (
    MobilityRobustnessOptimization as MRO,
    BayesianDigitalTwin,
    NormMethod,
    reattach_columns,
    calculate_mro_metric,
    _count_handovers,
)
from apps.mobility_robustness_optimization.simple_mro import SimpleMRO
from unittest.mock import MagicMock


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def setUp(self):
        data = {
            "ue_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
            "cell_id": ["A", "RLF", "RLF", "B", "A", "A", "X", "Y", "Z"],
            "tick": [1, 2, 3, 4, 1, 2, 1, 2, 3],
        }
        self.df = pd.DataFrame(data)
        self.dummy_topology = pd.DataFrame(
            {
                "cell_id": ["cell_001", "cell_002"],
                "cell_lat": [45.0, 46.0],
                "cell_lon": [-73.0, -74.0],
                "cell_carrier_freq_mhz": [2100, 2000],
                "cell_az_deg": [120, 240],
            }
        )

        # Mocking additional data attributes used in the method
        self.prediction_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "loc_x": [10.0, 20.0],
                "loc_y": [5.0, 6.0],
            }
        )
        self.simulation_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "lon": [15.0, 25.0],
                "lat": [10.0, 11.0],
                "cell_id": [1, 2],
                "cell_lat": [45.0, 46.0],
                "cell_lon": [-73.0, -74.0],
                "cell_carrier_freq_mhz": [
                    1800,
                    2100,
                ],
            }
        )
        self.update_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "loc_x": [30.0, 40.0],
                "loc_y": [12.0, 13.0],
            }
        )
        self.training_data = pd.DataFrame(
            {"ue_id": [0, 1], "tick": [0, 1], "loc_x": [5.0, 10.0], "loc_y": [0.0, 1.0]}
        )

        self.mobility_params = {
            "param1": {"value": 10, "type": "int"},
            "param2": {"value": 20, "type": "float"},
        }

        # Mock Bayesian Digital Twin
        self.mock_bdt = MagicMock(spec=BayesianDigitalTwin)

        # Instantiate MRO object
        self.mro = SimpleMRO(
            self.mobility_params, self.dummy_topology, bdt={"cell_001": self.mock_bdt}
        )
        self.mro.training_data = self.training_data
        self.mro.prediction_data = self.prediction_data
        self.mro.update_data = self.update_data
        self.mro.simulation_data = self.simulation_data

    def test_update(self):  # TODO: Implement AFTER PR
        pass

    def test_solve(self):  # TODO: Implement AFTER PR
        pass


    def test_training(self):
        mro = SimpleMRO(mobility_params={}, topology=self.dummy_topology)
        train_data = self.training_data.copy()
        train_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        # n_iter = 5
        # for different n_inter
        for n_iter in [5, 10, 20]:
            loss_vs_iter = mro._training(maxiter=n_iter, train_data=train_data)
            self.assertEqual(len(loss_vs_iter), len(self.dummy_topology["cell_id"]))
            self.assertEqual(loss_vs_iter[0].shape[0], n_iter)

    def test_predictions(self):
        # without _training() --> model not available --> empty df response
        mro = SimpleMRO(mobility_params={}, topology=self.dummy_topology)
        prediction_data = self.prediction_data.copy()
        mro.prediction_data = prediction_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        predicted, full_prediction_df = mro._predictions(pred_data=prediction_data)
        self.assertTrue(predicted.empty)
        self.assertTrue(full_prediction_df.empty)

        # with _training()
        topology = self.dummy_topology.copy()
        topology["cell_id"] = ["cell_1", "cell_2"]
        mro = SimpleMRO(mobility_params=self.mobility_params, topology=topology)
        train_data = self.training_data.copy()
        train_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        mro._training(20, train_data)  # needed, otherwise model won't be available
        prediction_data = self.prediction_data.copy()
        prediction_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        predicted, full_prediction_df = mro._predictions(prediction_data)
        self.assertEqual(predicted.shape, (2, 5))
        self.assertEqual(full_prediction_df.shape, (4, 15))

    def test_prepare_all_UEs_from_all_cells_df(self):
        result = self.mro._prepare_all_UEs_from_all_cells_df()
        self.assertEqual(result.shape[0], 2 * 2)  # 2 UEs x 2 cells

    def test_calculate_received_power(self):
        dummy_distance = 1
        dummy_freq = 1800
        expected_power = -74.55545010206612
        power = self.mro._calculate_received_power(
            distance_km=dummy_distance, frequency_mhz=dummy_freq
        )
        self.assertEqual(expected_power, power)

    def test_preprocess_ue_topology_data(self):
        result = self.mro._prepare_all_UEs_from_all_cells_df()
        # Ensure that the resulting dataframe has the correct number of columns
        self.assertIn("ue_id", result.columns)
        self.assertIn("cell_id", result.columns)

        # Check if the combined dataframe has the correct number of rows (Cartesian product of UEs and cells)
        self.assertEqual(
            result.shape[0], len(self.update_data) * len(self.dummy_topology)
        )

        # Ensure that the combined dataframe has the expected values
        self.assertTrue(all(result["ue_id"].isin(self.update_data["ue_id"])))

    def test_preprocess_ue_training_data(self):
        # fmt: off
        expected_columns = ["ue_id","tick", "latitude", "longitude", "cell_id", "cell_lat", "cell_lon",
                            "cell_carrier_freq_mhz", "cell_az_deg", "log_distance", "cell_rxpwr_dbm", "relative_bearing",]
        # fmt: on
        self.mro.training_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        training_data = self.mro._preprocess_ue_training_data()
        self.assertIsInstance(training_data, dict)
        self.assertEqual(len(training_data), len(self.dummy_topology))
        for df in training_data.values():
            self.assertListEqual(list(df.columns), expected_columns)
            self.assertTrue(all(df["ue_id"].isin(self.training_data["ue_id"])))

    def test_preprocess_ue_update_data(self):
        # fmt: off
        expected_columns = ["ue_id","tick", "latitude", "longitude", "cell_id", "cell_lat", "cell_lon",
                            "cell_carrier_freq_mhz", "cell_az_deg", "log_distance", "cell_rxpwr_dbm", "relative_bearing",]
        # fmt: on
        self.mro.update_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        update_data = self.mro._preprocess_ue_update_data()
        self.assertIsInstance(update_data, dict)
        self.assertEqual(len(update_data), len(self.dummy_topology))
        for df in update_data.values():
            self.assertListEqual(list(df.columns), expected_columns)
            self.assertTrue(all(df["ue_id"].isin(self.update_data["ue_id"])))

    def test_preprocess_prediction_data(self):
        # fmt: off
        expected_columns = ["ue_id","tick", "latitude", "longitude", "cell_id", "cell_lat", "cell_lon",
                            "cell_carrier_freq_mhz", "cell_az_deg", "log_distance", "cell_rxpwr_dbm", "relative_bearing",]
        # fmt: on
        self.mro.prediction_data.rename(
            columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True
        )
        data = self.mro._preprocess_prediction_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(
            len(data), len(self.dummy_topology) * len(self.mro.prediction_data["ue_id"])
        )
        self.assertListEqual(list(data.columns), expected_columns)
        self.assertTrue(all(data["ue_id"].isin(self.prediction_data["ue_id"])))

    def test_count_handovers(self):
        result = _count_handovers(self.df)
        self.assertEqual(result, 3)

    def test_reattach_columns(self):
        self.predicted_df = pd.DataFrame(
            {"tick": [1, 2, 3, 4], "loc_x": [10, 20, 30, 40], "loc_y": [5, 6, 7, 8]}
        )
        self.full_prediction_df = pd.DataFrame(
            {
                "mock_ue_id": [100, 101, 102, 103],
                "tick": [1, 2, 3, 4],
                "loc_x": [10, 20, 30, 40],
                "loc_y": [5, 6, 7, 8],
            }
        )
        full_prediction_df_no_match = pd.DataFrame(
            {
                "mock_ue_id": [200, 201, 202, 203],
                "tick": [1, 2, 3, 4],
                "loc_x": [50, 60, 70, 80],  # No matching loc_x values
                "loc_y": [9, 10, 11, 12],  # No matching loc_y values
            }
        )

        result = reattach_columns(self.predicted_df, self.full_prediction_df)

        self.assertTrue("ue_id" in result.columns)  # Ensure 'ue_id' column exists
        self.assertFalse(
            "mock_ue_id" in result.columns
        )  # Ensure 'mock_ue_id' column is removed
        self.assertEqual(
            result.shape[0], self.predicted_df.shape[0]
        )  # Ensure size matches predicted_df
        self.assertEqual(result.loc[0, "ue_id"], 100)
        self.assertEqual(result.loc[1, "ue_id"], 101)

        result = reattach_columns(self.predicted_df, full_prediction_df_no_match)
        self.assertTrue(result["ue_id"].isna().all())

    def test_calculate_mro_metric(self):
        result = calculate_mro_metric(self.df)
        self.assertEqual(result, 1.85)