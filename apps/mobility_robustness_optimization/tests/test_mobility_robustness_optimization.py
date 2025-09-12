import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from apps.mobility_robustness_optimization.mobility_robustness_optimization import (
    BayesianDigitalTwin,
    _count_handovers,
    _count_rlf,
    calculate_mro_metric,
    reattach_columns,
)
from apps.mobility_robustness_optimization.simple_mro import SimpleMRO
from notebooks.radp_library import preprocess_ue_data


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
        self.training_data = pd.DataFrame({"ue_id": [0, 1], "tick": [0, 1], "loc_x": [5.0, 10.0], "loc_y": [0.0, 1.0]})

        self.mobility_model_params = {
            "param1": {"value": 10, "type": "int"},
            "param2": {"value": 20, "type": "float"},
        }

        # Mock Bayesian Digital Twin
        self.mock_bdt = MagicMock(spec=BayesianDigitalTwin)

        # Instantiate MRO object
        self.mro = SimpleMRO(
            self.mobility_model_params,
            self.dummy_topology,
            bdt={"cell_001": self.mock_bdt},
        )

    def test_count_rlf(self):
        result = _count_rlf(self.df)
        self.assertEqual(result, 2)

    def test_train_or_update_rf_twin(self):  # TODO: Implement AFTER PR
        pass

    def test_solve(self):  # TODO: Implement AFTER PR
        pass

    def test_training(self):
        mro = SimpleMRO(mobility_model_params={}, topology=self.dummy_topology)
        train_data = self.training_data.copy()
        train_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)

        # for different n_inter
        train_data = preprocess_ue_data(train_data, self.dummy_topology)

        if self.dummy_topology["cell_id"].dtype == int:
            self.dummy_topology["cell_id"] = self.dummy_topology["cell_id"].apply(lambda x: f"cell_{x}")
        if train_data["cell_id"].dtype == int:
            train_data["cell_id"] = train_data["cell_id"].apply(lambda x: f"cell_{x}")

        # Prepare the new data for training or updating
        prepared_data = mro._prepare_train_or_update_data(train_data)

        for n_iter in [5, 10, 20]:
            loss_vs_iter = mro._training(maxiter=n_iter, train_data=prepared_data)
            self.assertEqual(len(loss_vs_iter), len(self.dummy_topology["cell_id"]))
            self.assertEqual(loss_vs_iter[0].shape[0], n_iter)  # type: ignore

    def test_predictions(self):
        topology = self.dummy_topology.copy()
        topology["cell_id"] = ["cell_1", "cell_2"]
        mro = SimpleMRO(mobility_model_params=self.mobility_model_params, topology=topology)
        train_data = self.training_data.copy()
        train_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)

        train_data = preprocess_ue_data(train_data, self.dummy_topology)

        if self.dummy_topology["cell_id"].dtype == int:
            self.dummy_topology["cell_id"] = self.dummy_topology["cell_id"].apply(lambda x: f"cell_{x}")
        if train_data["cell_id"].dtype == int:
            train_data["cell_id"] = train_data["cell_id"].apply(lambda x: f"cell_{x}")

        # Prepare the new data for training or updating
        prepared_data = mro._prepare_train_or_update_data(train_data)

        # Train the models
        mro._training(20, prepared_data)

        # Perform predictions
        prediction_data = self.prediction_data.copy()
        prediction_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)
        predicted, full_prediction_df = mro._predictions(prediction_data)

        # Assertions
        self.assertEqual(predicted.shape, (2, 5))  # Check the shape of the predicted DataFrame
        self.assertEqual(full_prediction_df.shape, (4, 15))  # Check the shape of the full prediction DataFrame
        self.assertIn("pred_means", full_prediction_df.columns)  # Ensure predictions column exists
        self.assertTrue(all(full_prediction_df["cell_id"].isin(["cell_1", "cell_2"])))  # Check cell IDs

    def test_count_handovers(self):
        result = _count_handovers(self.df)
        self.assertEqual(result, 3)

    def test_reattach_columns(self):
        self.predicted_df = pd.DataFrame({"tick": [1, 2, 3, 4], "loc_x": [10, 20, 30, 40], "loc_y": [5, 6, 7, 8]})
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
        self.assertFalse("mock_ue_id" in result.columns)  # Ensure 'mock_ue_id' column is removed
        self.assertEqual(result.shape[0], self.predicted_df.shape[0])  # Ensure size matches predicted_df
        self.assertEqual(result.loc[0, "ue_id"], 100)
        self.assertEqual(result.loc[1, "ue_id"], 101)

        result = reattach_columns(self.predicted_df, full_prediction_df_no_match)
        self.assertTrue(result["ue_id"].isna().all())

    def test_calculate_mro_metric(self):
        result = calculate_mro_metric(self.df)
        self.assertEqual(result, 1.85)

    def test_prepare_train_or_update_data(self):
        # Create a sample DataFrame for input
        input_data = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_1", "cell_2"],
                "latitude": [45.0, 46.0, 45.1, 46.1],
                "longitude": [-73.0, -74.0, -73.1, -74.1],
                "cell_rxpwr_dbm": [-80, -75, -85, -70],
            }
        )

        # Add required columns to topology
        topology = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2"],
                "cell_lat": [45.0, 46.0],
                "cell_lon": [-73.0, -74.0],
                "cell_az_deg": [120, 240],
            }
        )

        # Instantiate the MRO object
        mro = SimpleMRO(mobility_model_params={}, topology=topology)

        # Call the method
        result = mro._prepare_train_or_update_data(input_data)

        # Assertions
        self.assertIsInstance(result, dict)  # Ensure the result is a dictionary
        self.assertEqual(len(result), 2)  # Ensure there are two keys (one for each cell_id)
        self.assertIn("cell_1", result)  # Ensure 'cell_1' is a key
        self.assertIn("cell_2", result)  # Ensure 'cell_2' is a key

        # Check the structure of the DataFrame for one cell
        cell_1_data = result["cell_1"]
        self.assertIsInstance(cell_1_data, pd.DataFrame)  # Ensure it's a DataFrame
        self.assertTrue(
            {"log_distance", "relative_bearing", "cell_rxpwr_dbm"}.issubset(cell_1_data.columns)
        )  # Check columns

    def test_preprocess_simulation_data(self):
        self.dummy_topology = pd.DataFrame({"cell_id": ["cell_1", "cell_2"]})
        self.optimizer = SimpleMRO({}, self.dummy_topology)
        self.optimizer.topology["cell_id"] = self.optimizer.topology["cell_id"].str.replace("cell_", "").astype(int)

        df = pd.DataFrame(
            {
                "mock_ue_id": [0],
                "tick": [1],
                "cell_id": ["cell_1"],
                "log_distance": [0.75],
                "pred_means": [-95],
                "rxpower_stddev_dbm": [1.0],
                "rxpower_dbm": [-90],
                "cell_rxpwr_dbm": [-92],
                "cell_carrier_freq_mhz": [1800],  # Required for SINR calculation
            }
        )

        result = self.optimizer._preprocess_simulation_data(df)

        self.assertIn("ue_id", result.columns)
        self.assertIn("distance_km", result.columns)
        self.assertIn("cell_rxpower_dbm", result.columns)

        self.assertNotIn("rxpower_stddev_dbm", result.columns)
        self.assertNotIn("rxpower_dbm", result.columns)
        self.assertNotIn("cell_rxpwr_dbm", result.columns)

        self.assertTrue(np.issubdtype(result["cell_id"].dtype, np.integer))

        self.assertIn("sinr_db", result.columns)
        self.assertTrue(np.isfinite(result["sinr_db"].iloc[0]))

    def test_add_sinr_column_basic(self):
        data = {
            "ue_id": [0, 0],
            "tick": [1, 1],
            "cell_id": [1, 2],
            "cell_rxpower_dbm": [-100, -95],
            "cell_carrier_freq_mhz": [2100.0, 2100.0],
        }
        df = pd.DataFrame(data)
        mro = SimpleMRO(mobility_model_params={}, topology=self.dummy_topology)
        result_df = mro._add_sinr_column(df)

        # Check that 'sinr_db' column was added
        self.assertIn("sinr_db", result_df.columns)

        # Check that SINR is finite and of correct length
        self.assertEqual(len(result_df["sinr_db"]), len(df))
        for val in result_df["sinr_db"]:
            self.assertTrue(np.isfinite(val))

    def test_update_no_sampling_when_small(self):
        # Arrange: small dataframe (<= 500 rows) with required columns
        topology = pd.DataFrame({"cell_id": ["cell_1"]})
        mro = SimpleMRO(
            mobility_model_params={}, topology=topology, bdt={"cell_1": MagicMock(spec=BayesianDigitalTwin)}
        )

        n_rows = 100
        df_small = pd.DataFrame(
            {
                "cell_id": ["cell_1"] * n_rows,
                "log_distance": [float(i) for i in range(n_rows)],
                "relative_bearing": [float((i * 3) % 180) for i in range(n_rows)],
                "cell_rxpwr_dbm": [-80.0 - (i % 10) for i in range(n_rows)],
            }
        )

        # Act & Assert: get_percell_data should NOT be called when <= 500 rows
        with patch(
            "apps.mobility_robustness_optimization.mobility_robustness_optimization.get_percell_data",
            side_effect=AssertionError("get_percell_data should not be called for <= 500 rows"),
        ):
            mro._update("cell_1", df_small)

    def test_update_sampling_and_dedup_when_large(self):
        # Arrange: large dataframe (> 500 rows) with a deliberate duplicate pair
        topology = pd.DataFrame({"cell_id": ["cell_1"]})
        mro = SimpleMRO(
            mobility_model_params={}, topology=topology, bdt={"cell_1": MagicMock(spec=BayesianDigitalTwin)}
        )

        n_rows = 502
        log_dists = [float(i) for i in range(n_rows)]
        bearings = [float((i * 7) % 180) for i in range(n_rows)]
        # Introduce a duplicate for first two rows
        log_dists[1] = log_dists[0]
        bearings[1] = bearings[0]

        df_large = pd.DataFrame(
            {
                "cell_id": ["cell_1"] * n_rows,
                "log_distance": log_dists,
                "relative_bearing": bearings,
                "cell_rxpwr_dbm": [-70.0 - (i % 20) for i in range(n_rows)],
            }
        )

        called = {"ok": False}

        def fake_get_percell_data(
            data_in, n_samples, choose_strongest_samples_percell=False, invalid_value=-500.0, seed=0
        ):
            # After dedup, there should be exactly one fewer row (501)
            assert len(data_in) == 501, f"Expected 501 rows after dedup, got {len(data_in)}"
            assert choose_strongest_samples_percell is True, "Expected strongest sampling for > 500 rows"
            assert n_samples == 500, f"Expected n_samples=500, got {n_samples}"
            called["ok"] = True
            # Return the expected structure: (List[pd.DataFrame], List[pd.DataFrame])
            return [data_in.iloc[:500].copy()], [pd.DataFrame()]

        # Act
        with patch(
            "apps.mobility_robustness_optimization.mobility_robustness_optimization.get_percell_data",
            side_effect=fake_get_percell_data,
        ):
            mro._update("cell_1", df_large)

        # Assert
        self.assertTrue(called["ok"], "Expected get_percell_data to be invoked with strongest sampling and dedup")
