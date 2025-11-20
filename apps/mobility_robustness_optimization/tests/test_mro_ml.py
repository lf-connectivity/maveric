import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from apps.mobility_robustness_optimization.mro_ml import BayesianMRO
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin


class TestBayesianMRO(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mobility_model_params = {
            "param1": {"value": 10, "type": "int"},
            "param2": {"value": 20.5, "type": "float"},
        }

        self.topology = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                "cell_lat": [45.0, 46.0, 47.0],
                "cell_lon": [-73.0, -74.0, -75.0],
                "cell_carrier_freq_mhz": [2100, 2000, 1800],
                "cell_az_deg": [120, 240, 0],
            }
        )

        # Mock Bayesian Digital Twin
        self.mock_bdt = MagicMock(spec=BayesianDigitalTwin)
        self.bdt_dict = {"cell_1": self.mock_bdt, "cell_2": self.mock_bdt, "cell_3": self.mock_bdt}

    def test_init_default_parameters(self):
        """Test BayesianMRO initialization with default parameters."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology)

        self.assertEqual(bmro.model_type, "gpr")
        self.assertIn(bmro.device, ["cuda", "cpu"])
        self.assertEqual(bmro.mobility_model_params, self.mobility_model_params)
        pd.testing.assert_frame_equal(bmro.topology, self.topology)

    def test_init_with_custom_parameters(self):
        """Test BayesianMRO initialization with custom parameters."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict, model_type="xgboost")

        self.assertEqual(bmro.model_type, "xgboost")
        self.assertEqual(bmro.bayesian_digital_twins, self.bdt_dict)

    def test_init_model_gpr(self):
        """Test _init_model for Gaussian Process Regression."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="gpr")
        model = bmro._init_model()

        self.assertIsInstance(model, GaussianProcessRegressor)
        self.assertTrue(model.normalize_y)

    def test_init_model_xgboost_not_available(self):
        """Test _init_model raises ImportError when XGBoost is not available."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'xgboost'")):
            bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="xgboost")

            with self.assertRaises(ImportError) as context:
                bmro._init_model()

            self.assertIn("xgboost is required", str(context.exception))

    def test_expected_improvement_xgboost(self):
        """Test _expected_improvement for XGBoost model."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="xgboost")

        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.9, 0.7])

        X = np.array([[1.0, 2], [1.5, 3], [2.0, 4]])
        best_y = 0.75

        ei = bmro._expected_improvement(X, mock_model, best_y)

        expected = np.array([0.05, 0.15, -0.05])  # y_pred - best_y
        np.testing.assert_array_almost_equal(ei, expected, decimal=10)
        mock_model.predict.assert_called_once_with(X)

    def test_expected_improvement_gpr(self):
        """Test _expected_improvement for Gaussian Process model."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="gpr")

        # Mock GPR model
        mock_model = Mock()
        mu = np.array([0.8, 0.9, 0.7])
        std = np.array([0.1, 0.15, 0.05])
        mock_model.predict.return_value = (mu, std)

        X = np.array([[1.0, 2], [1.5, 3], [2.0, 4]])
        best_y = 0.75

        ei = bmro._expected_improvement(X, mock_model, best_y)

        self.assertEqual(len(ei), 3)
        self.assertTrue(all(isinstance(val, (int, float)) for val in ei))
        mock_model.predict.assert_called_once_with(X, return_std=True)

    def test_expected_improvement_gpr_zero_std(self):
        """Test _expected_improvement handles zero standard deviation."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="gpr")

        # Mock GPR model with zero std
        mock_model = Mock()
        mu = np.array([0.8])
        std = np.array([0.0])  # Zero std
        mock_model.predict.return_value = (mu, std)

        X = np.array([[1.0, 2]])
        best_y = 0.75

        ei = bmro._expected_improvement(X, mock_model, best_y)

        # Should handle zero std by setting minimum std to 1e-9
        self.assertTrue(np.isfinite(ei[0]))

    @patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data")
    @patch("apps.mobility_robustness_optimization.mro_ml.find_hyst_diff")
    @patch("apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt")
    @patch("apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric")
    def test_solve_no_bayesian_twins(self, mock_metric, mock_attachment, mock_hyst_diff, mock_get_data):
        """Test solve raises error when Bayesian Digital Twins are not trained."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology)

        with self.assertRaises(ValueError) as context:
            bmro.solve()

        self.assertIn("Bayesian Digital Twins are not trained", str(context.exception))

    @patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data")
    @patch("apps.mobility_robustness_optimization.mro_ml.find_hyst_diff")
    @patch("apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt")
    @patch("apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric")
    def test_solve_success(self, mock_metric, mock_attachment, mock_hyst_diff, mock_get_data):
        """Test successful solve execution."""
        # Setup mocks
        mock_data = pd.DataFrame({"lat": [45.0, 46.0], "lon": [-73.0, -74.0], "tick": [1, 2], "ue_id": [1, 2]})
        mock_get_data.return_value = mock_data
        mock_hyst_diff.return_value = 10.0
        mock_attachment.return_value = pd.DataFrame({"cell_id": ["A", "B"]})
        mock_metric.return_value = 0.85

        bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)

        # Mock _predictions and _preprocess_simulation_data methods
        bmro._predictions = Mock(return_value=(Mock(), mock_data))
        bmro._preprocess_simulation_data = Mock(return_value=mock_data)

        result = bmro.solve(n_epochs=2, init_samples=2)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        best_hyst, best_ttt = result
        self.assertIsInstance(best_hyst, float)
        self.assertIsInstance(best_ttt, int)

    def test_solve_integer_cell_id_conversion(self):
        """Test solve converts integer cell_id to string format."""
        # Create topology with integer cell_id
        topology_int = pd.DataFrame(
            {
                "cell_id": [1, 2, 3],
                "cell_lat": [45.0, 46.0, 47.0],
                "cell_lon": [-73.0, -74.0, -75.0],
                "cell_carrier_freq_mhz": [2100, 2000, 1800],
                "cell_az_deg": [120, 240, 0],
            }
        )

        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric:

            mock_data = pd.DataFrame({"lat": [45.0], "lon": [-73.0], "tick": [1], "ue_id": [1]})
            mock_get_data.return_value = mock_data
            mock_hyst_diff.return_value = 5.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["cell_1"]})
            mock_metric.return_value = 0.75

            bmro = BayesianMRO(self.mobility_model_params, topology_int, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), mock_data))
            bmro._preprocess_simulation_data = Mock(return_value=mock_data)

            bmro.solve(n_epochs=1, init_samples=1)

            # Verify cell_id conversion
            expected_cell_ids = ["cell_1", "cell_2", "cell_3"]
            self.assertTrue(all(cell_id in expected_cell_ids for cell_id in bmro.topology["cell_id"]))

    def test_solve_data_preprocessing_pipeline(self):
        """Test the data preprocessing pipeline in solve method."""
        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric:

            # Setup mock data with lat/lon columns
            original_data = pd.DataFrame({"lat": [45.0, 46.0], "lon": [-73.0, -74.0], "tick": [1, 2], "ue_id": [1, 2]})
            mock_get_data.return_value = original_data
            mock_hyst_diff.return_value = 8.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["A"]})
            mock_metric.return_value = 0.9

            bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), original_data))
            bmro._preprocess_simulation_data = Mock(return_value=original_data)

            bmro.solve(n_epochs=1, init_samples=1)

            # Verify data renaming happened
            mock_get_data.assert_called_once_with(self.mobility_model_params)
            # Verify preprocessing pipeline was called
            bmro._predictions.assert_called_once()
            bmro._preprocess_simulation_data.assert_called_once()

    def test_solve_bayesian_optimization_loop(self):
        """Test the Bayesian optimization loop functionality."""
        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric:

            mock_data = pd.DataFrame(
                {"latitude": [45.0, 46.0], "longitude": [-73.0, -74.0], "tick": [1, 2], "ue_id": [1, 2]}
            )
            mock_get_data.return_value = mock_data
            mock_hyst_diff.return_value = 5.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["A"]})
            # Return different metrics to test optimization
            mock_metric.side_effect = [0.7, 0.8, 0.9, 0.85]  # Initial + 3 epochs

            bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), mock_data))
            bmro._preprocess_simulation_data = Mock(return_value=mock_data)

            result = bmro.solve(n_epochs=3, init_samples=1)

            # Verify that optimization found the best result (0.9)
            self.assertIsInstance(result, tuple)
            # Verify the optimization loop ran the expected number of times
            self.assertEqual(mock_metric.call_count, 4)  # init_samples + n_epochs

    def test_solve_score_dataframe_creation(self):
        """Test that solve properly creates and populates score dataframe."""
        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric:

            mock_data = pd.DataFrame(
                {"latitude": [45.0, 46.0], "longitude": [-73.0, -74.0], "tick": [1, 2], "ue_id": [1, 2]}
            )
            mock_get_data.return_value = mock_data
            mock_hyst_diff.return_value = 10.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["A"]})
            mock_metric.return_value = 0.8

            bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), mock_data))
            bmro._preprocess_simulation_data = Mock(return_value=mock_data)

            bmro.solve(n_epochs=2, init_samples=2)

            # Verify score dataframe structure
            self.assertIsInstance(bmro.score, pd.DataFrame)
            self.assertListEqual(list(bmro.score.columns), ["hyst", "ttt", "score"])
            self.assertEqual(len(bmro.score), 4)  # init_samples + n_epochs

    def test_solve_parameter_ranges(self):
        """Test that solve uses correct parameter ranges."""
        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric, patch(
            "numpy.random.uniform"
        ) as mock_uniform, patch(
            "numpy.random.randint"
        ) as mock_randint:

            mock_data = pd.DataFrame(
                {
                    "latitude": [45.0, 46.0, 47.0, 48.0],
                    "longitude": [-73.0, -74.0, -75.0, -76.0],
                    "tick": [1, 2, 3, 4],
                    "ue_id": [1, 2, 3, 4],
                }
            )
            mock_get_data.return_value = mock_data
            mock_hyst_diff.return_value = 15.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["A"]})
            mock_metric.return_value = 0.8

            # Mock random values
            mock_uniform.return_value = 5.0
            mock_randint.return_value = 3

            bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), mock_data))
            bmro._preprocess_simulation_data = Mock(return_value=mock_data)

            bmro.solve(n_epochs=1, init_samples=1)

            # Verify ranges are used correctly
            # hyst_range should be [0, max_diff] = [0, 15.0]
            # ttt_range should be [2, num_ticks + 1] = [2, 5]
            mock_uniform.assert_called_with(0, 15.0, size=100)  # Called in optimization loop
            mock_randint.assert_called_with(2, 5, size=100)  # Called in optimization loop

    def test_solve_output_format(self):
        """Test that solve logs output in correct format."""
        with patch("apps.mobility_robustness_optimization.mro_ml.get_ue_data") as mock_get_data, patch(
            "apps.mobility_robustness_optimization.mro_ml.find_hyst_diff"
        ) as mock_hyst_diff, patch(
            "apps.mobility_robustness_optimization.mro_ml.perform_attachment_hyst_ttt"
        ) as mock_attachment, patch(
            "apps.mobility_robustness_optimization.mro_ml.calculate_mro_metric"
        ) as mock_metric:

            mock_data = pd.DataFrame({"latitude": [45.0], "longitude": [-73.0], "tick": [1], "ue_id": [1]})
            mock_get_data.return_value = mock_data
            mock_hyst_diff.return_value = 5.0
            mock_attachment.return_value = pd.DataFrame({"cell_id": ["A"]})
            mock_metric.return_value = 0.85

            bmro = BayesianMRO(self.mobility_model_params, self.topology, bdt=self.bdt_dict)
            bmro._predictions = Mock(return_value=(Mock(), mock_data))
            bmro._preprocess_simulation_data = Mock(return_value=mock_data)

            with patch.object(bmro.logger, "info") as mock_logger_info:
                result = bmro.solve(n_epochs=1, init_samples=1)
                self.assertIsNotNone(result)
                # Verify logger.info was called with correct format
                # The logger is called multiple times (header, separator, epochs, final result)
                # We need to check that the final call contains the optimized parameters
                self.assertGreater(mock_logger_info.call_count, 0)
                # Get the last call (which should be the optimized parameters)
                last_call_args = mock_logger_info.call_args[0][0]
                self.assertIn("Optimized Hyst:", last_call_args)
                self.assertIn("Optimized TTT:", last_call_args)

    def test_device_selection(self):
        """Test device selection logic."""
        # Test when CUDA is available
        with patch("torch.cuda.is_available", return_value=True):
            bmro = BayesianMRO(self.mobility_model_params, self.topology)
            self.assertEqual(bmro.device, "cuda")

        # Test when CUDA is not available
        with patch("torch.cuda.is_available", return_value=False):
            bmro = BayesianMRO(self.mobility_model_params, self.topology)
            self.assertEqual(bmro.device, "cpu")


class TestBayesianMROEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.mobility_model_params = {"param1": {"value": 10, "type": "int"}}
        self.topology = pd.DataFrame(
            {
                "cell_id": ["cell_1"],
                "cell_lat": [45.0],
                "cell_lon": [-73.0],
                "cell_carrier_freq_mhz": [2100],
                "cell_az_deg": [120],
            }
        )

    def test_empty_topology(self):
        """Test behavior with empty topology."""
        empty_topology = pd.DataFrame(columns=["cell_id", "cell_lat", "cell_lon"])
        bmro = BayesianMRO(self.mobility_model_params, empty_topology)
        self.assertTrue(bmro.topology.empty)

    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology, model_type="invalid")
        # Should default to GPR behavior
        model = bmro._init_model()
        self.assertIsInstance(model, GaussianProcessRegressor)

    def test_expected_improvement_empty_input(self):
        """Test _expected_improvement with empty input."""
        bmro = BayesianMRO(self.mobility_model_params, self.topology)
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([]), np.array([]))

        X = np.array([]).reshape(0, 2)
        result = bmro._expected_improvement(X, mock_model, 0.5)

        self.assertEqual(len(result), 0)
