import unittest
import numpy as np
import pandas as pd
from scipy import optimize
from radp.digital_twin.mobility.param_regression import (
    _initialize,
    _next,
    _residual_vector,
    optimize_alpha,
)


class TestParameterFunctions(unittest.TestCase):
    """
     Set up a mock DataFrame for testing the ParameterRegression class.

     Initializes a DataFrame with 2 'mock_ue_id' and 5 'velocity' values each,
     and creates an instance of ParameterRegression using this DataFrame.

     How to Run:
     ------------
     To run these tests, execute the following command in your terminal:

    python3 -m unittest radp/digital_twin/mobility/tests/test_parameter_regression.py

    """

    def setUp(self):
        # Setup the DataFrame for testing with 2 distinct ue_ids and 5 velocities each
        data = {
            "velocity": [1.2, 2.5, 3.0, 4.1, 5.7, 1.5, 2.0, 3.5, 4.2, 5.0],
            "mock_ue_id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
        self.df = pd.DataFrame(data)

        # Initialize the test data
        (
            self.t_array,
            self.t_next_array,
            self.velocity_mean,
            self.variance,
            self.rng,
        ) = _initialize(self.df, 42)
        self.alpha0 = 0.5

    def test_initialize(self):
        # Check if initialization returns correct data types and values
        self.assertIsInstance(self.t_array, np.ndarray)
        self.assertIsInstance(self.t_next_array, np.ndarray)

        # Check if the length of arrays matches the input data
        self.assertEqual(self.t_array.shape[1], len(self.df) - 1)
        self.assertEqual(self.t_next_array.shape[1], len(self.df) - 1)

    def test_next(self):
        # Test if the next function returns correct shape of results
        alpha = 0.5
        result = _next(
            alpha, self.t_array, 0, 1, self.rng
        )  # Passing arbitrary mean and variance

        # Check if result has the correct shape
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], self.t_array.shape[1])

        # Check if result is a numpy array
        self.assertIsInstance(result, np.ndarray)

    def test_residual_vector(self):
        # Test if the residuals are calculated correctly
        alpha = 0.5
        residuals = _residual_vector(
            alpha, self.t_array, self.t_next_array, 0, 1, self.rng
        )  # Passing arbitrary mean and variance

        # Check if residuals are a flat array
        self.assertEqual(residuals.shape[0], 2 * self.t_array.shape[1])
        self.assertIsInstance(residuals, np.ndarray)

    def test_optimize_alpha(self):
        # Test the optimization process for alpha
        popt, pcov = optimize_alpha(
            self.alpha0, self.t_array, self.t_next_array, 0, 1, self.rng
        )  # Passing arbitrary mean and variance

        # Check if popt is a numpy array and pcov is None (leastsq returns None for covariance)
        self.assertIsInstance(popt, np.ndarray)

        # Check that optimized alpha is within a reasonable range of the calculated alpha
        self.assertTrue(np.allclose(popt[0], 0.49999419))
