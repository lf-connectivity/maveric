import unittest
import numpy as np
import pandas as pd

from radp.digital_twin.mobility.param_regression import ParameterRegression


class TestParameterRegression(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a mock DataFrame for testing the ParameterRegression class.

        Initializes a DataFrame with random 'mock_ue_id' and 'velocity' values,
        and creates an instance of ParameterRegression using this DataFrame.

        How to Run:
        ------------
        To run these tests, execute the following command in your terminal:
        ```
        python3 -m unittest radp/digital_twin/mobility/tests/test_parameter_regression.py
        ```
        """
        data = {
            "mock_ue_id": np.random.randint(0, 10, size=100),
            "velocity": np.random.uniform(0, 100, size=100),
        }
        self.df = pd.DataFrame(data)
        self.model = ParameterRegression(self.df)

    def test_model_function_output_shape(self) -> None:
        """
        Test that model_function returns an output with the correct shape.

        Asserts that the output shape of the model_function matches the expected shape,
        which is (2, len(self.model.v_t)).
        """
        alpha = 0.5
        x = np.array([self.model.v_t, self.model.theta_t])
        result = self.model.model_function(alpha, x)

        self.assertEqual(result.shape, (2, len(self.model.v_t)))

    def test_residual_vector_output_shape(self) -> None:
        """
        Test that residual_vector returns a 1D array with the correct shape.

        Asserts that the shape of the residuals returned from residual_vector matches
        the expected shape of (2 * len(self.model.v_t),).
        """
        alpha = 0.5
        t = np.array([self.model.v_t, self.model.theta_t])
        t_next = np.array([self.model.v_t_next, self.model.theta_t_next])
        result = self.model.residual_vector(alpha, t, t_next)

        self.assertEqual(result.shape, (2 * len(self.model.v_t),))

    def test_optimize_alpha_output(self) -> None:
        """
        Test that optimize_alpha returns an optimized alpha and a covariance matrix.

        Asserts that popt is an instance of np.ndarray and pcov is also an instance
        of np.ndarray. Additionally checks that popt contains a single optimized value.
        """
        alpha0 = [0.8]
        popt, pcov = self.model.optimize_alpha(alpha0)

        self.assertIsInstance(popt, np.ndarray)
        self.assertIsInstance(pcov, int)

        self.assertEqual(len(popt), 1)

    def test_velocity_mean_calculation(self) -> None:
        """
        Test that the velocity mean is calculated correctly.

        Asserts that the calculated mean of the velocity in the model matches the
        expected mean calculated from the DataFrame.
        """
        expected_mean = np.mean(self.df["velocity"].to_numpy())
        self.assertAlmostEqual(self.model.velocity_mean, expected_mean, places=5)

    def test_variance_calculation(self) -> None:
        """
        Test that the variance of velocity is calculated correctly.

        Asserts that the calculated variance of the velocity in the model matches the
        expected variance calculated from the DataFrame.
        """
        expected_variance = np.var(self.df["velocity"].to_numpy())
        self.assertAlmostEqual(self.model.variance, expected_variance, places=5)

    def test_theta_t_full_calculation(self) -> None:
        """
        Test that theta_t_full is calculated correctly with added noise.

        Asserts that the shape of the calculated theta_t_full matches the expected shape
        after applying the polynomial function and adding noise.
        """
        f = np.poly1d([8, 7, 5, 1])
        expected_theta_t_full = f(self.model.v_t_full) + 6 * np.random.normal(
            size=len(self.model.v_t_full)
        )

        self.assertEqual(self.model.theta_t_full.shape, expected_theta_t_full.shape)
