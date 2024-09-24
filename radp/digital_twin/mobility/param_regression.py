import numpy as np
from scipy import optimize
from typing import Tuple


class ParameterRegression:
    """
    A class designed for parameter regression analysis,
    tailored to model the movement dynamics of user equipment (UE) using velocities and angles.
    This class employs polynomial regression with a least squares optimization technique to estimate the parameter alpha.
    Alpha characterizes the dependency of future states on current states,
    optimizing it to minimize residuals and closely align predicted states with actual observed states.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing 'velocity' and 'mock_ue_id' columns which represent the velocities of the UEs and their respective identifiers.
        num_users (int): Number of unique users (UEs) determined by the count of unique 'mock_ue_id'.
        MAX_X (int), MAX_Y (int): Constants used as spatial boundaries or limits in computations, set to 100.
        USERS (np.ndarray): Array of user indices based on the number of users.
        velocity_mean (float): The mean of all velocity readings across the dataset.
        variance (float): The variance of the velocity readings, used in the regression model.
        rng (np.random.Generator): Random number generator with a predefined seed for reproducibility.
        v_t_full_data (np.ndarray): The velocity data converted to a numpy array for processing.
        f (np.poly1d): A polynomial function applied to the velocity data to simulate angle (theta) values.
        v_t, theta_t (np.ndarray): Current state velocities and angles.
        v_t_next, theta_t_next (np.ndarray): Subsequent state velocities and angles used for comparison and fitting.
        t_array, t_next_array (np.ndarray): Arrays combining the current and next state values for velocities and angles for use in optimization.
    """

    def __init__(self, df) -> None:
        """
        Initializes the ParameterRegression class with the dataset and precomputes constants and data arrays.

        Args:
            df (pd.DataFrame): The DataFrame containing 'velocity' and 'mock_ue_id' columns.
        """
        self.df = df
        self.v_t_full_data = df["velocity"].to_numpy()

        # CONSTANTS
        self.num_users = df["mock_ue_id"].nunique()
        self.MAX_X, self.MAX_Y = 100, 100
        self.USERS = np.arange(self.num_users)
        self.velocity_mean = np.mean(self.v_t_full_data)
        self.variance = np.var(self.v_t_full_data)
        self.rng = np.random.default_rng(seed=41)

        # Data
        self.f = np.poly1d([8, 7, 5, 1])
        self.v_t_full = (
            self.v_t_full_data
        )  # Replaces the velocity with UE generated Data.
        self.v_t = self.v_t_full[:-1]
        self.v_t_next = self.v_t_full[1:]

        self.theta_t_full = self.f(self.v_t_full) + 6 * np.random.normal(
            size=len(self.v_t_full)
        )
        self.theta_t = self.theta_t_full[:-1]
        self.theta_t_next = self.theta_t_full[1:]

        self.t_array = np.array((self.v_t, self.theta_t))
        self.t_next_array = np.array((self.v_t_next, self.theta_t_next))

    def model_function(self, alpha: float, x: np.ndarray) -> np.ndarray:
        """
        Computes the next velocity and angle values based on the current ones, using a given alpha.

        Args:
            alpha (float): The parameter used for regression.
            x (np.ndarray): A 2D array where the first row is velocities (v_t) and the second row is angles (theta_t).

        Returns:
            np.ndarray: A 2D array with predicted next velocities and angles.
        """
        v_t, theta_t = x[0], x[1]
        alpha2 = 1.0 - alpha
        alpha3 = np.sqrt(1.0 - alpha * alpha) * self.variance
        v_t_next = (
            alpha * v_t + alpha2 * self.velocity_mean + alpha3 * np.random.normal()
        )
        angle_mean = theta_t  # Simplified model without margin correction
        theta_t_next = (
            alpha * theta_t + alpha2 * angle_mean + alpha3 * np.random.normal()
        )
        return np.array([v_t_next, theta_t_next])

    def residual_vector(
        self, alpha: float, t: np.ndarray, t_next: np.ndarray
    ) -> np.ndarray:
        """
        Computes the residuals between the predicted next state and the actual next state.

        Args:
            alpha (float): The parameter being optimized.
            t (np.ndarray): The current state (velocities and angles).
            t_next (np.ndarray): The next state to compare against (velocities and angles).

        Returns:
            np.ndarray: A flattened array of residuals (differences) between predicted and actual next states.
        """
        return (self.model_function(alpha, t) - t_next).flatten()

    def optimize_alpha(self, alpha0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the alpha parameter using least-squares fitting to minimize the residuals between the predicted and actual states.

        Args:
            alpha0 (float): The initial guess for alpha.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - popt: Optimized alpha value.
                - pcov: Covariance of the optimized parameter.
        """
        popt, pcov = optimize.leastsq(
            self.residual_vector, alpha0, args=(self.t_array, self.t_next_array)
        )
        return popt, pcov
