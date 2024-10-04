import numpy as np
import pandas as pd
from scipy import optimize
from typing import Tuple, List, Any


def initialize(
    df: pd.DataFrame, seed: int
) -> Tuple[np.ndarray, np.ndarray, float, float, np.random.Generator]:
    """
    Initializes and preprocesses data from a DataFrame for mobility modeling.

    Args:
    df (pd.DataFrame): DataFrame containing 'velocity' and 'mock_ue_id' columns.

    """
    v_t_full_data = df["velocity"].to_numpy()

    # CONSTANTS
    num_users = df["mock_ue_id"].nunique()
    rng = np.random.default_rng(seed)

    # Data
    f = np.poly1d([8, 7, 5, 1])
    v_t_full = v_t_full_data
    v_t = v_t_full[:-1]
    v_t_next = v_t_full[1:]

    theta_t_full = f(v_t_full) + 6 * rng.normal(size=len(v_t_full))
    theta_t = theta_t_full[:-1]
    theta_t_next = theta_t_full[1:]

    t_array = np.array((v_t, theta_t))
    t_next_array = np.array((v_t_next, theta_t_next))

    velocity_mean = np.mean(v_t_full_data)
    variance = np.var(v_t_full_data)

    return t_array, t_next_array, velocity_mean, variance, rng


def _next(
    alpha: float,
    x: np.ndarray,
    velocity_mean: float,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Computes the next state of velocity and angle using the current state and alpha.

    Args:
    alpha (float): The regression parameter.
    x (np.ndarray): Current state vector [velocity, angle].
    velocity_mean (float): Mean velocity used for normalization.
    variance (float): Variance of the velocity.
    rng (np.random.Generator): Random number generator for noise.

    """
    v_t, theta_t = x[0], x[1]
    alpha2 = 1.0 - alpha
    alpha3 = np.sqrt(1.0 - alpha * alpha) * variance
    v_t_next = alpha * v_t + alpha2 * velocity_mean + alpha3 * rng.normal()
    angle_mean = theta_t  # Simplified model without margin correction
    theta_t_next = alpha * theta_t + alpha2 * angle_mean + alpha3 * rng.normal()
    return np.array([v_t_next, theta_t_next])


def _residual_vector(
    alpha: float,
    t: np.ndarray,
    t_next: np.ndarray,
    velocity_mean: float,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Computes residuals for optimization by comparing predicted and actual next states.

    Args:
    alpha (float): The regression parameter.
    t (np.ndarray): Current state array [velocity, angle].
    t_next (np.ndarray): Actual next state array [next_velocity, next_angle].
    velocity_mean (float): Mean velocity used for normalization.
    variance (float): Variance of the velocity.
    rng (np.random.Generator): Random number generator for noise.

    """
    predicted = _next(alpha, t, velocity_mean, variance, rng)
    return (predicted - t_next).flatten()


def optimize_alpha(
    alpha0: float,
    t_array: np.ndarray,
    t_next_array: np.ndarray,
    velocity_mean: float,
    variance: float,
    rng: np.random.Generator,
) -> Tuple[float, Any]:
    """
    Optimizes alpha using least squares based on the provided mobility data.

    Args:
    alpha0 (float): Initial guess for alpha.
    t_array (np.ndarray): Array of current state values [velocity, angle].
    t_next_array (np.ndarray): Array of next state values [next_velocity, next_angle].
    velocity_mean (float): Mean of velocity.
    variance (float): Variance of velocity.
    rng (np.random.Generator): Random number generator for noise.

    """
    popt, pcov = optimize.leastsq(
        _residual_vector,
        alpha0,
        args=(t_array, t_next_array, velocity_mean, variance, rng),
    )
    return popt, pcov
