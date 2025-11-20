"""Reinforcement Learning-based Mobility Robustness Optimization.

This module implements a reinforcement learning approach to optimize Mobility Robustness
Optimization (MRO) parameters using Proximal Policy Optimization (PPO). It provides
an alternative to traditional optimization methods by framing MRO as an RL problem.

The module contains:
    - ReinforcedMRO: Main optimization class using PPO for parameter tuning
    - ReinforcedMROEnv: Gymnasium environment for training RL agents on MRO tasks

The RL approach learns optimal hysteresis and Time-to-Trigger (TTT) values by
interacting with a simulation environment and receiving rewards based on the
MRO metric, which balances handover performance and radio link failures.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from notebooks.radp_library import find_sim_boundary, get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class ReinforcedMRO(MobilityRobustnessOptimization):
    """Mobility Robustness Optimization using Proximal Policy Optimization (PPO).

    This class implements a reinforcement learning approach to optimize handover
    parameters (hysteresis and TTT) in cellular networks. It uses the PPO algorithm
    from Stable-Baselines3 to learn optimal parameter values through interaction
    with a custom Gymnasium environment.

    The RL agent explores the hysteresis-TTT parameter space and receives rewards
    based on the MRO metric, which evaluates handover performance including
    successful handovers, radio link failures, and ping-pongs.

    Attributes:
        logger (logging.Logger): Logger instance for tracking optimization progress.
        simulation_data (pd.DataFrame): Preprocessed simulation data for evaluation.
    """

    def __init__(
        self,
        mobility_model_params: dict[str, dict],
        topology: pd.DataFrame,
        new_data: Optional[pd.DataFrame] = None,
        bdt: Optional[dict[str, BayesianDigitalTwin]] = None,
    ):
        """Initialize the ReinforcedMRO optimizer.

        Args:
            mobility_model_params: Dictionary containing mobility model configuration
                including UE track generation parameters and simulation boundaries.
            topology: DataFrame containing network topology with cell locations,
                azimuth angles, and carrier frequencies.
            new_data: Optional DataFrame containing pre-existing UE measurement data.
                If None, new data will be generated using mobility models.
            bdt: Optional dictionary of pre-trained Bayesian Digital Twin models
                indexed by cell identifiers. Required for making signal predictions.
        """
        super().__init__(mobility_model_params, topology, new_data, bdt)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def solve(self, total_timesteps=100):
        """Train a PPO agent to optimize hysteresis and TTT parameters.

        This method sets up and trains a Proximal Policy Optimization (PPO) agent
        to learn optimal handover parameters through interaction with the MRO
        environment. The training process involves:
        1. Generating or loading UE mobility simulation data
        2. Making signal strength predictions using Bayesian Digital Twins
        3. Determining valid parameter ranges based on simulation data
        4. Training a PPO agent using the MlpPolicy network
        5. Extracting optimal parameters from the trained agent

        Args:
            total_timesteps: Total number of environment steps for training the
                PPO agent. More timesteps generally lead to better convergence
                but increase training time. Defaults to 100.

        Returns:
            tuple: A tuple (hyst, ttt) containing:
                - hyst (float): Optimized hysteresis value in dB
                - ttt (int): Optimized Time-to-Trigger value in ticks

        Raises:
            ValueError: If Bayesian Digital Twins are not trained before calling solve.

        Notes:
            - Uses GPU acceleration (CUDA) if available, otherwise falls back to CPU
            - PPO hyperparameters: n_steps=64, batch_size=64
            - The agent uses a deterministic policy for final parameter extraction
            - Training progress is logged at INFO level
        """
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Determine simulation boundaries
        bounds = find_sim_boundary(self.topology, self.new_data)
        self.mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)

        # Load and prepare simulation data
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        predictions, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = self._preprocess_simulation_data(full_prediction_df)

        # Define parameter ranges
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Create and vectorize RL environment
        env = DummyVecEnv([lambda: ReinforcedMROEnv(self.simulation_data, RLF_THRESHOLD, hyst_range, ttt_range)])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # PPO agent
        model = PPO("MlpPolicy", env, verbose=2, n_steps=64, batch_size=64, device=device)
        model.learn(total_timesteps)

        # Predict optimal action using trained model
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        # Ensure ttt is an integer
        hyst, ttt = action[0]
        ttt = int(round(ttt))
        self.logger.info(f"\nOptimized Hyst: {hyst},\nOptimized TTT: {ttt}")
        return hyst, ttt


class ReinforcedMROEnv(Env):
    """Gymnasium environment for Mobility Robustness Optimization using RL.

    This custom Gymnasium environment models the MRO parameter optimization problem
    as a reinforcement learning task. The agent selects hysteresis and TTT values
    (actions), and receives rewards based on the resulting MRO metric which evaluates
    handover performance.

    The environment uses a continuous action space for both hysteresis (dB) and
    TTT (ticks), with parameter ranges automatically determined from simulation data.
    Episodes terminate after a fixed number of steps, allowing the agent to explore
    multiple parameter configurations.

    Attributes:
        df (pd.DataFrame): Preprocessed simulation data containing UE measurements
            and signal strength predictions.
        rlf_threshold (float): Radio Link Failure threshold in dB.
        hyst_range (List[float]): Valid range for hysteresis values [min, max].
        ttt_range (List[float]): Valid range for TTT values [min, max].
        action_space (Box): Continuous action space for [hysteresis, TTT].
        observation_space (Box): State space containing normalized reward.
        state (np.ndarray): Current environment state (previous reward).
        current_step (int): Current timestep within the episode.
        max_steps (int): Maximum steps per episode (default: 20).
        episode_num (int): Current episode number for logging.
        episode_reward (float): Cumulative reward for current episode.
        logger (logging.Logger): Logger for tracking training progress.
    """

    def __init__(self, df, rlf_threshold, hyst_range, ttt_range):
        """Initialize the MRO reinforcement learning environment.

        Args:
            df: Preprocessed simulation DataFrame containing UE tracks with
                predicted signal strengths and cell identifiers.
            rlf_threshold: Radio Link Failure threshold in dB below which
                connections fail.
            hyst_range: List containing [min_hyst, max_hyst] bounds for
                hysteresis parameter in dB.
            ttt_range: List containing [min_ttt, max_ttt] bounds for
                Time-to-Trigger parameter in ticks.
        """
        super().__init__()
        self.df = df
        self.rlf_threshold = rlf_threshold
        self.hyst_range = hyst_range
        self.ttt_range = ttt_range

        self.action_space = Box(
            low=np.array([hyst_range[0], ttt_range[0]]),
            high=np.array([hyst_range[1], ttt_range[1]]),
            dtype=np.float64,
        )
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)
        self.logger = logging.getLogger(__name__)

        self.state = np.array([0.0])
        self.current_step = 0
        self.max_steps = 20
        self.episode_num = 1
        self.episode_reward = 0.0

    def step(self, action):
        """Execute one environment step with the given action.

        Applies the selected hysteresis and TTT parameters to the simulation data,
        performs cell attachment decisions, calculates the MRO metric, and returns
        the reward. The MRO metric accounts for successful handovers, radio link
        failures, and ping-pong handovers.

        Args:
            action: Array containing [hysteresis, ttt] values selected by the agent.
                Hysteresis is in dB, TTT is in ticks (will be rounded to integer).

        Returns:
            tuple: A tuple (observation, reward, terminated, truncated, info) containing:
                - observation (np.ndarray): New state (current reward value)
                - reward (float): MRO metric value (higher is better)
                - terminated (bool): True if episode reached max_steps
                - truncated (bool): Always False (can be customized for early stopping)
                - info (dict): Empty dictionary for additional information

        Notes:
            - TTT value is automatically rounded to the nearest integer
            - Logs episode progress including parameters, rewards, and episode statistics
            - Resets episode tracking when terminated
        """
        hyst, ttt = action
        ttt = int(round(ttt))

        attached_df = perform_attachment_hyst_ttt(self.df, hyst, ttt, self.rlf_threshold)
        mro_metric = calculate_mro_metric(attached_df)

        reward = mro_metric
        self.episode_reward += reward
        self.state = np.array([reward])
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False  # Can be customized if needed

        self.logger.info(
            f"Episode: {self.episode_num}, Timestep: {self.current_step}, "
            f"Hyst: {hyst:.6f}, TTT: {ttt}, Reward: {reward:.6f}, Done: {terminated}"
        )

        if terminated:
            avg_reward = self.episode_reward / self.max_steps
            self.logger.info(f"Episode {self.episode_num} average reward: {avg_reward:.6f}\n")
            self.episode_num += 1
            self.episode_reward = 0.0

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state for a new episode.

        Reinitializes the environment state and step counter to start fresh training
        episode. This method is called automatically by the RL training loop at the
        beginning of each episode.

        Args:
            seed: Optional random seed for reproducibility. Currently unused but
                included for Gymnasium API compatibility.
            options: Optional dictionary of environment configuration options.
                Currently unused but included for Gymnasium API compatibility.

        Returns:
            tuple: A tuple (observation, info) containing:
                - observation (np.ndarray): Initial state [0.0]
                - info (dict): Empty dictionary for additional reset information
        """
        self.state = np.array([0.0])
        self.current_step = 0
        return self.state, {}

    def render(self):
        """Render the current environment state for visualization.

        Logs the current state and timestep to the console. This method is primarily
        used for debugging and monitoring the environment during development.

        Returns:
            None. Outputs state information to the logger at INFO level.
        """
        self.logger.info(f"Current State: {self.state}, Current Step: {self.current_step}")
