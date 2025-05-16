from typing import Optional

import numpy as np
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from notebooks.radp_library import get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class ReinforcedMRO(MobilityRobustnessOptimization):
    """
    Solves the mobility robustness optimization problem using reinforcement learning (PPO).
    """

    def __init__(
        self,
        mobility_model_params: dict[str, dict],
        topology: pd.DataFrame,
        bdt: Optional[dict[str, BayesianDigitalTwin]] = None,
    ):
        super().__init__(mobility_model_params, topology, bdt)

    def solve(self, total_timesteps=100):
        """
        Trains a PPO agent to optimize hysteresis and TTT values.
        """
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

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
        print(f"\nOptimized Hyst: {hyst},\nOptimized TTT: {ttt}")
        return hyst, ttt


class ReinforcedMROEnv(Env):
    def __init__(self, df, rlf_threshold, hyst_range, ttt_range):
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

        self.state = np.array([0.0])
        self.current_step = 0
        self.max_steps = 20
        self.episode_num = 1
        self.episode_reward = 0.0

    def step(self, action):
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

        print(
            f"Episode: {self.episode_num}, Timestep: {self.current_step}, "
            f"Hyst: {hyst:.6f}, TTT: {ttt}, Reward: {reward:.6f}, Done: {terminated}"
        )

        if terminated:
            avg_reward = self.episode_reward / self.max_steps
            print(f"Episode {self.episode_num} average reward: {avg_reward:.6f}\n")
            self.episode_num += 1
            self.episode_reward = 0.0

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self.state = np.array([0.0])
        self.current_step = 0
        return self.state, {}

    def render(self):
        print(f"Current State: {self.state}, Current Step: {self.current_step}")
