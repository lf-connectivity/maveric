from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt, find_hyst_diff
from notebooks.radp_library import get_ue_data
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box
import numpy as np


class ReinforcedMRO(MobilityRobustnessOptimization):
    """
    Solves the mobility robustness optimization problem using reinforcement learning (PPO).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self):
        """
        Trains a PPO agent to optimize hysteresis and TTT values.
        """
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Load and prepare simulation data
        self.simulation_data = get_ue_data(self.mobility_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})
        predictions, full_prediction_df = self._predictions(self.simulation_data)
        df = self._preprocess_simulation_data(full_prediction_df)

        # Define parameter ranges
        max_diff = find_hyst_diff(df)
        num_ticks = df["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Create and vectorize RL environment
        env = DummyVecEnv([lambda: ReinforcedMROEnv(df, RLF_THRESHOLD, hyst_range, ttt_range)])

        # PPO agent
        model = PPO("MlpPolicy", env, verbose=2, n_steps=64, batch_size=64)
        model.learn(total_timesteps=1000)

        # Predict optimal action using trained model
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        # Ensure ttt is an integer
        hyst, ttt = action[0]
        ttt = int(round(ttt))

        return hyst, ttt


class ReinforcedMROEnv(Env):
    def __init__(self, df, rlf_threshold, hyst_range, ttt_range):
        super().__init__()
        self.df = df
        self.rlf_threshold = rlf_threshold
        self.hyst_range = hyst_range
        self.ttt_range = ttt_range

        self.action_space = Box(low=np.array([hyst_range[0], ttt_range[0]]),
                                high=np.array([hyst_range[1], ttt_range[1]]),
                                dtype=np.float64)
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

        done = self.current_step >= self.max_steps

        print(f"Episode: {self.episode_num}, Timestep: {self.current_step}, "
              f"Hyst: {hyst:.6f}, TTT: {ttt}, Reward: {reward:.6f}, Done: {done}")

        if done:
            avg_reward = self.episode_reward / self.max_steps
            print(f"Episode {self.episode_num} average reward: {avg_reward:.6f}\n")
            self.episode_num += 1
            self.episode_reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0])
        self.current_step = 0
        return self.state

    def render(self, mode="human"):
        print(f"Current State: {self.state}, Current Step: {self.current_step}")