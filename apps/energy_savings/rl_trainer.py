# rl_trainer.py
"""
This module is designed to initialize and train an RL agent for energy-saving applications.
The env is set up using site configuration data, user equipment (UE) data, and a Bayesian Digital Twin (BDT) model.
The RL agent is trained using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3.
Logging is configured for monitoring the training process, and checkpoints are saved periodically.
"""

import logging
import os
import sys
from typing import List

import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
sys.path.insert(0, PROJECT_ROOT)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from apps.energy_savings.rl_energy_saving_env import TickAwareEnergyEnv
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
DEFAULT_HTX = 25.0
DEFAULT_HRX = 1.5


def run_rl_training(
    bdt_model_path: str,
    base_ue_data_dir: str,
    training_days: List[int],
    topology_path: str,
    config_path: str,
    rl_model_path: str,
    log_dir: str,
    total_timesteps: int,
):
    """
    The RL environment is initialized and the RL agent is trained for energy-saving purposes by this function.

    Parameters
    ----------
    bdt_model_path : str
        The path to the Bayesian Digital Twin model pickle file is specified.
    base_ue_data_dir : str
        The base directory containing user equipment (UE) data is specified.
    training_days : List[int]
        The list of days to be used for training is specified.
    topology_path : str
        The path to the topology CSV file is specified.
    config_path : str
        The path to the configuration CSV file is specified.
    rl_model_path : str
        The path where the trained RL model will be saved is specified.
    log_dir : str
        The directory for logging and checkpoint files is specified.
    total_timesteps : int
        The total number of timesteps for RL training is specified.
    """
    logger.info("--- Starting RL Energy Saver Training ---")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.dirname(rl_model_path):
        os.makedirs(os.path.dirname(rl_model_path), exist_ok=True)

    try:
        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        site_config_df = pd.merge(topology_df, config_df, on="cell_id", how="left")
        site_config_df["cell_el_deg"].fillna(TILT_SET[len(TILT_SET) // 2], inplace=True)

        required_cols = {"hTx": DEFAULT_HTX, "hRx": DEFAULT_HRX}
        for col, val in required_cols.items():
            if col not in site_config_df.columns:
                site_config_df[col] = val

        bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(bdt_model_map)} cells.")

        ue_data_per_tick = {tick: [] for tick in range(24)}
        for day in training_days:
            day_dir = os.path.join(base_ue_data_dir, f"Day_{day}", "ue_data_gym_ready")
            if not os.path.isdir(day_dir):
                continue
            for tick in range(24):
                filepath = os.path.join(day_dir, f"generated_ue_data_for_cco_{tick}.csv")
                if os.path.exists(filepath):
                    ue_data_per_tick[tick].append(pd.read_csv(filepath))

        if not any(ue_data_per_tick.values()):
            raise FileNotFoundError("No UE data loaded.")

    except Exception as e:
        logger.exception(f"Failed during data loading: {e}")
        return

    try:
        reward_weights = {"cco_score": 0.2, "load_balance_score": 0.1, "energy_saving_score": 1.0}
        env = TickAwareEnergyEnv(
            bayesian_digital_twins=bdt_model_map,
            site_config_df=site_config_df,
            ue_data_per_tick=ue_data_per_tick,
            tilt_set=TILT_SET,
            reward_weights=reward_weights,
            weak_coverage_threshold=-95.0,
            over_coverage_threshold=-65.0,
            qos_sinr_threshold=0.0,
            horizon=24 * 7,  # e.g., one week episode
        )
        env = Monitor(env, log_dir)
    except Exception as e:
        logger.exception(f"Failed to create RL environment: {e}")
        return

    # Define policy network architecture for more capacity
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Define and train the agent with hyperparameters, tune where needed
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-2,
        ent_coef=0.001,
        vf_coef=0.7,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="energy_rl_model")

    logger.info(f"Starting RL training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="PPO_EnergySaver")
        model.save(rl_model_path)
        logger.info(f"Training complete. Model saved to {rl_model_path}")
    except Exception as e:
        logger.exception(f"Error during RL training: {e}")
    finally:
        env.close()
