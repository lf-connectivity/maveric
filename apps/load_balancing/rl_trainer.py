# rl_trainer.py

import logging
import os
import sys
from typing import List

import pandas as pd

# The project root directory is added to the system path to enable RADP imports.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
sys.path.insert(0, PROJECT_ROOT)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    # The environment is imported from its new file.
    from apps.load_balancing.cco_rl_env import CCO_RL_Env
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
    from radp.digital_twin.utils import constants as c
except ImportError as e:
    print(
        f"""FATAL: Error importing necessary libraries for RL Trainer: {e}.
        Ensure all dependencies are installed and paths are correct."""
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default values for potentially missing topology columns are defined.
DEFAULT_HTX = 25.0
DEFAULT_HRX = 1.5
DEFAULT_CELL_AZ_DEG = 0.0
DEFAULT_CELL_CARRIER_FREQ_MHZ = 2100.0
# The tilt set must match the values used in rl_predictor.py.
TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


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
    Data is loaded, the CCO environment is initialized, and the RL agent is trained by this function.
    """
    logger.info("--- Starting Load Balancing RL Agent Training ---")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.dirname(rl_model_path):
        os.makedirs(os.path.dirname(rl_model_path), exist_ok=True)

    try:
        # Dataframes required for RL training are loaded and prepared.
        logger.info("Loading and preparing dataframes for RL training...")
        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        site_config_df = pd.merge(topology_df, config_df, on="cell_id", how="left")

        site_config_df["cell_el_deg"].fillna(TILT_SET[len(TILT_SET) // 2], inplace=True)

        # It is ensured that all required columns exist in site_config_df for the Gym environment.
        required_cols_with_defaults = {
            getattr(c, "HTX", "hTx"): DEFAULT_HTX,
            getattr(c, "HRX", "hRx"): DEFAULT_HRX,
            getattr(c, "CELL_AZ_DEG", "cell_az_deg"): DEFAULT_CELL_AZ_DEG,
            getattr(c, "CELL_CARRIER_FREQ_MHZ", "cell_carrier_freq_mhz"): DEFAULT_CELL_CARRIER_FREQ_MHZ,
        }
        for col, default_val in required_cols_with_defaults.items():
            if col not in site_config_df.columns:
                logger.warning(f"Topology data missing '{col}'. Adding default value: {default_val}")
                site_config_df[col] = default_val
        # End of required columns check.

        bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(bdt_model_map)} cells.")

        # UE data from all specified training days is loaded.
        ue_data_per_tick = {tick: [] for tick in range(24)}
        for day in training_days:
            day_dir = os.path.join(base_ue_data_dir, f"Day_{day}", "ue_data_gym_ready")
            if not os.path.isdir(day_dir):
                logger.warning(f"Training data directory not found for Day_{day}: {day_dir}")
                continue
            for tick in range(24):
                filepath = os.path.join(day_dir, f"generated_ue_data_for_cco_{tick}.csv")
                if os.path.exists(filepath):
                    ue_data_per_tick[tick].append(pd.read_csv(filepath))

        if not any(ue_data_per_tick.values()):
            raise FileNotFoundError(f"No UE data loaded from any training day directories in {base_ue_data_dir}")

    except Exception as e:
        logger.exception(f"Failed during data loading for RL training: {e}")
        return

    # The environment is instantiated.
    try:
        logger.info("Instantiating CCO RL Environment...")
        reward_weights = {"cco": 1.0, "load_balance": 2.0}  # These weights are tunable.
        env = CCO_RL_Env(
            bdt_predictors=bdt_model_map,
            topology_df=site_config_df,  # The hydrated dataframe is passed.
            ue_data_per_tick=ue_data_per_tick,
            possible_tilts=TILT_SET,
            reward_weights=reward_weights,
            weak_coverage_threshold_reward=-95.0,
            qos_sinr_threshold=0.0,
            horizon=24 * 7,  # An episode of one week is used before 'done' is returned as True.
        )
        env = Monitor(env, log_dir)
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create RL environment: {e}")
        return

    # The agent is defined and trained.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="cco_rl_model")

    logger.info(f"Starting RL training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="PPO_LoadBalance")
        model.save(rl_model_path)
        logger.info(f"Training complete. Model saved to {rl_model_path}")
    except Exception as e:
        logger.exception(f"An error occurred during RL agent training: {e}")
    finally:
        env.close()
