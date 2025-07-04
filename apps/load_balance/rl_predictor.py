# rl_predictor.py

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO

    # The path is added to enable the import of constants.
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
    sys.path.insert(0, PROJECT_ROOT)
    from radp.digital_twin.utils import constants as c
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}. Ensure stable-baselines3 is installed and RADP_ROOT is correct.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The set of possible tilt values for the predictor is defined here and must match those used during training.
TILT_SET_PREDICTOR = [float(t) for t in range(21)]


def map_action_to_config_df(action: np.ndarray, cell_ids: List[str], possible_tilts: List[float]) -> pd.DataFrame:
    """A numerical action from the RL agent is converted to a readable configuration DataFrame by this function."""
    config_list = []
    num_tilt_options = len(possible_tilts)
    for i, cell_action_idx in enumerate(action):
        cell_id = cell_ids[i]
        tilt_index = np.clip(cell_action_idx, 0, num_tilt_options - 1)
        tilt = possible_tilts[tilt_index]
        config_list.append({"cell_id": cell_id, "predicted_cell_el_deg": tilt})
    return pd.DataFrame(config_list)


def run_rl_prediction(model_load_path: str, topology_path: str, target_tick: int):
    """
    A trained RL agent is loaded and used to predict the optimal cell tilt configuration
    for a specified tick by this function.
    """
    logger.info(f"--- Running CCO RL Prediction for Tick {target_tick} ---")
    if not (0 <= target_tick <= 23):
        logger.error(f"Target tick {target_tick} is out of range (0-23).")
        return

    try:
        topology_df = pd.read_csv(topology_path)
        cell_ids_ordered = topology_df[getattr(c, "CELL_ID", "cell_id")].unique().tolist()

        model_file = model_load_path if model_load_path.endswith(".zip") else f"{model_load_path}.zip"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"RL Model file not found: {model_file}")

        rl_model = PPO.load(model_file)
        logger.info(f"Loaded trained RL model from {model_file}")

        action_indices, _ = rl_model.predict(target_tick, deterministic=True)

        predicted_config_df = map_action_to_config_df(action_indices, cell_ids_ordered, TILT_SET_PREDICTOR)

        print("\n--- Predicted Optimal Tilt Configuration ---")
        print(f"--- For Tick/Hour: {target_tick} ---")
        print(predicted_config_df.to_string(index=False))

    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")
