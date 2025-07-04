# rl_predictor.py

# This module is designed to load a trained RL agent and predict the optimal cell configuration,
# including tilt angles and on/off states, for energy-saving applications. The prediction is performed for a specified
# time tick (hour), and the results are displayed in a structured format. The module utilizes the PPO algorithm from
# stable_baselines3 and relies on project-specific constants and topology data.

import logging
import os
import sys

import pandas as pd

try:
    from stable_baselines3 import PPO

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
    sys.path.insert(0, PROJECT_ROOT)
    from radp.digital_twin.utils import constants as c
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


def run_rl_prediction(model_load_path: str, topology_path: str, target_tick: int):
    """
    The process of loading a trained RL agent and predicting the optimal cell configuration,
    including tilts and on/off states, is performed by this function. The prediction is executed
    for a specified tick (hour), and the results are presented in a tabular format.
    """
    logger.info(f"--- Running RL Energy Saver Prediction for Tick {target_tick} ---")
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

        config_list = []
        if len(action_indices) != len(cell_ids_ordered):
            logger.error(f"Action length {len(action_indices)} != num cells {len(cell_ids_ordered)}")
            return

        for i, cell_action_idx in enumerate(action_indices):
            cell_id = cell_ids_ordered[i]
            if cell_action_idx == len(TILT_SET):  # The special index for "OFF" is checked.
                state, tilt = "OFF", "N/A"
            else:
                state, tilt = "ON", TILT_SET[cell_action_idx]
            config_list.append({"cell_id": cell_id, "predicted_state": state, "predicted_cell_el_deg": tilt})

        predicted_config_df = pd.DataFrame(config_list)
        print("\n--- Predicted Optimal Configuration ---")
        print(f"--- For Tick/Hour: {target_tick} ---")
        print(predicted_config_df.to_string(index=False))

    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")
