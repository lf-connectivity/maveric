# cco_rl_env.py

import logging
import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# It is assumed that RADP imports are configured correctly via sys.path or RADP_ROOT.
# The import process is managed by the scripts that import this module (e.g., rl_trainer.py).
try:
    from apps.coverage_capacity_optimization.cco_engine import CcoEngine
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.cell_selection import perform_attachment
except ImportError as e:
    print(f"FATAL: Error importing RADP modules for CCO_RL_Env: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class CCO_RL_Env(gym.Env):
    """
    An RL environment for CCO is provided. The hour of the day is observed by the agent, and tilts are selected.
    RF simulation is performed locally using a pre-trained BDT model map.
    UE data is sampled from a pool containing data from multiple training days.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bdt_predictors: Dict[str, BayesianDigitalTwin],
        topology_df: pd.DataFrame,
        ue_data_per_tick: Dict[int, List[pd.DataFrame]],
        possible_tilts: List[float],
        reward_weights: Dict[str, float],
        qos_sinr_threshold: float,
        weak_coverage_threshold_reward: float,
        horizon: int,
    ):
        super().__init__()
        self.bdt_predictors = bdt_predictors
        self.topology_df = topology_df.copy()

        # Constants are defined using getattr for safety.
        self.COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")
        self.COL_CELL_EL_DEG = getattr(c, "CELL_EL_DEG", "cell_el_deg")
        self.COL_RSRP_DBM = getattr(c, "RSRP_DBM", "rsrp_dbm")
        self.COL_SINR_DB = getattr(c, "SINR_DB", "sinr_db")
        self.COL_LOC_X = getattr(c, "LOC_X", "loc_x")
        self.COL_LOC_Y = getattr(c, "LOC_Y", "loc_y")

        self.cell_ids = self.topology_df[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_per_tick = (
            ue_data_per_tick  # A dictionary mapping each tick to a list of DataFrames containing UE data.
        )
        self.possible_tilts = possible_tilts
        self.num_tilt_options = len(self.possible_tilts)
        self.reward_weights = reward_weights
        self.qos_sinr_threshold = qos_sinr_threshold
        self.weak_coverage_threshold_reward = weak_coverage_threshold_reward
        self.over_coverage_threshold_reward = -65.0  # A default value is provided for use by CcoEngine if required.
        self.horizon = horizon

        self.current_step_in_episode = 0
        self.current_tick = 0
        self.site_config_df_state = self.topology_df.copy()

        self.action_space = spaces.MultiDiscrete([self.num_tilt_options] * self.num_cells)
        self.observation_space = spaces.Discrete(24)
        logger.info(f"CCO RL Env initialized for {self.num_cells} cells.")

    def _map_action_to_config(self, action: np.ndarray) -> pd.DataFrame:
        # The provided action is mapped to a configuration DataFrame containing
        # cell IDs and their corresponding tilt values.
        config_data = []
        for i, cell_id in enumerate(self.cell_ids):
            tilt_index = np.clip(action[i], 0, self.num_tilt_options - 1)
            tilt_value = self.possible_tilts[tilt_index]
            config_data.append({self.COL_CELL_ID: cell_id, self.COL_CELL_EL_DEG: tilt_value})
        return pd.DataFrame(config_data)

    def _run_local_simulation(self, config_df: pd.DataFrame, ue_data_template: pd.DataFrame) -> Optional[pd.DataFrame]:
        # A local RF simulation is performed using the provided configuration and UE data template.
        if ue_data_template is None or ue_data_template.empty:
            return None
        all_cell_predictions = []
        for cell_id in self.cell_ids:
            bdt_predictor = self.bdt_predictors.get(cell_id)
            if not bdt_predictor:
                continue

            current_cell_config = self.site_config_df_state[
                self.site_config_df_state[self.COL_CELL_ID] == cell_id
            ].copy()
            current_cell_config[self.COL_CELL_EL_DEG] = config_df[config_df[self.COL_CELL_ID] == cell_id][
                self.COL_CELL_EL_DEG
            ].iloc[0]

            try:
                pred_frames = BayesianDigitalTwin.create_prediction_frames(
                    site_config_df=current_cell_config, prediction_frame_template=ue_data_template
                )
            except Exception as e:
                logger.error(f"Error in create_prediction_frames for cell {cell_id}: {e}")
                continue

            df_to_predict_on = pred_frames.get(cell_id)
            if df_to_predict_on is not None and not df_to_predict_on.empty:
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_to_predict_on])
                all_cell_predictions.append(df_to_predict_on)

        if not all_cell_predictions:
            return None
        combined_rf = pd.concat(all_cell_predictions, ignore_index=True)
        # The perform_attachment function is used to assign UEs to cells based on the simulated RF data.
        return perform_attachment(combined_rf, self.topology_df)

    def _calculate_load_balancing_objective(self, rf_dataframe: pd.DataFrame) -> float:
        # The load balancing objective is calculated as the negative standard deviation of UE counts per cell.
        if rf_dataframe is None or rf_dataframe.empty:
            return -100.0  # A high penalty is applied if the input is invalid.
        ue_counts = rf_dataframe[self.COL_CELL_ID].value_counts().reindex(self.cell_ids, fill_value=0)
        return -ue_counts.std() if len(ue_counts) > 1 else 0.0

    def _calculate_reward(self, cell_selected_rf: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        # The reward is calculated based on coverage_capacity (CCO), load balancing, and QoS metrics.
        info = {"cco_score": 0.0, "load_balance_score": 0.0}
        if cell_selected_rf is None or cell_selected_rf.empty:
            logger.warning(f"Tick {self.current_tick}: Calculating reward based on empty RF data. High penalty.")
            return -200.0, info

        try:
            cov_df = CcoEngine.rf_to_coverage_dataframe(
                cell_selected_rf,
                weak_coverage_threshold=self.weak_coverage_threshold_reward,
                over_coverage_threshold=self.over_coverage_threshold_reward,
            )

            active_cell_ids = self.cell_ids

            info["cco_score"] = CcoEngine.get_cco_objective_value(
                cov_df, active_ids_list=active_cell_ids, id_field=self.COL_CELL_ID
            )
            info["load_balance_score"] = self._calculate_load_balancing_objective(cell_selected_rf)

            reward = (
                self.reward_weights.get("cco", 1.0) * info["cco_score"]
                + self.reward_weights.get("load_balance", 1.0) * info["load_balance_score"]
            )
        except Exception as e:
            logger.exception(f"Error during reward calculation: {e}")
            return -500.0, info

        info["reward_total"] = reward
        return reward, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        # The environment is reset to its initial state, and a random tick is selected as the starting observation.
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        self.current_tick = self.observation_space.sample()
        return self.current_tick, {}

    def step(self, action: np.ndarray):
        # The environment is advanced by one step using the provided action.
        config_df = self._map_action_to_config(action)

        # A random day's data is sampled for the current tick.
        ue_data_df_list = self.ue_data_per_tick.get(self.current_tick)

        if not ue_data_df_list:
            reward, info = -1000.0, {"error": f"No UE data for tick {self.current_tick}"}
        else:
            # A DataFrame is randomly selected from the list corresponding to the current tick.
            random_day_index = np.random.randint(0, len(ue_data_df_list))
            ue_data_df = ue_data_df_list[random_day_index]

            cell_selected_rf = self._run_local_simulation(config_df, ue_data_df)
            reward, info = self._calculate_reward(cell_selected_rf)

        info["tick"] = self.current_tick
        self.current_step_in_episode += 1
        self.current_tick = (self.current_tick + 1) % 24
        next_observation = self.current_tick

        terminated = self.current_step_in_episode >= self.horizon
        truncated = False

        return next_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
