# rl_energy_saving_env.py

import logging
import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

try:
    # It is assumed that CCOEngine is located in a module accessible from this path.
    # Adjustments should be made if the project structure differs.
    from apps.coverage_capacity_optimization.cco_engine import CcoEngine
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.cell_selection import perform_attachment
except ImportError as e:
    print(f"FATAL: Error importing RADP modules for TickAwareEnergyEnv: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class TickAwareEnergyEnv(gym.Env):
    """
    An RL environment for energy savings is provided, which is aware of the time of day (tick)
    and utilizes corresponding per-tick UE data for local simulation with BDT models.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bayesian_digital_twins: Dict[str, BayesianDigitalTwin],
        site_config_df: pd.DataFrame,
        ue_data_per_tick: Dict[int, List[pd.DataFrame]],
        tilt_set: List[float],
        reward_weights: Dict[str, float],
        weak_coverage_threshold: float,
        over_coverage_threshold: float,
        qos_sinr_threshold: float,
        horizon: int,
    ):
        super().__init__()
        self.bdt_predictors = bayesian_digital_twins
        self.site_config_df_initial = site_config_df.copy()

        self.COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")
        self.COL_CELL_EL_DEG = getattr(c, "CELL_EL_DEG", "cell_el_deg")
        self.COL_RSRP_DBM = getattr(c, "RSRP_DBM", "rsrp_dbm")
        self.COL_SINR_DB = getattr(c, "SINR_DB", "sinr_db")
        self.COL_UE_ID = getattr(c, "UE_ID", "ue_id")  # The UE ID is standardized if present in constants.

        self.cell_ids = self.site_config_df_initial[self.COL_CELL_ID].unique().tolist()
        self.num_cells = len(self.cell_ids)
        self.ue_data_per_tick = ue_data_per_tick
        self.tilt_set = tilt_set
        self.num_tilt_options = len(self.tilt_set)
        self.reward_weights = reward_weights
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.horizon = horizon

        self.current_step_in_episode = 0
        self.current_tick_of_day = 0
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()

        self.action_space = spaces.MultiDiscrete([self.num_tilt_options + 1] * self.num_cells)
        self.observation_space = spaces.Discrete(24)
        logger.info(f"TickAwareEnergyEnv initialized for {self.num_cells} cells.")

    def _take_action(self, action: np.ndarray):
        """
        The on/off and tilt state of each cell is updated based on the provided action.
        If the action is invalid, the previous state is retained for that cell.
        """
        new_on_off_state = np.ones(self.num_cells, dtype=int)
        new_tilt_state = np.zeros(self.num_cells, dtype=float)
        for i, action_for_cell in enumerate(action):
            if action_for_cell == self.num_tilt_options:
                new_on_off_state[i] = 0
                new_tilt_state[i] = self.tilt_state[i]
            elif 0 <= action_for_cell < self.num_tilt_options:
                new_on_off_state[i] = 1
                new_tilt_state[i] = self.tilt_set[action_for_cell]
            else:
                logger.warning(f"Invalid action {action_for_cell} for cell {i}. Keeping previous state.")
                new_on_off_state[i] = self.on_off_state[i]
                new_tilt_state[i] = self.tilt_state[i]
        self.on_off_state = new_on_off_state
        self.tilt_state = new_tilt_state
        self.site_config_df_state[self.COL_CELL_EL_DEG] = self.tilt_state

    def _run_local_simulation(self, ue_data_template: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        A local simulation is performed using the current configuration and UE data template.
        Predictions are generated for each active cell using the corresponding BDT predictor.
        The results are combined and the attachment procedure is performed.
        """
        if ue_data_template is None or ue_data_template.empty:
            return None
        all_cell_predictions = []
        active_cell_ids = self.site_config_df_state[self.on_off_state == 1][self.COL_CELL_ID].tolist()

        for cell_id in active_cell_ids:
            bdt_predictor = self.bdt_predictors.get(cell_id)
            if not bdt_predictor:
                continue

            current_cell_config = self.site_config_df_state[self.site_config_df_state[self.COL_CELL_ID] == cell_id]
            pred_frames = BayesianDigitalTwin.create_prediction_frames(
                site_config_df=current_cell_config, prediction_frame_template=ue_data_template
            )
            df_to_predict_on = pred_frames.get(cell_id)
            if df_to_predict_on is not None and not df_to_predict_on.empty:
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_to_predict_on])
                all_cell_predictions.append(df_to_predict_on)

        if not all_cell_predictions:
            return pd.DataFrame()
        combined_rf = pd.concat(all_cell_predictions, ignore_index=True)
        return perform_attachment(
            combined_rf, self.site_config_df_state[self.site_config_df_state[self.COL_CELL_ID].isin(active_cell_ids)]
        )

    def _calculate_reward(self, cell_selected_rf_df: Optional[pd.DataFrame]) -> Tuple[float, Dict]:
        """
        The reward is calculated based on coverage_capacity (CCO), load balance, and energy saving.
        If no RF data is available, a penalty is applied unless all cells are off.
        """
        info = {"cco_score": 0.0, "load_balance_score": 0.0, "energy_saving_score": 0.0}
        num_active_cells = np.sum(self.on_off_state)
        info["energy_saving_score"] = (1.0 - (num_active_cells / self.num_cells)) * 100 if self.num_cells > 0 else 100

        if cell_selected_rf_df is None or cell_selected_rf_df.empty:
            if num_active_cells == 0:
                logger.debug("All cells off. Reward is based purely on energy saving.")
            else:
                logger.warning(
                    f"Tick {self.current_tick_of_day}: No RF data for reward calc. High penalty on other metrics."
                )
                info["cco_score"] = -100.0
        else:
            try:
                cov_df = CcoEngine.rf_to_coverage_dataframe(
                    cell_selected_rf_df,
                    weak_coverage_threshold=self.weak_coverage_threshold,
                    over_coverage_threshold=self.over_coverage_threshold,
                )

                active_cell_ids = self.site_config_df_state[self.on_off_state == 1][self.COL_CELL_ID].tolist()

                info["cco_score"] = CcoEngine.get_cco_objective_value(
                    cov_df, active_ids_list=active_cell_ids, id_field=self.COL_CELL_ID
                )

                active_topo = self.site_config_df_state[self.on_off_state == 1]

                if not active_topo.empty:
                    ue_counts = (
                        cell_selected_rf_df[self.COL_CELL_ID]
                        .value_counts()
                        .reindex(active_topo[self.COL_CELL_ID], fill_value=0)
                    )
                    info["load_balance_score"] = -ue_counts.std() if len(ue_counts) > 1 else 0.0

                else:
                    info["load_balance_score"] = 0.0  # Perfect balance is assumed if no active cells are present.

            except Exception as e:
                logger.exception(f"Error during reward calculation: {e}")
                return -500.0, info

        reward = sum(self.reward_weights.get(k, 1.0) * v for k, v in info.items() if k in self.reward_weights)
        info["reward_total"] = reward

        return reward, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        The environment state is reset to its initial configuration.
        The tick of the day is randomly sampled.
        """
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        self.current_tick_of_day = self.observation_space.sample()
        self.site_config_df_state = self.site_config_df_initial.copy()
        self.on_off_state = np.ones(self.num_cells, dtype=int)
        self.tilt_state = self.site_config_df_state[self.COL_CELL_EL_DEG].values.copy()
        return self.current_tick_of_day, {}

    def step(self, action: np.ndarray):
        """
        The provided action is applied, the simulation is run, and the reward is calculated.
        The tick of the day is incremented, and the episode termination is checked.
        """
        self._take_action(action)
        ue_data_df_list = self.ue_data_per_tick.get(self.current_tick_of_day)

        if not ue_data_df_list:
            reward, info = -1000.0, {"error": f"No UE data for tick {self.current_tick_of_day}"}
        else:
            # A random index is used to select a DataFrame from the list for the current tick.
            random_day_index = np.random.randint(0, len(ue_data_df_list))
            ue_data_df = ue_data_df_list[random_day_index]

            # active_config = self.site_config_df_state[self.on_off_state == 1]
            cell_selected_rf = self._run_local_simulation(ue_data_df)
            reward, info = self._calculate_reward(cell_selected_rf)

        info["tick"] = self.current_tick_of_day
        self.current_step_in_episode += 1
        self.current_tick_of_day = (self.current_tick_of_day + 1) % 24
        next_observation = self.current_tick_of_day

        terminated = self.current_step_in_episode >= self.horizon
        truncated = False

        return next_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
