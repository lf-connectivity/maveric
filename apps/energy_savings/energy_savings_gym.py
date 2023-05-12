# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import gym
import numpy as np
import pandas as pd
from gym.spaces import Box, MultiDiscrete

from apps.coverage_capacity_optimization.cco_engine import CcoEngine
from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import perform_attachment
from radp.digital_twin.utils.constants import CELL_EL_DEG, LOC_X, LOC_Y, RXPOWER_DBM
from radp.digital_twin.utils.gis_tools import GISTools


class EnergySavingsGym(gym.Env):
    """RIC x/r-App Energy Savings Problem formulated as Open AI Gym Environment.

    Ref : https://fb.quip.com/xqk4AO44BmbW
    """

    ENERGY_MAX_PER_CELL = 40
    MAX_SUM_PEAK_RATE_CLUSTER = 10000
    MAX_CLUSTER_CCO = 0

    def __init__(
        self,
        bayesian_digital_twin: BayesianDigitalTwin,
        site_config_df: pd.DataFrame,
        prediction_frame_template: pd.DataFrame,
        tilt_set: List[int],
        weak_coverage_threshold: float = -90,
        over_coverage_threshold: float = 0,
        # CPO constraint violation weighting factor
        lambda_: float = 0.5,
        # cols : [zoom, tile_x, tile_y, hour_idx_over_upcoming_168_hrs, peak_rate_ucb]
        traffic_model_df: pd.DataFrame = None,
        ue_track_generator: UETracksGenerator = None,
        horizon: int = 168,
        min_rsrp: float = -140.0,
        seed: int = 0,
        debug: bool = False,
    ):
        """
        @args
            `bayesian_digital_twin`: Trained Bayesian Gaussian Process Regression Model
            `site_config_df` : 1 unique cell per row, contains at least the columns
                [cell_lat, cell_lon, cell_el_deg, cell_az_deg, cell_id]
                Assumption : `bayesian_digital_twin` was trained with respect tp `site_config_df`
            `prediction_frame_template` : 1 prediction point per row, contains columsn [loc_x, loc_y]
                e.g. loc_x is longitude, and loc_y is latitude
            `valid_configuration_values`: dictionary containibg valid values for settings,
                keyed by attribute id that must correspond to a column in site_config_df,
                for e.g. `cell_el_deg`
        """
        # TODO (paulvarkey) : ensure consistencies :
        # 1. that bayesian_digital_twin has x_columns
        #    [CELL_LAT, CELL_LON, CELL_EL_DEG, LOG_DISTANCE, RELATIVE_BEARING]
        # 2. that bayesian_digital_twin.num_cells == len(site_config_df)

        super(EnergySavingsGym, self).__init__()

        self.bayesian_digital_twin = bayesian_digital_twin
        self.site_config_df = site_config_df
        self.num_cells = len(self.site_config_df)
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold
        self.lambda_ = lambda_
        self.horizon = horizon
        self.current_step = 0
        self.tilt_set = tilt_set
        self.min_rsrp = min_rsrp
        self.traffic_model_df = traffic_model_df
        self.ue_track_generator = ue_track_generator
        self.ue_tracks = None

        if self.ue_track_generator:
            self.ue_tracks = self.ue_track_generator.generate()
            self.min_lat = self.ue_track_generator.min_lat
            self.max_lat = self.ue_track_generator.max_lat
            self.min_lon = self.ue_track_generator.min_lon
            self.max_lon = self.ue_track_generator.max_lon
            self.x_dim = self.ue_track_generator.lon_x_dims
            self.y_dim = self.ue_track_generator.lon_y_dims

        # State
        # create prediction frames, one per cell
        # the order of `prediction_dfs` is the same
        # as the row-order of `site_config_df`
        self.prediction_frame_template = prediction_frame_template
        self.prediction_dfs = BayesianDigitalTwin.create_prediction_frames(
            site_config_df=self.site_config_df,
            prediction_frame_template=prediction_frame_template,
        )
        # Init on-off and tilt states
        self.on_off_state = [1] * len(self.prediction_dfs)
        self.tilt_state = []
        # site_config_df will hold the original and site_config_df_state will hold the variable
        self.site_config_df_state = site_config_df.copy(deep=True)
        for c in self.site_config_df.itertuples():
            self.tilt_state.append(c.cell_el_deg)
        self.tilt_state = np.array(self.tilt_state)

        """ MultiDiscrete action space in which the on/off state
        of a particular cell is encoded, where the value of len(self.tilt_set)
        indicates 'off', and all other valid tilt values implicitly mean 'on'.
        """
        self.action_space = MultiDiscrete(self.num_cells * [len(self.tilt_set) + 1])
        self.action_space.seed(seed)

        # Observations :
        #  1. E(c) : average energy used in the network
        #  2. T(B) : traffic across all tiles B, spanned by cluster
        #  3. CCO(c, T(B)) : Coverage and Capacity Metric
        self.observation_space = Box(
            low=np.float16(np.array([0, 0, 0])),
            high=np.float16(
                np.array(
                    [
                        EnergySavingsGym.ENERGY_MAX_PER_CELL,
                        EnergySavingsGym.MAX_SUM_PEAK_RATE_CLUSTER,
                        EnergySavingsGym.MAX_CLUSTER_CCO,
                    ]
                )
            ),
            dtype=np.float16,
        )
        self.debug = debug

        # Reward when all cells are off:
        self.r_norm = (1 - lambda_) * (
            -10 * np.log10(self.num_cells) - over_coverage_threshold + min_rsrp - weak_coverage_threshold
        )

    def _next_observation(self):

        if self.ue_tracks:
            data = next(self.ue_tracks)
            for batch in data:
                lonlat_pairs = GISTools.converting_xy_points_into_lonlat_pairs(
                    batch,
                    x_dim=self.x_dim,
                    y_dim=self.y_dim,
                    min_latitude=self.min_lat,
                    max_latitude=self.max_lat,
                    max_longitude=self.max_lon,
                    min_longitude=self.min_lon,
                )

                main_dataframe = pd.DataFrame(lonlat_pairs, columns=[LOC_X, LOC_Y])

            self.prediction_dfs = BayesianDigitalTwin.create_prediction_frames(
                site_config_df=self.site_config_df,
                prediction_frame_template=main_dataframe,
            )
        prediction_dfs = list(self.prediction_dfs.values())
        (pred_means, _) = self.bayesian_digital_twin.predict_distributed_gpmodel(
            prediction_dfs=prediction_dfs,
        )

        # zero out cells that are turned off and re-compute RSRP and SINR
        for idx in range(self.num_cells):
            if self.on_off_state[idx] == 0:
                pred_means[idx] = self.min_rsrp

        # add pred_means and pred_std to prediction_dfs
        for idx in range(self.num_cells):
            prediction_dfs[idx][RXPOWER_DBM] = pred_means[idx]

        # compute RSRP and SINR
        rf_dataframe = perform_attachment(prediction_dfs[0], self.site_config_df)

        coverage_dataframe = CcoEngine.rf_to_coverage_dataframe(
            rf_dataframe=rf_dataframe,
            weak_coverage_threshold=self.weak_coverage_threshold,
            over_coverage_threshold=self.over_coverage_threshold,
        )
        if self.traffic_model_df is None:
            cco_objective_metric = (
                coverage_dataframe["weak_coverage"].mean() + coverage_dataframe["over_coverage"].mean()
            )

        else:
            processed_coverage_dataframe = CcoEngine.augment_coverage_df_with_normalized_traffic_model(
                self.traffic_model_df,
                "avg_of_average_egress_kbps_across_all_time",
                coverage_dataframe,
            )

            cco_objective_metric = CcoEngine.traffic_normalized_cco_metric(processed_coverage_dataframe)

        # Output for debugging/postprocessing purposes
        if self.debug:
            self.rf_dataframe = rf_dataframe
            self.coverage_dataframe = coverage_dataframe

        return (
            EnergySavingsGym.ENERGY_MAX_PER_CELL * sum(self.on_off_state) / len(self.on_off_state),
            0.0,
            cco_objective_metric,  # TODO : normalized this against MAX_CLUSTER_CCO
        )

    def reward(
        self,
        energy_consumption: float,
        cco_objective_metric: float,
    ) -> float:
        return self.lambda_ * -1.0 * energy_consumption + (1 - self.lambda_) * cco_objective_metric - self.r_norm

    def make_action_from_state(self):
        action = np.empty(self.num_cells, dtype=int)
        for idx in range(self.num_cells):
            if self.on_off_state[idx] == 0:
                action[idx] = len(self.tilt_set)
            else:
                action[idx] = self.tilt_set.index(self.tilt_state[idx])
        return action

    def _take_action(self, action):

        num_cells = len(self.site_config_df)
        # on_off_cell_state captures the on/off state of each cell (on is `1`)
        on_off_cell_state = [1] * num_cells

        for idx, a in enumerate(action):
            if a == len(self.tilt_set):
                on_off_cell_state[idx] = 0

        self.on_off_state = on_off_cell_state

        self.tilt_state = []
        for idx in range(num_cells):
            if action[idx] == len(self.tilt_set):
                cell_id = self.site_config_df.iloc[idx].cell_id
                self.tilt_state.append(self.prediction_dfs[cell_id].cell_el_deg)
            else:
                self.tilt_state.append(self.tilt_set[action[idx]])

        # Update site_config and re-create the prediction frames
        self.site_config_df_state[CELL_EL_DEG] = self.tilt_state
        self.prediction_dfs = BayesianDigitalTwin.create_prediction_frames(
            site_config_df=self.site_config_df_state,
            prediction_frame_template=self.prediction_frame_template,
        )

    def reset(self):
        # TODO : check if we need to reset self.current_step
        return self._next_observation()

    def step(self, action):

        # Execute one time step within the environment
        self._take_action(action)

        obs = self._next_observation()

        # Check if we should exit
        self.current_step += 1
        done = False
        if self.current_step == self.horizon:
            done = True

        reward = self.reward(obs[0], obs[2])

        return obs, reward, done, {}

    def get_all_possible_actions(self, possible_actions: List[List[int]]) -> List[List[int]]:
        """
        A recursive function to get all possible actions as a list.
        Useful for bruteforce search.
        """
        if len(possible_actions) == 1:
            return possible_actions
        pairs = []
        for action in possible_actions[0]:
            for action2 in possible_actions[1]:
                pairs.append(self.flatten([action, action2]))
        new_possible_actions = [pairs] + possible_actions[2:]
        possible_action_vectors = self.get_all_possible_actions(new_possible_actions)
        return possible_action_vectors

    @staticmethod
    def flatten(actions: List[int]) -> List[int]:
        new_actions = []
        for action in actions:
            if type(action) == list:
                new_actions += action
            elif type(action) == int:
                new_actions.append(action)
        return new_actions
