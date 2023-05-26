# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from apps.coverage_capacity_optimization.cco_engine import CcoEngine
from apps.energy_savings.energy_savings_gym import EnergySavingsGym
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.constants import (
    CELL_AZ_DEG,
    CELL_CARRIER_FREQ_MHZ,
    CELL_EL_DEG,
    CELL_ID,
    CELL_LAT,
    CELL_LON,
    CELL_RXPWR_DBM,
    CELL_TXPWR_DBM,
    HRX,
    HTX,
    LOC_X,
    LOC_Y,
    LOG_DISTANCE,
    RELATIVE_BEARING,
    RELATIVE_TILT,
    RELATIVE_TILT_SQUARED,
)


class TestEnergySavingsGym(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_max = {
            CELL_EL_DEG: 50,
            CELL_LAT: 90,
            CELL_LON: 180,
            LOG_DISTANCE: 5,  # log(1 + max distance of 50000 meters)
            RELATIVE_BEARING: 360,
            RELATIVE_TILT: 180,
            RELATIVE_TILT_SQUARED: 32400,
        }
        cls.x_min = {
            CELL_EL_DEG: -10,
            LOC_Y: -90,
            LOC_X: -180,
            LOG_DISTANCE: 0,
            RELATIVE_BEARING: 0,
            RELATIVE_TILT: -90,
            RELATIVE_TILT_SQUARED: 0,
        }

    def test_construction_and_iterations(self):
        # initialize list of lists
        data = [
            [35.631569, 139.705848, 1, 50, 71560196, 2100, 30, 2],
            [
                35.631569,
                139.705848,
                5,
                180,
                71560197,
                2100,
                30,
                2,
            ],
        ]

        # Create the pandas DataFrame
        site_config_df = pd.DataFrame(
            data,
            columns=[
                CELL_LAT,
                CELL_LON,
                CELL_EL_DEG,
                CELL_AZ_DEG,
                CELL_ID,
                CELL_CARRIER_FREQ_MHZ,
                HTX,
                HRX,
            ],
        )

        # initialize list of lists
        data_0 = [
            [
                139.692500,
                35.629167,
                46,
                50,
                2,
                -129.364914,
                71560196,
                35.631569,
                139.705848,
                7.121227,
                27.510535,
                66.415073,
            ],
            [
                139.682778,
                35.622222,
                46,
                50,
                2,
                -152.646576,
                71560196,
                35.631569,
                139.705848,
                7.755081,
                13.499837,
                64.691793,
            ],
        ]
        data_1 = [
            [
                139.693889,
                35.635833,
                46,
                180,
                2,
                -128.653122,
                71560197,
                35.631569,
                139.705848,
                7.075433,
                293.684706,
                66.541222,
            ],
            [
                139.680556,
                35.623611,
                46,
                180,
                2,
                -129.459549,
                71560197,
                35.631569,
                139.705848,
                7.805878,
                248.831687,
                64.555558,
            ],
        ]

        # Create the pandas DataFrame
        df_0 = pd.DataFrame(
            data_0,
            columns=[
                LOC_X,
                LOC_Y,
                CELL_TXPWR_DBM,
                CELL_AZ_DEG,
                CELL_EL_DEG,
                CELL_RXPWR_DBM,
                CELL_ID,
                CELL_LAT,
                CELL_LON,
                LOG_DISTANCE,
                RELATIVE_BEARING,
                RELATIVE_TILT,
            ],
        )
        df_1 = pd.DataFrame(
            data_1,
            columns=[
                LOC_X,
                LOC_Y,
                CELL_TXPWR_DBM,
                CELL_AZ_DEG,
                CELL_EL_DEG,
                CELL_RXPWR_DBM,
                CELL_ID,
                CELL_LAT,
                CELL_LON,
                LOG_DISTANCE,
                RELATIVE_BEARING,
                RELATIVE_TILT,
            ],
        )

        cell_id_1 = 71560196
        cell_id_2 = 71560197
        training_data = {
            cell_id_1: df_0,
            cell_id_2: df_1,
        }

        bayesian_digital_twins = {}
        for idx, cell_id in enumerate(site_config_df.cell_id):
            bayesian_digital_twin = BayesianDigitalTwin(
                data_in=[training_data[cell_id]],
                x_columns=[
                    CELL_LAT,
                    CELL_LON,
                    CELL_EL_DEG,
                    LOG_DISTANCE,
                    RELATIVE_BEARING,
                ],
                y_columns=[CELL_RXPWR_DBM],
                x_max=self.x_max,
                x_min=self.x_min,
            )
            bayesian_digital_twins[cell_id] = bayesian_digital_twin

        energy_savings_gym = EnergySavingsGym(
            bayesian_digital_twins=bayesian_digital_twins,
            site_config_df=site_config_df,
            prediction_frame_template=training_data,
            tilt_set=[0, 1],
            horizon=2,
        )

        step_max = 3
        step_counter = 0
        done = False
        while not done and step_counter < step_max:
            # Sample a random action from the entire action space
            random_action = energy_savings_gym.action_space.sample()
            # Take the action and get the new observation space
            new_obs, reward, done, info = energy_savings_gym.step(random_action)
            step_counter += 1
            print(reward)

    def test_traffic_normalized_cco_metric(self):
        coverage_df = pd.DataFrame(
            data={
                "tile_x": [127894, 127895, 125894],
                "tile_y": [127187, 127188, 123187],
                LOC_X: [5.32754209119737, 5.32617473102958, -7.11090087890625],
                LOC_Y: [-4.36431884765625, -4.36294555664063, 10.7645084587186],
                "normalized_traffic_statistic": [0.2, 0.3, 0.5],
            }
        )
        # One pixel is weakly_covered and one is overly_covered
        coverage_df["weak_coverage"] = [1, 0, 0]
        coverage_df["over_coverage"] = [0, 1, 0]

        # traffic is equally split among weak_coverage and over_coverage
        coverage_df["normalized_traffic_statistic"] = [0.5, 0.5, 0]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 1
        self.assertTrue(value == expected_value)

        # traffic is all on one pixel (out of three) that is not weak or over
        coverage_df["normalized_traffic_statistic"] = [0, 0, 1]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 0
        self.assertTrue(value == expected_value)

        # traffic is equally split among one overcoverage pixel and one pixel which is good.
        coverage_df["normalized_traffic_statistic"] = [0, 0.5, 0.5]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 0.5
        self.assertTrue(value == expected_value)

        # two are weakly_covered and one is over_covered
        coverage_df["weak_coverage"] = [1, 0, 1]
        coverage_df["over_coverage"] = [0, 1, 0]

        # traffic is all on over_covered pixel.
        coverage_df["normalized_traffic_statistic"] = [0, 1, 0]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 1
        self.assertTrue(value == expected_value)

        # traffic is equally distributed in which two are weakly_covered and one is over_covered.
        coverage_df["normalized_traffic_statistic"] = [0.33, 0.33, 0.33]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 0.99
        self.assertTrue(value == expected_value)

        # all are over_covered pixels.
        coverage_df["weak_coverage"] = [0, 0, 0]
        coverage_df["over_coverage"] = [1, 1, 1]

        # Traffic is equally distributed over all the pixels.
        coverage_df["normalized_traffic_statistic"] = [0.33, 0.33, 0.33]
        value = CcoEngine.traffic_normalized_cco_metric(coverage_df)
        expected_value = 0.99
        self.assertTrue(value == expected_value)
