# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from apps.coverage_capacity_optimization.cco_engine import CcoEngine, CcoMetric
from radp.digital_twin.utils.constants import CELL_ID, LOC_X, LOC_Y  # noqa: E402


class TestCCO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_df = pd.DataFrame(data={CELL_ID: [1, 2, 73], LOC_X: [3, 4, 89], LOC_Y: [7, 8, 10]})

    def test_invalid_lambda(self):
        self.dummy_df["rsrp_dbm"] = [98, 92, 86]
        self.dummy_df["sinr_db"] = [12, -13, 90]
        self.assertRaises(
            ValueError,
            lambda: CcoEngine.rf_to_coverage_dataframe(
                self.dummy_df,
                weak_coverage_threshold=-100,
                over_coverage_threshold=0,
                lambda_=-1,
            ),
        )

    def testing_weakly_covered(self):
        # weakly_covered  := rsrp <= weak_coverage_threshold
        self.dummy_df["rsrp_dbm"] = [-98, -60, -100]
        self.dummy_df["sinr_db"] = [12, -13, 90]
        returned_df = CcoEngine.rf_to_coverage_dataframe(
            self.dummy_df, weak_coverage_threshold=-100, over_coverage_threshold=0
        )
        self.assertEqual(
            returned_df["weakly_covered"][returned_df["weakly_covered"] == 1].count() == 1,
            True,
        )

    def testing_all_weakly_covered(self):
        # weakly_covered  := rsrp <= weak_coverage_threshold
        self.dummy_df["rsrp_dbm"] = [-100, -100, -120]
        self.dummy_df["sinr_db"] = [12, -13, 90]
        returned_df = CcoEngine.rf_to_coverage_dataframe(
            self.dummy_df, weak_coverage_threshold=-100, over_coverage_threshold=0
        )
        self.assertEqual((returned_df["weakly_covered"] == 1).all(), True)

    def test_overly_covered(self):
        # overly_covered := not weakly_covered but sinr <= over_coverage_threshold
        self.dummy_df["rsrp_dbm"] = [-99, -98, -97]
        self.dummy_df["sinr_db"] = [-1, -1, 12]
        returned_df = CcoEngine.rf_to_coverage_dataframe(
            self.dummy_df, weak_coverage_threshold=-100, over_coverage_threshold=0
        )
        self.assertEqual(
            returned_df["overly_covered"][returned_df["overly_covered"] == 0].count() == 1,
            True,
        )

    def test_none_overly_covered(self):
        self.dummy_df["rsrp_dbm"] = [-10, -20, -30]
        self.dummy_df["sinr_db"] = [11, 96, 100]
        returned_df = CcoEngine.rf_to_coverage_dataframe(
            self.dummy_df, weak_coverage_threshold=-100, over_coverage_threshold=0
        )
        self.assertEqual((returned_df["overly_covered"] == 1).any(), False)

    def testing_some_not_weakly_or_overcovered(self):
        self.dummy_df["rsrp_dbm"] = [-110, -90, -95]
        self.dummy_df["sinr_db"] = [5, -3, 12]
        returned_df = CcoEngine.rf_to_coverage_dataframe(
            self.dummy_df, weak_coverage_threshold=-100, over_coverage_threshold=0
        )
        self.assertEqual(
            returned_df["covered"][returned_df["covered"] == 1].count() == 1,
            True,
        )
        self.assertEqual(
            returned_df["weakly_covered"][returned_df["weakly_covered"] == 1].count() == 1,
            True,
        )
        self.assertEqual(
            returned_df["overly_covered"][returned_df["overly_covered"] == 1].count() == 1,
            True,
        )

    def test_add_tile_x_and_tile_y(self):
        d = pd.DataFrame({LOC_X: [1, 2, 3], LOC_Y: [0, 2, 3]})
        df = CcoEngine.add_tile_x_and_tile_y(d)
        expected_df = pd.DataFrame(
            {
                LOC_X: [1, 2, 3],
                LOC_Y: [0, 2, 3],
                "tile_x": [131800, 132528, 133256],
                "tile_y": [131072, 129615, 128886],
            }
        )
        self.assertTrue(df.equals(expected_df))

    def test_augment_coverage_df_with_normalized_traffic_model(self):
        d = {
            "tile_x": [134951, 134950],
            "tile_y": [134253, 134252],
            "avg_of_average_egress_kbps_across_all_time": [
                0.2,
                0.3,
            ],
        }
        traffic_model_df = pd.DataFrame(data=d)
        d = {
            LOC_X: [5.32754209119737, 5.32617473102958],
            LOC_Y: [-4.36431884765625, -4.36294555664063],
            "weak_coverage": [0, 1],
            "over_coverage": [1, 0],
        }
        coverage_df = pd.DataFrame(data=d)
        df = CcoEngine.augment_coverage_df_with_normalized_traffic_model(
            traffic_model_df, "avg_of_average_egress_kbps_across_all_time", coverage_df
        )
        expected_df = pd.DataFrame(
            data={
                "tile_x": [134951, 134950],
                "tile_y": [134253, 134252],
                LOC_X: [5.32754209119737, 5.32617473102958],
                LOC_Y: [-4.36431884765625, -4.36294555664063],
                "normalized_traffic_statistic": [
                    0.4,
                    0.6,
                ],
                "weak_coverage": [0, 1],
                "over_coverage": [1, 0],
            }
        )
        self.assertTrue((df.round(6).equals(expected_df.round(6))))

    def test_get_cco_objective_value(self):
        d = {
            "tile_x": [134951, 134950],
            "tile_y": [134253, 134252],
            CELL_ID: [1, 2],
            "avg_of_average_egress_kbps_across_all_time": [
                0.2,
                0.3,
            ],
        }
        traffic_model_df = pd.DataFrame(data=d)
        d = {
            LOC_X: [5.32754209119737, 5.32617473102958],
            LOC_Y: [-4.36431884765625, -4.36294555664063],
            "weak_coverage": [0, 1],
            "over_coverage": [1, 0],
            CELL_ID: [1, 2],
            "network_coverage_utility": [0.6, 0.4],
        }
        coverage_df = pd.DataFrame(data=d)

        k = CcoEngine.get_cco_objective_value(
            coverage_df,
            [1, 2],
            CELL_ID,
            cco_metric=CcoMetric.PIXEL,
            traffic_model_df=traffic_model_df,
        )
        # asserting the multiplied version of network_coverage_utility
        self.assertTrue(
            coverage_df["network_coverage_utility"][0] == 0.24 and coverage_df["network_coverage_utility"][1] == 0.24
        )

        # asserting the average of network_coverage_utility
        self.assertTrue(0.24 == coverage_df["network_coverage_utility"].sum() / len(coverage_df))

        # asserting the returned cco_objective_value
        expected_value = 0.24
        self.assertTrue(k.round(2) == expected_value)
