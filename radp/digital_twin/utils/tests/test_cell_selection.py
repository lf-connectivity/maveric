# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

"""

# Run tests

On command line (ensure that you are two directories above, i.e. in the directory
that contains `digital_twin` directory) :

        python3 -m unittest discover
or
        python3 -m unittest digital_twin/utils/tests/test_cell_selection.py

"""

import unittest
from typing import Dict, List, Tuple

import pandas as pd

from radp.digital_twin.utils import constants
from radp.digital_twin.utils.cell_selection import get_rsrp_dbm_sinr_db_by_layer, perform_attachment


class TestCellSelection(unittest.TestCase):
    def test_get_rsrp_dbm_sinr_db_by_layer(self):
        rx_dbm = 5
        freq = 2100

        # 1. 1 layer, 2 equal powered cells
        rx_powers_by_layer: Dict[float, List[Tuple[str, float]]] = {freq: [("A", rx_dbm), ("B", rx_dbm)]}
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(rx_powers_by_layer)
        self.assertEqual(len(rsrp_dbm_by_layer), 1)  # 1 layer
        self.assertEqual(len(sinr_db_by_layer), 1)  # 1 layer
        self.assertTrue(freq in rsrp_dbm_by_layer)  # layer unchanged
        self.assertTrue(freq in sinr_db_by_layer)  # layer unchanged
        self.assertEqual(rsrp_dbm_by_layer[freq][1], rx_dbm)  # rsrp is the same value
        self.assertAlmostEqual(sinr_db_by_layer[freq][1], 0)  # SINR is very close to 0

        # 2. 1 layer, one cell is twice the other in dbm scale
        rx_powers_by_layer: Dict[float, List[Tuple[str, float]]] = {freq: [("A", rx_dbm), ("B", 2 * rx_dbm)]}
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(rx_powers_by_layer)
        self.assertEqual(len(rsrp_dbm_by_layer), 1)  # 1 layer
        self.assertEqual(len(sinr_db_by_layer), 1)  # 1 layer
        self.assertTrue(freq in rsrp_dbm_by_layer)  # layer unchanged
        self.assertTrue(freq in sinr_db_by_layer)  # layer unchanged
        self.assertEqual(rsrp_dbm_by_layer[freq][0], "B")  # bigger one wins
        self.assertEqual(rsrp_dbm_by_layer[freq][1], 2 * rx_dbm)  # bigger one wins
        self.assertAlmostEqual(sinr_db_by_layer[freq][0], "B")  # SINR winner is same
        self.assertAlmostEqual(sinr_db_by_layer[freq][1], rx_dbm)  # SINR is difference between bigger and smaller

        # 3. 2 layers, second cell 3x stronger for layer 1, first cell 3x stronger for layer 2
        rx_powers_by_layer: Dict[float, List[Tuple[str, float]]] = {
            freq: [("A", rx_dbm), ("B", 3 * rx_dbm)],
            freq * 2: [("A2", 3 * rx_dbm), ("B2", rx_dbm)],
        }
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(rx_powers_by_layer)
        self.assertEqual(len(rsrp_dbm_by_layer), 2)  # 2 layers
        self.assertEqual(len(sinr_db_by_layer), 2)  # 2 layers
        # layers unchanged
        self.assertTrue(
            all(f in rsrp_dbm_by_layer for f in [freq, freq * 2])
            and all(f in sinr_db_by_layer for f in [freq, freq * 2])
        )
        self.assertEqual(rsrp_dbm_by_layer[freq][0], "B")  # bigger one wins
        self.assertEqual(rsrp_dbm_by_layer[freq][1], 3 * rx_dbm)  # bigger one wins
        self.assertAlmostEqual(sinr_db_by_layer[freq][1], 2 * rx_dbm)  # SINR is difference between bigger and smaller
        self.assertAlmostEqual(rsrp_dbm_by_layer[freq * 2][0], "A2")  # bigger one wins
        self.assertEqual(rsrp_dbm_by_layer[freq * 2][1], 3 * rx_dbm)  # bigger one wins
        self.assertAlmostEqual(
            sinr_db_by_layer[freq * 2][1], 2 * rx_dbm
        )  # SINR is difference between bigger and smaller

    def test_perform_attachment(self):
        """Scenario : 2 pixel map, 4 cells on two different carriers.

        In the first pixel (0, 0), the high SINR cell B on layer 1 gets attached, instead of the
        high raw Rx power cell C, but which has low SINR, on layer 2.

        In the second pixel (1, 1), the high SINR cell D on layer 2 gets attached,
        and it also happens to be a highest raw Rx power cell. Another equally high
        raw Rx power cell (cell B on layer 1) loses out since its SINR is low.
        """

        rx_dbm = 5.0

        freq_layer1 = 2100
        freq_layer2 = 1750
        freq_layer3 = 1550

        topo_df_columns = [
            constants.CELL_ID,
            constants.CELL_CARRIER_FREQ_MHZ,
        ]

        topology_df = pd.DataFrame(
            [
                [
                    "A",
                    freq_layer1,
                ],
                [
                    "B",
                    freq_layer1,
                ],
                [
                    "C",
                    freq_layer2,
                ],
                [
                    "D",
                    freq_layer2,
                ],
                [
                    "E",
                    freq_layer3,
                ],
            ],
            columns=topo_df_columns,
        )

        prediction_dfs_columns = [
            constants.LOC_X,
            constants.LOC_Y,
            constants.CELL_ID,
            constants.RXPOWER_DBM,
        ]

        prediction_dfs = pd.DataFrame(
            [
                [
                    0,
                    0,
                    "A",
                    rx_dbm,
                ],
                [
                    1,
                    1,
                    "A",
                    2 * rx_dbm,
                ],
                [
                    0,
                    0,
                    "B",
                    3 * rx_dbm,
                ],
                [
                    1,
                    1,
                    "B",
                    5 * rx_dbm,
                ],
                [
                    0,
                    0,
                    "C",
                    5 * rx_dbm,
                ],
                [
                    1,
                    1,
                    "C",
                    rx_dbm,
                ],
                [
                    0,
                    0,
                    "D",
                    4 * rx_dbm,
                ],
                [
                    1,
                    1,
                    "D",
                    5 * rx_dbm,
                ],
                [
                    2,
                    2,
                    "E",
                    4 * rx_dbm,
                ],
            ],
            columns=prediction_dfs_columns,
        )

        rf_dataframe_expected = pd.DataFrame(
            [
                [
                    0,
                    0,
                    "B",
                    2 * rx_dbm,
                    3 * rx_dbm,
                ],
                [
                    1,
                    1,
                    "D",
                    4 * rx_dbm,
                    5 * rx_dbm,
                ],
                [
                    2,
                    2,
                    "E",
                    150 + 4 * rx_dbm,
                    4 * rx_dbm,
                ],
            ],
            columns=[
                "loc_x",
                "loc_y",
                constants.CELL_ID,
                "sinr_db",
                "rsrp_dbm",
            ],
        )

        rf_dataframe = perform_attachment(prediction_dfs, topology_df)

        pd.testing.assert_frame_equal(rf_dataframe, rf_dataframe_expected)
