# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Default logic for cell selection, camping and attachment.
"""

from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from radp.digital_twin.utils import constants


def perform_attachment(
    ue_prediction_data: pd.DataFrame,
    topology: pd.DataFrame,
) -> pd.DataFrame:
    """This method looks at Rx power (predictions) from cells to pixels,
    for a given snapshot in time (e.g. simulation tick),
    and performs cell selection (camping or attachment).

    `ue_prediction_data` is a dataframe which contains at least the
    following columns:
        lon,            # pixel longitude or similar
        lat,            # pixel latitude or similar
        cell_id,        # cell ID for a cell
        rxpower_dbm,    # (predicted) Rx power from given cell to given location

    `topology` is a dataframe which contains at least the following columns:
        cell_id:                # cell ID for a cell
        cell_carrier_freq_mhz:  # carrier frequency (in MHz) for given cell

    This method returns a single dataframe back that represents
    the result of cell selection. The following columns are present in the result :
        loc_x,      # pixel longitude or similar
        loc_y,      # pixel latitude or similar
        cell_id,    # cell ID of selected cell
        rsrp_dbm,   # (predicted) Rx power of selected cell
        sinr_db,    # (predicted) SINR, that accounts for
                    # interference from other cells on the same layer (carrier frequency)
    """

    # initiate a dictionary to store power-by-layer dictionaries on a per-pixel basis
    rx_powers_by_layer_by_loc: Dict[
        Tuple[float, float], Dict[float, List[Tuple[Any, float]]]
    ] = defaultdict(lambda: defaultdict(list))

    # pull per-cell frequencies for faster lookup
    cell_id_to_freq = {
        row.cell_id: row.cell_carrier_freq_mhz for _, row in topology.iterrows()
    }

    # iterate over ue_prediction_data, to
    # build rx_powers_by_layer_by_loc map
    for _, row in ue_prediction_data.iterrows():
        # pull cell carrier frequency
        cell_carrier_freq_mhz = cell_id_to_freq[row.cell_id]

        # fetch the pixel longitude and latitude
        loc_x = row.get(constants.LOC_X, row.get(constants.LON))
        loc_y = row.get(constants.LOC_Y, row.get(constants.LAT))
        if loc_x is None or loc_y is None:
            raise Exception("loc_x or loc_y cannot be found in the dataset")

        # add (cell_id, rxpower) tuple on a per-row, per-freq basis
        rx_powers_by_layer_by_loc[(loc_x, loc_y)][cell_carrier_freq_mhz].append(
            (row.cell_id, row.rxpower_dbm)
        )

    # perform cell selection per location
    rf_dataframe_dict = defaultdict(list)

    for loc, rx_powers_by_layer in rx_powers_by_layer_by_loc.items():
        # compute strongest server, interference and SINR
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(
            rx_powers_by_layer
        )

        # pull sinr_db, cell_id and rsrp_dbm based on highest SINR
        max_sinr_db_item = max(sinr_db_by_layer.items(), key=lambda k: k[1][1])
        max_sinr_db_cell_id, max_sinr_db = max_sinr_db_item[1]
        rsrp_dbm = next(
            v[1] for v in rsrp_dbm_by_layer.values() if v[0] == max_sinr_db_cell_id
        )

        # update rf_dataframe output
        rf_dataframe_dict[constants.LOC_X].append(loc[0])
        rf_dataframe_dict[constants.LOC_Y].append(loc[1])
        rf_dataframe_dict[constants.CELL_ID].append(max_sinr_db_cell_id)
        rf_dataframe_dict[constants.SINR_DB].append(max_sinr_db)
        rf_dataframe_dict[constants.RSRP_DBM].append(rsrp_dbm)

    # return as dataframe
    return pd.DataFrame(rf_dataframe_dict)


def get_rsrp_dbm_sinr_db_by_layer(
    rx_powers_by_layer: Dict[float, List[Tuple[str, float]]],
) -> Tuple[Dict[float, Tuple[str, float]], Dict[float, Tuple[str, float]]]:
    """Given (predicted) Rx powers per layer, determine
    RSRP (served by max power) and SINR, within layer.
    """

    rsrp_dbm_by_layer: Dict[float, Tuple[str, float]] = {}
    sinr_db_by_layer: Dict[float, Tuple[str, float]] = {}

    # for each layer, compute strongest server, interference and SINR
    for cell_carrier_freq_mhz, rx_powers in rx_powers_by_layer.items():
        # get the max rsrp from the layer
        rsrp_dbm_by_layer[cell_carrier_freq_mhz] = max(rx_powers, key=itemgetter(1))

        # calculate background noise
        pred_noise = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # calculate prediction interference by subtracting
        # the strongest rx power from total sum of rx powers
        pred_interference = sum(10 ** (p / 10) for i, p in rx_powers) - (
            10 ** (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] / 10)
        )
        pred_interference_noise_dBm = 10 * np.log10(pred_interference + pred_noise)
        sinr_db_by_layer[cell_carrier_freq_mhz] = (
            rsrp_dbm_by_layer[cell_carrier_freq_mhz][0],
            (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] - pred_interference_noise_dBm),
        )

    return rsrp_dbm_by_layer, sinr_db_by_layer
