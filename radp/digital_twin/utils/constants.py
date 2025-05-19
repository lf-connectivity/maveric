# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Final

RADIUS_EARTH_EQUATOR_KM = 6378.137
CIRC_KM_TO_DEG_LAT: float = 180.0 / (math.pi * RADIUS_EARTH_EQUATOR_KM)

LATENT_BACKGROUND_NOISE_DB = -150
RLF_THRESHOLD = -4
TXPWR_DBM = 23.0

ANTENNA_GAIN: Final[str] = "antenna_gain"
CELL_AZ_DEG: Final[str] = "cell_az_deg"
CELL_CARRIER_FREQ_MHZ: Final[str] = "cell_carrier_freq_mhz"
CELL_EL_DEG: Final[str] = "cell_el_deg"
CELL_ID: Final[str] = "cell_id"
CELL_LAT: Final[str] = "cell_lat"
CELL_LON: Final[str] = "cell_lon"
CELL_RXPWR_DBM: Final[str] = "cell_rxpwr_dbm"
CELL_TXPWR_DBM: Final[str] = "cell_txpwr_dbm"
HRX: Final[str] = "hRx"
HTX: Final[str] = "hTx"
LAT: Final[str] = "lat"
LOC_X: Final[str] = "loc_x"
LOC_Y: Final[str] = "loc_y"
LOG_DISTANCE: Final[str] = "log_distance"
LON: Final[str] = "lon"
RELATIVE_BEARING: Final[str] = "relative_bearing"
RELATIVE_TILT_SQUARED: Final[str] = "relative_tilt_squared"
RELATIVE_TILT: Final[str] = "relative_tilt"
RSRP_DBM: Final[str] = "rsrp_dbm"
RXPOWER_DBM: Final[str] = "rxpower_dbm"
RXPOWER_STDDEV_DBM: Final[str] = "rxpower_stddev_dbm"
SIM_IDX: Final[str] = "sim_idx"
SINR_DB: Final[str] = "sinr_db"
