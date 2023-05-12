# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

# Application related
CELL_CARRIER_FREQ_MHZ = "cell_carrier_freq_mhz"
LATENT_BACKGROUND_NOISE_DB = -150

# Digital Twin releated
CELL_EL_DEG = "cell_el_deg"
CELL_ID = "cell_id"
CELL_LAT = "lat"
CELL_LON = "lon"
LOC_X = "loc_x"
LOC_Y = "loc_y"
RELATIVE_BEARING = "relative_bearing"
RSRP_DBM = "rsrp_dbm"
RXPOWER_DBM = "rxpower_dbm"
SINR_DB = "sinr_db"

# API related
MODEL_EXISTS = "exists"
MODEL_STATUS = "status"
MODEL_TRAINED = "trained"

# GIS tools related
RADIUS_EARTH_EQUATOR_KM = 6378.137
CIRC_KM_TO_DEG_LAT: float = 180.0 / (math.pi * RADIUS_EARTH_EQUATOR_KM)
