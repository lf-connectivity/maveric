# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Dict

import pandas as pd
from dotenv import load_dotenv

from apps.coverage_capacity_optimization.dgpco_cco import DgpcoCCO

load_dotenv()

from radp.client.client import RADPClient  # noqa: E402
from radp.client.helper import ModelStatus, RADPHelper  # noqa: E402

""" ----- APPLICATION SETUP ----- """
# NOTE: Download the anp_sim_data files from drive:
# https://drive.google.com/drive/u/1/folders/1gOS0R9XfS8yt5Al3TzqndEApVlsCzawl
# Set input files

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
# get example app absolute path
data_path = os.path.join(SCRIPT_ROOT, "../anp_sim_data")

TOPOLOGY_FILE = os.path.join(data_path, "topology.csv")
TRAINING_DATA_FILES = [
    os.path.join(data_path, "sim_001", "full_data.csv"),
    # os.path.join(data_path, "sim_002", "full_data.csv"),
    # os.path.join(data_path, "sim_003", "full_data.csv"),
    # os.path.join(data_path, "sim_004", "full_data.csv"),
    # os.path.join(data_path, "sim_005", "full_data.csv"),
]
PREDICTION_DATA_FILES = [
    os.path.join(data_path, "sim_004", "full_data.csv"),
]
PREDICTION_CONFIG = os.path.join(data_path, "starting_config.csv")
MODEL_ID = "cco_anp_test_model"
TRAINING_PARAMS: Dict = {}

# Load input data
topology = pd.read_csv(TOPOLOGY_FILE)
training_data = pd.concat([pd.read_csv(file) for file in TRAINING_DATA_FILES])

prediction_data = pd.concat([pd.read_csv(file) for file in PREDICTION_DATA_FILES])
prediction_config = pd.read_csv(PREDICTION_CONFIG)

# instantiate RADP client
radp_client = RADPClient()
radp_helper = RADPHelper(radp_client)


""" ----- TRAINING A DIGITAL TWIN ----- """
# call train API
train_response = radp_client.train(
    model_id=MODEL_ID,
    params=TRAINING_PARAMS,
    ue_training_data=training_data,
    topology=topology,
)

# resolve the model status -- this blocking call ensures training is done and model is available for use
model_status: ModelStatus = radp_helper.resolve_model_status(
    MODEL_ID, wait_interval=10, max_attempts=100, verbose=False
)

# handle an exception if one occurred
if not model_status.success:
    print(f"Error occurred training the model: {model_status.error_message}")
    sys.exit(1)


""" ----- RUNNING DGPCO CCO ALGORITHM ----- """
# set the valid config values
VALID_CONFIGURATION_VALUES = {
    "cell_el_deg": [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
    ]
}

# instantiate the dGPCO CCO class
dgpco_cco = DgpcoCCO(
    topology=topology,
    valid_configuration_values=VALID_CONFIGURATION_VALUES,
    bayesian_digital_twin_id=MODEL_ID,
    ue_data=prediction_data,
    config=prediction_config,
)

# run the dGPCO CCO algorithm
(
    rf_dataframe_per_epoch,
    coverage_dataframe_per_epoch,
    cco_objective_per_epoch,
    opt_per_epoch,
) = dgpco_cco.run(num_epochs=20)


""" ----- OUTPUT RESULTS ----- """
print("\n ----- rf_dataframe_per_epoch ----- ")
print(rf_dataframe_per_epoch)
print("\n ----- coverage_dataframe_per_epoch ----- ")
print(coverage_dataframe_per_epoch)
print("\n ----- cco_objective_per_epoch ----- ")
print(cco_objective_per_epoch)
print("\n ----- opt_per_epoch ----- ")
print(opt_per_epoch)
