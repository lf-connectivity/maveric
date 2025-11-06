# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from typing import Dict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from radp.client.client import RADPClient  # noqa: E402
from radp.client.helper import ModelStatus, RADPHelper, SimulationStatus  # noqa: E402

""" ----- APPLICATION SETUP ----- """
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
# get example app absolute path
data_path = os.path.join(SCRIPT_ROOT, "../data")

# Set input files
TOPOLOGY_FILE = os.path.join(data_path, "topology.csv")
TRAINING_DATA_FILES = [os.path.join(data_path, "ue_training_data.csv")]
PREDICTION_DATA_FILES = [os.path.join(data_path, "ue_data.csv")]
PREDICTION_CONFIG = os.path.join(data_path, "config.csv")
MODEL_ID = "happy_case_rf_prediction_model"
SIMULATION_EVENT_FILE = os.path.join(data_path, "example_sim_1.json")

# NOTE: specify optional training parameters here. See README for info on these parameters
TRAINING_PARAMS: Dict = {}


def happy_case__rf_prediction():
    """End to end test for RF prediction only

    This integration test runs the following steps:
    1. train a new model
    2. run simulation including only RF prediction on a small dataset
    3. output results
    """
    # Load input data
    topology = pd.read_csv(TOPOLOGY_FILE)
    training_data = pd.concat([pd.read_csv(file) for file in TRAINING_DATA_FILES])

    prediction_data = pd.concat([pd.read_csv(file) for file in PREDICTION_DATA_FILES])
    prediction_config = pd.read_csv(PREDICTION_CONFIG)

    # NOTE: you may provide a different ip or port but these values must
    # match those of your running RADP service
    radp_client = RADPClient()
    radp_helper = RADPHelper(radp_client)

    """ ----- TRAINING A DIGITAL TWIN ----- """
    # call train API
    radp_client.train(
        model_id=MODEL_ID,
        params=TRAINING_PARAMS,
        ue_training_data=training_data,
        topology=topology,
    )

    # use RADP helper to resolve the status of model training
    # optionally you can also describe the model status directly
    # using radp_client.describe_model
    model_status: ModelStatus = radp_helper.resolve_model_status(
        MODEL_ID, wait_interval=3, max_attempts=10, verbose=True
    )

    if not model_status.success:
        # NOTE: add your custom error-handling logic here
        sys.exit(1)

    """ ----- RUN AN EXAMPLE SIMULATION ----- """
    with open(SIMULATION_EVENT_FILE, "r") as file:
        simulation_event = json.load(file)

    simulation_response = radp_client.simulation(
        simulation_event=simulation_event,
        ue_data=prediction_data,
        config=prediction_config,
    )
    simulation_id = simulation_response["simulation_id"]

    # use RADP helper to resolve the status of the simulation
    # optionally you can instead describe the simulation status directly
    # using radp_client.describe_simulation
    simulation_status: SimulationStatus = radp_helper.resolve_simulation_status(
        simulation_id, wait_interval=3, max_attempts=10, verbose=True
    )

    if not simulation_status.success:
        print("Error occurred when running the simulation!")
        sys.exit(1)

    """ ----- OUTPUT RESULTS ----- """
    # download results to pandas dataframe
    rf_dataframe = radp_client.consume_simulation_output(simulation_id)

    assert len(rf_dataframe) == 45

    expected_rf_df_cols = [
        "cell_id",
        "rxpower_dbm",
        "lon",
        "lat",
        "mock_ue_id",
        "tick",
    ]
    assert set(expected_rf_df_cols).issubset(rf_dataframe.columns)
