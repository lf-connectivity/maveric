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


SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
# get example app absolute path
data_path = os.path.join(SCRIPT_ROOT, "data")

# set input files
MODEL_ID = "example_test_model"
TOPOLOGY_FILE = os.path.join(data_path, "topology.csv")
TRAINING_DATA_FILES = [
    os.path.join(data_path, "ue_training_data_1.csv"),
    os.path.join(data_path, "ue_training_data_2.csv"),
]
PREDICTION_DATA_FILES = [
    os.path.join(data_path, "ue_data_1.csv"),
    os.path.join(data_path, "ue_data_2.csv"),
]
PREDICTION_CONFIG = os.path.join(data_path, "config.csv")
SIMULATION_EVENT_FILE = os.path.join(data_path, "example_sim_data.json")
SIMULATION_EVENT_FILES = [
    os.path.join(data_path, "example_sim_data_1_1.json"),
    os.path.join(data_path, "example_sim_data_1_2.json"),
]


# NOTE: toggle to demonstrate incremental model update approach vs precombining all input data
# the Bayesian Engine is flexible enough that you could also dynamically alter the topology
#   files, to add entirely new cell ids to the mix
#   although this demo does not do that
INCREMENTAL_UPDATE = True


# NOTE: specify optional training parameters here. See README for info on these parameters
# these are arguments passed all the way down into gpytorch
TRAINING_PARAMS: Dict = {}


# NOTE: you may provide a different ip or port but these values must
# match those of your running RADP service
radp_client = RADPClient()
radp_helper = RADPHelper(radp_client)


if INCREMENTAL_UPDATE:
    # call train API with some data
    train_response = radp_client.train(
        model_id=MODEL_ID,
        params=TRAINING_PARAMS,
        ue_training_data=pd.read_csv(TRAINING_DATA_FILES[0]),
        topology=pd.read_csv(TOPOLOGY_FILE),
    )

    model_status: ModelStatus = radp_helper.resolve_model_status(
        model_id=train_response["model_id"],
        wait_interval=3,
        max_attempts=10,
        verbose=True,
        job_id=train_response["job_id"],
    )

    if not model_status.success:
        # NOTE: add your custom error-handling logic here
        print("Error occurred training the model!")
        sys.exit(1)

    # run a simulation on the data so far
    with open(SIMULATION_EVENT_FILES[0], "r") as file:
        simulation_event = json.load(file)

    simulation_response = radp_client.simulation(
        simulation_event=simulation_event,
        ue_data=pd.read_csv(PREDICTION_DATA_FILES[0]),
        config=pd.read_csv(PREDICTION_CONFIG),
    )
    simulation_id = simulation_response["simulation_id"]

    # use RADP helper to resolve the status of the simulation
    # optionally you can instead describe the simulation status directly
    # using radp_client.describe_simulation
    simulation_status: SimulationStatus = radp_helper.resolve_simulation_status(
        simulation_id,
        wait_interval=3,
        max_attempts=10,
        verbose=True,
        job_id=simulation_response["job_id"],
    )

    if not simulation_status.success:
        # NOTE: add your custom error-handling logic here
        print("Error occurred when running the simulation!")
        sys.exit(1)

    # call train API with updated data
    train_response = radp_client.train(
        model_id=MODEL_ID,
        params=TRAINING_PARAMS,
        ue_training_data=pd.read_csv(TRAINING_DATA_FILES[1]),
        topology=pd.read_csv(TOPOLOGY_FILE),
        model_update=True,
    )

    model_status: ModelStatus = radp_helper.resolve_model_status(
        model_id=train_response["model_id"],
        wait_interval=3,
        max_attempts=10,
        verbose=True,
        job_id=train_response["job_id"],
    )

    if not model_status.success:
        # NOTE: add your custom error-handling logic here
        print("Error occurred updating the trained model!")
        sys.exit(1)

    with open(SIMULATION_EVENT_FILES[1], "r") as file:
        simulation_event = json.load(file)
else:
    # call train API with all the data at once
    train_response = radp_client.train(
        model_id=MODEL_ID,
        params=TRAINING_PARAMS,
        ue_training_data=pd.concat(
            [pd.read_csv(file) for file in TRAINING_DATA_FILES]
        ),
        topology=pd.read_csv(TOPOLOGY_FILE),
    )

    # use RADP helper to resolve the status of model training
    # optionally you can also describe the model status directly
    # using radp_client.describe_model
    model_status: ModelStatus = radp_helper.resolve_model_status(
        model_id=train_response["model_id"],
        wait_interval=3,
        max_attempts=10,
        verbose=True,
        job_id=train_response["job_id"],
    )

    if not model_status.success:
        # NOTE: add your custom error-handling logic here
        print("Error occurred training the model!")
        sys.exit(1)

    with open(SIMULATION_EVENT_FILE, "r") as file:
        simulation_event = json.load(file)

# run simulation on cumulative data passed to model
simulation_response = radp_client.simulation(
    simulation_event=simulation_event,
    ue_data=pd.concat(
        [pd.read_csv(file) for file in PREDICTION_DATA_FILES]
    ),
    config=pd.read_csv(PREDICTION_CONFIG),
)
simulation_id = simulation_response["simulation_id"]

# use RADP helper to resolve the status of the simulation
# optionally you can instead describe the simulation status directly
# using radp_client.describe_simulation
simulation_status: SimulationStatus = radp_helper.resolve_simulation_status(
    simulation_id,
    wait_interval=3,
    max_attempts=10,
    verbose=True,
    job_id=simulation_response["job_id"],
)

if not simulation_status.success:
    # NOTE: add your custom error-handling logic here
    print("Error occurred when running the simulation!")
    sys.exit(1)


# download results to pandas dataframe
rf_dataframe = radp_client.consume_simulation_output(simulation_id)

print(f"rf_dataframe: {rf_dataframe}")
