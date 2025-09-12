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
PREDICTION_CONFIG = os.path.join(data_path, "config.csv")
MODEL_ID = "happy_case_ue_track_gen_rf_prediction_model"
SIMULATION_EVENT_FILE = os.path.join(data_path, "example_sim_2.json")

# NOTE: specify optional training parameters here. See README for info on these parameters
TRAINING_PARAMS: Dict = {}


def happy_case__ue_track_gen_rf_prediction():
    """End to end test to run ue tracks generation and rf prediction

    This integration test runs the following steps:
    1. train a model
    2. run simulation with ue track generation + rf prediction
    3. consume output
    """

    # Load input data
    topology = pd.read_csv(TOPOLOGY_FILE)
    training_data = pd.concat([pd.read_csv(file) for file in TRAINING_DATA_FILES])
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

    # Enhanced validation of results
    assert len(rf_dataframe) == 6000, f"Expected 6000 rows, got {len(rf_dataframe)}"

    expected_rf_df_cols = [
        "cell_id",
        "rxpower_dbm",
        "lon", 
        "lat",
        "mock_ue_id",
        "tick",
    ]
    assert set(expected_rf_df_cols).issubset(rf_dataframe.columns), \
        f"Missing columns: {set(expected_rf_df_cols) - set(rf_dataframe.columns)}"
    
    # Validate data ranges and consistency
    assert rf_dataframe["rxpower_dbm"].min() >= -200, "RSRP values too low"
    assert rf_dataframe["rxpower_dbm"].max() <= 0, "RSRP values too high"
    assert rf_dataframe["lat"].between(-90, 90).all(), "Invalid latitude values"
    assert rf_dataframe["lon"].between(-180, 180).all(), "Invalid longitude values"
    assert rf_dataframe["tick"].min() >= 0, "Negative tick values found"
    
    # Validate UE tracking consistency
    ue_count = rf_dataframe["mock_ue_id"].nunique()
    expected_ue_count = 20  # 5 + 5 + 5 + 5 from simulation config
    assert ue_count == expected_ue_count, f"Expected {expected_ue_count} UEs, found {ue_count}"
    
    # Validate temporal consistency
    ticks_per_ue = rf_dataframe.groupby("mock_ue_id")["tick"].nunique()
    expected_ticks = 100  # 10 seconds / 0.1 interval
    assert (ticks_per_ue == expected_ticks).all(), "Inconsistent tick counts across UEs"
    
    print(f"✅ UE track generation + RF prediction completed successfully")
    print(f"   Generated {len(rf_dataframe)} RF predictions for {ue_count} UEs over {rf_dataframe['tick'].max() + 1} ticks")
    print(f"   RSRP range: {rf_dataframe['rxpower_dbm'].min():.1f} to {rf_dataframe['rxpower_dbm'].max():.1f} dBm")
    print(f"   UE mobility range: lat {rf_dataframe['lat'].min():.3f}-{rf_dataframe['lat'].max():.3f}, "
          f"lon {rf_dataframe['lon'].min():.3f}-{rf_dataframe['lon'].max():.3f}")
    print(f"   Temporal coverage: {ticks_per_ue.iloc[0]} ticks per UE")
