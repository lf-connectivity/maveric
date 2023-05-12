# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from confluent_kafka import Producer

from radp.common import constants
from radp.common.enums import OutputStatus
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.utility.kafka_utils import produce_object_to_kafka_topic
from radp.utility.pandas_utils import cross_replicate, read_feather_df, write_feather_df
from rf_prediction.config.kafka import kafka_producer_config
from rf_prediction.rf_prediction_helper import RFPredictionHelper

logger = logging.getLogger(__name__)


class RFPredictionDriver:
    """
    The RFPredictionDriver Class handles execution of RF Prediction jobs

    Input/Output:
    The RF Prediction service will take in as input an RF Prediction job
    with the following format:
        {
            "job_id": ""
            "job_type": "rf_prediction",
            "simulation_id": "123",
            "batch": 3,
            "rf_prediction": {
                "ue_data_file_path": "",
                "output_file_path": "",
                "model_id": "",
                "model_file": "",
                "config_file_path": "",
                "topology_file_path": "",
                "params": {},
            }
        }

    The RF Prediction service will output the following:
    - write output file to "output_file" with per-cell pixel Rx Power added
    - produce "output" event to the "outputs" topic signaling success/failure

    The RFPrediction class will handle two responsibilities:
    1. Pre-processing of UE data
    2. RF Prediction on data using RF Digital Twin model

    The pre-processing of UE data involves the following:
    - analyzing input UE data
    - cross-multiplication of UE data with config data if necessary
    - appending of config and topology data to dataframe

    Cross-multiplication only needs to occur if the input UE data does not
    already have a cell_id column in place

    RF Prediction involves taking preprocessed data and running prediction using the trained
    BayesianDigitalTwin object corresponding to each row's specific cell_id.
    """

    def __init__(self):
        """Set up the orchestrator's producer instance"""
        self.producer = Producer(kafka_producer_config)

    def handle_rf_prediction_job(self, job_data):
        """Top-level handler for RF Prediction job"""

        logger.info(f"Handling RF Prediction job: {job_data}")

        # pull the model
        model_id, model_file_path = RFPredictionHelper.get_model_parameters(job_data)
        
        job_id = job_data[constants.JOB_ID]

        logger.info(f"Running RF prediction using model: {model_id}")

        # pull the input file path and config file path
        ue_data_file_path = RFPredictionHelper.get_ue_data_file_path(job_data)
        config_file_path = RFPredictionHelper.get_config_file_path(job_data)
        topology_file_path = RFPredictionHelper.get_topology_file_path(job_data)

        # load input data frames
        ue_data_df = read_feather_df(ue_data_file_path)
        config_df = read_feather_df(config_file_path)
        topology_df = read_feather_df(topology_file_path)

        # determine whether cross-multiplication needs to occur
        cross_replications_required = constants.CELL_ID not in ue_data_df

        # run data preprocessing steps
        ue_data_df, cell_id_ue_data_map = self._preprocess_ue_data(
            ue_data_df=ue_data_df,
            config_df=config_df,
            topology_df=topology_df,
            cross_replications_required=cross_replications_required,
        )

        # load model map
        bayesian_digital_twin_map = BayesianDigitalTwin.load_model_map_from_pickle(model_file_path=model_file_path)
        logger.debug(f"Loaded bayesian digital twin model {model_id} from '{model_file_path}'")

        # run per-cell prediction
        prediction_output_map = self._run_inference_per_cell(cell_id_ue_data_map, bayesian_digital_twin_map)

        # attach per-cell rxpower_dbm to ue_data_df
        rx_powers = []
        for cell_id in config_df[constants.CELL_ID]:
            _, _, rf_dataframe = prediction_output_map[cell_id]
            rx_powers.extend(rf_dataframe[constants.RXPOWER_DBM])

        # get the expected RF Prediction output data
        # TODO: Reorder to first pull then insert rx_powers
        ue_data_df.insert(loc=len(ue_data_df.columns), column=constants.RXPOWER_DBM, value=rx_powers)
        rf_prediction_output = ue_data_df.loc[
            :,
            [
                constants.CELL_ID,
                constants.RXPOWER_DBM,
                constants.LONGITUDE,
                constants.LATITUDE,
            ],
        ]

        # if mock_ue_id provided, append to output
        if constants.MOCK_UE_ID in ue_data_df:
            rf_prediction_output[constants.MOCK_UE_ID] = ue_data_df[constants.MOCK_UE_ID]

        # if tick provided, append to output
        if constants.TICK in ue_data_df:
            rf_prediction_output[constants.TICK] = ue_data_df[constants.TICK]

        # save output to file
        output_file_path = RFPredictionHelper.get_output_file_path(job_data)
        write_feather_df(output_file_path, rf_prediction_output)
        logger.info(f"Saved RF prediction output DF to {output_file_path}")

        # pull simulation/batch info from job
        simulation_id = RFPredictionHelper.get_simulation_id(job_data)
        batch = RFPredictionHelper.get_batch_index(job_data)

        # produce output event to outputs topic
        output_event = {
            constants.SIMULATION_ID: simulation_id,
            constants.SERVICE: constants.RF_PREDICTION,
            constants.BATCH: batch,
            constants.STATUS: OutputStatus.SUCCESS.value,
        }
        produce_object_to_kafka_topic(self.producer, topic=constants.OUTPUTS, value=output_event)
        logger.info(f"Produced successful output event to topic: {output_event}")

    def _preprocess_ue_data(
        self,
        ue_data_df: pd.DataFrame,
        config_df: pd.DataFrame,
        topology_df: pd.DataFrame,
        cross_replications_required: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Preprocess RF Prediction input data

        If the cross_replications_required flag is set to true, the input
        data is assumed to be only UE Data without any cell id column

        In this case, all rows will be replicated by the number of cells
        in config, with each cell getting a copy of the entire
        DF
        """

        logger.info("Preprocessing UE data...")

        # perform cross replication if required
        if cross_replications_required:
            logger.info("No cell_id column found in UE data, running cross replication...")

            cell_ids = pd.DataFrame(config_df[constants.CELL_ID])
            ue_data_df = cross_replicate(ue_data_df, cell_ids)

            logger.info("Finished running cross replication!")

        # run Bayesian digital twin preprocessing
        cell_id_ue_data_map: Dict[str, pd.DataFrame] = BayesianDigitalTwin.preprocess_ue_prediction_data(
            ue_data_df=ue_data_df,
            config_df=config_df,
            topology_df=topology_df,
        )
        logger.info("Finished preprocessing UE data!")
        return ue_data_df, cell_id_ue_data_map

    def _run_inference_per_cell(
        self,
        cell_id_ue_data_map: pd.DataFrame,
        bayesian_digital_twin_map: Dict[str, BayesianDigitalTwin],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """Run inference for cell df using the respective model for that cell"""

        logger.info("Running per-cell inference on UE data...")
        prediction_output_map = {}

        for cell_id, ue_prediction_data in cell_id_ue_data_map.items():
            logger.info(f"Running inference for cell_id: {cell_id}...")
            # run prediction
            pred_means, pred_std = bayesian_digital_twin_map[cell_id].predict_distributed_gpmodel(
                prediction_dfs=[ue_prediction_data]
            )

            # store to output map
            prediction_output_map[cell_id] = (pred_means, pred_std, ue_prediction_data)
        logger.info("Finished running per-cell inference on UE data!")
        return prediction_output_map
