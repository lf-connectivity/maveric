# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The Simulation API handler.

This API starts a custom RIC simulation
"""

import logging
import os
from typing import Dict

from api_manager.config.kafka import kafka_producer_config
from api_manager.dtos.responses.simulation_response import SimulationResponse
from api_manager.preprocessors.simulation_request_preprocessor import (
    RICSimulationRequestPreprocessor,
)
from confluent_kafka import Producer

from radp.common import constants
from radp.common.helpers.file_system_helper import RADPFileSystemHelper
from radp.utility.kafka_utils import produce_object_to_kafka_topic

logger = logging.getLogger(__name__)


class SimulationHandler:
    """The API handler for starting a new RIC Simulation"""

    def __init__(self):
        """Initialize kafka producer"""
        self.producer = Producer(kafka_producer_config)

    def handle_simulation_request(self, request: Dict, files: Dict) -> Dict:
        """Handle simulation request"""

        # VALIDATION
        # TODO: implement simulation request format validation!
        # This should just make sure that correct syntax of request is passed in

        # TODO: implement deep validation of simulation event!
        # This validation will involve logical validation of simulation stages requested
        # and ensure that the request is serviceable within the simulation pipeline.
        # For example, check that the RF Digital Twin model referenced actually exists.

        # If the user has inputted their own data, check that the required columns are
        # present given their specified starting stage

        # If the user has inputted their own data, ensure they have supplied both a
        # ue data file and a config file

        # Set and check limit on the total tick count of simulation. Make sure
        # a user is not making too large of a simlulation request

        # PREPROCESSING

        # TODO: Calculate the number of batches and batch size to use, save these
        # amounts to the simulation event object

        processed_request = RICSimulationRequestPreprocessor.preprocess(
            request=request,
            files=files,
        )
        simulation_id = processed_request[constants.SIMULATION_ID]

        # create the simulation directory if it doesn't already exist
        simulation_directory = RADPFileSystemHelper.gen_simulation_directory(
            simulation_id
        )
        if not os.path.exists(simulation_directory):
            os.makedirs(simulation_directory)

        # write simulation metadata to file
        RADPFileSystemHelper.save_simulation_metadata(
            sim_metadata=processed_request, simulation_id=simulation_id
        )

        # save ue data and config to simulation directory if provided
        if constants.UE_DATA_FILE_PATH_KEY in files:
            RADPFileSystemHelper.save_simulation_ue_data(
                simulation_id, ue_data_file_path=files[constants.UE_DATA_FILE_PATH_KEY]
            )
        if constants.CONFIG_FILE_PATH_KEY in files:
            RADPFileSystemHelper.save_simulation_cell_config(
                simulation_id, config_file_path=files[constants.CONFIG_FILE_PATH_KEY]
            )

        # produce orchestration job to jobs topic
        orchestration_job = {
            constants.KAFKA_JOB_TYPE: constants.JOB_TYPE_ORCHESTRATION,
            constants.SIMULATION_ID: simulation_id,
        }

        job_id = produce_object_to_kafka_topic(
            producer=self.producer,
            topic=constants.KAFKA_JOBS_TOPIC_NAME,
            value=orchestration_job,
        )
        return SimulationResponse(
            job_id=job_id,
            simulation_id=simulation_id,
        ).to_dict()

    def _validate_request(self, model_id):
        """Validate a simulation request"""
        pass
