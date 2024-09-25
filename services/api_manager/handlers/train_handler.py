# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import pandas as pd
from api_manager.config.kafka import kafka_producer_config
from api_manager.dtos.requests.train_request import TrainRequest
from api_manager.dtos.responses.train_response import TrainResponse
from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from confluent_kafka import Producer

from radp.common import constants
from radp.common.enums import ModelStatus, ModelType
from radp.common.helpers.file_system_helper import RADPFileSystemHelper
from radp.utility.kafka_utils import produce_object_to_kafka_topic
from radp.utility.pandas_utils import write_feather_df

logger = logging.getLogger(__name__)


class TrainHandler:
    """The API handler for training an ML model

    This handler produces a training_request to the jobs kafka topic
    and creates a job entry in the RADP database

    """

    def __init__(self):
        """Initialize kafka producer"""
        self.producer = Producer(kafka_producer_config)

    # TODO: consider making this train API model-type agnostic once we begin
    # training mobility models
    def handle_train_request(self, request: Dict, files: Dict) -> Dict:
        """
        Handle a train request

        request - dictionary with training parameters
        files - a dictionary with the following key-value file paths
        {
            "ue_training_data_file_path": <path_to_training_data_file>,
            "topology_file_path": <path_to_topology_file>
        }

        """
        # parse the request
        train_request: TrainRequest = self._parse_train_request(request)
        logger.info(f"Received request to train model: {train_request}")

        # TODO: create model directory here instead of in _save_metadata_and_topology call

        # build metadata file
        self._save_metadata_and_topology(
            model_id=train_request.model_id,
            topology_file_path=files[constants.TOPOLOGY_FILE_PATH_KEY],
        )

        model_file_path = RADPFileSystemHelper.gen_model_file_path(
            train_request.model_id
        )

        # create a unique id for the kafka job event key
        job = {
            constants.KAFKA_JOB_TYPE: constants.JOB_TYPE_TRAINING,
            constants.MODEL_ID: train_request.model_id,
            constants.MODEL_FILE_PATH: model_file_path,
            constants.MODEL_UPDATE: train_request.model_update,
            constants.TRAINING_PARAMS: train_request.params.to_dict(),
        }

        # add training data file paths to job object
        job.update(files)

        # produce the message to the jobs topic
        logger.info(f"Producing event to jobs topic: {job}")
        job_id = produce_object_to_kafka_topic(
            producer=self.producer,
            topic=constants.KAFKA_JOBS_TOPIC_NAME,
            value=job,
        )

        logger.info(f"Initiated ML training on model {train_request.model_id}.")
        return TrainResponse(job_id=job_id, model_id=train_request.model_id).to_dict()

    # TODO: refactor this method, it's gross
    def _save_metadata_and_topology(self, model_id: str, topology_file_path: str):
        """Create a model metadata file"""
        # pull model-specific information to store in model metadata
        try:
            with open(topology_file_path, "r") as csv_file:
                topology_df = pd.read_csv(csv_file)
                num_cells = len(topology_df)
        except Exception as e:
            logger.exception(
                f"Exception occurred while reading file: {topology_file_path}"
            )
            raise e
        model_specific_params = {constants.NUM_CELLS: num_cells}

        # create a model metadata object
        model_metadata = RADPFileSystemHelper.gen_model_metadata_frame(
            model_id=model_id,
            model_type=ModelType.RF_DIGITAL_TWIN,
            status=ModelStatus.IN_TRAINING,
            model_specific_params=model_specific_params,
        )

        # save metadata to the model's folder
        RADPFileSystemHelper.save_model_metadata(
            model_id=model_id,
            model_metadata=model_metadata,
        )
        logger.debug(f"Saved metadata to file for model: {model_id}")

        # load topology df to write to model folder
        try:
            with open(topology_file_path, "r") as csv_file:
                topology_df = pd.read_csv(csv_file)
        except Exception as e:
            logger.exception("Exception occurred reading cell topology.csv")
            raise e

        model_topology_file_path = RADPFileSystemHelper.gen_model_topology_file_path(
            model_id=model_id
        )
        write_feather_df(file_path=model_topology_file_path, df=topology_df)

    def _parse_train_request(self, event) -> TrainRequest:
        """Parse an incoming Train API request"""

        logger.info(f"Parsing train model event: {event}")
        try:
            train_request = TrainRequest.from_dict(event)
        except TypeError:
            logger.exception("Invalid event body passed in to train API")
            raise InvalidParameterException("Invalid event body passed in to train API")
        return train_request
