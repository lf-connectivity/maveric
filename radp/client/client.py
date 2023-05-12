# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, Set, Union

import pandas as pd
import requests
from retry import retry

from radp.client import constants
from radp.client.requests_client import RequestsClient

logger = logging.getLogger(__name__)

RETRY_EXCEPTIONS = requests.exceptions.ConnectionError

RADP_SERVICE_IP = os.environ.get("RADP_SERVICE_IP", "127.0.0.1")
RADP_SERVICE_PORT = int(os.environ.get("RADP_SERVICE_PORT", "8080"))


def df_to_csv_file(df: pd.DataFrame, csv_file_name: str):
    """Helper method to save a pandas dataframe to csv file"""

    # create tmp folder if it does not already exist
    if not os.path.exists(constants.TMP_DIRECTORY):
        os.makedirs(constants.TMP_DIRECTORY)

    try:
        logger.debug(f"Saving df to file: {csv_file_name}")
        df.to_csv(csv_file_name, index=False)
    except Exception as e:
        logger.exception(f"Exception occurred while saving df to csv file: {e}")
        raise e


# TODO: clean up this client, it's ugly
class RADPClient(RequestsClient):
    """Simple RADP client"""

    def __init__(self, ip=RADP_SERVICE_IP, port=RADP_SERVICE_PORT):
        super().__init__(ip, port)

    # TODO: add error handling logic for when something goes wrong
    @retry(exceptions=RETRY_EXCEPTIONS, tries=3, delay=1, backoff=2)
    def describe_model(self, model_id: str) -> Dict:
        logger.debug(f"Calling describe_model api with model: '{model_id}'")
        path = f"{constants.DESCRIBE_MODEL_API_PATH}/{model_id}"
        response = self._send_get_request(path, {})

        # format response
        response_dict = {}
        response_dict[constants.MODEL_EXISTS] = response.status_code == 200
        response_dict.update(response.json())

        return response_dict

    # TODO: add error handling logic for when something goes wrong
    @retry(exceptions=RETRY_EXCEPTIONS, tries=3, delay=1, backoff=2)
    def describe_simulation(self, simulation_id: str) -> Dict:
        logger.debug(f"Calling describe_simulation api with simulation: '{simulation_id}'")
        path = f"{constants.DESCRIBE_SIMULATION_API_PATH}/{simulation_id}"
        response = self._send_get_request(path, {})

        # format response
        response_dict = {}
        response_dict[constants.SIMULATION_EXISTS] = response.status_code == 200
        response_dict.update(response.json())

        return response_dict

    # TODO: add error handling logic for when something goes wrong
    @retry(exceptions=RETRY_EXCEPTIONS, tries=3, delay=1, backoff=2)
    def train(
        self,
        model_id: str,
        params: Dict,
        ue_training_data: Union[str, pd.DataFrame],
        topology: Union[str, pd.DataFrame],
        model_update: bool = False,
    ) -> Dict:
        logger.debug(f"calling RADP train api to create model: '{model_id}'")

        # save dfs to csv files if provided
        if isinstance(ue_training_data, pd.DataFrame):
            df_to_csv_file(ue_training_data, constants.TMP_UE_TRAINING_DATA_FILE_NAME)
            ue_training_data = constants.TMP_UE_TRAINING_DATA_FILE_NAME
        if isinstance(topology, pd.DataFrame):
            df_to_csv_file(topology, constants.TMP_TOPOLOGY_FILE_NAME)
            topology = constants.TMP_TOPOLOGY_FILE_NAME

        payload = {
            constants.PAYLOAD_KEY_MODEL_ID: model_id,
            constants.PAYLOAD_KEY_MODEL_UPDATE: model_update,
            constants.PAYLOAD_KEY_PARAMS: params,
        }
        payload_file = json.dumps(payload)

        # open user provided training csv files
        with open(ue_training_data, "r") as ue_training_data_file, open(topology, "r") as topology_file:
            # send a json body file as well as both csv files
            files: Set[Any] = {
                (
                    constants.POST_REQUEST_PAYLOAD_LABEL,
                    (
                        constants.RECOGNIZED_PAYLOAD_FILE_NAME,
                        payload_file,
                        constants.HTTP_CONTENT_TYPE_JSON,
                    ),
                ),
                (
                    constants.TRAIN_REQUEST_UE_TRAINING_DATA_LABEL,
                    (
                        constants.TRAIN_REQUEST_UE_TRAINING_DATA_LABEL,
                        ue_training_data_file,
                        constants.HTTP_CONTENT_TYPE_CSV,
                    ),
                ),
                (
                    constants.TRAIN_REQUEST_TOPOLOGY_LABEL,
                    (
                        constants.TRAIN_REQUEST_TOPOLOGY_LABEL,
                        topology_file,
                        constants.HTTP_CONTENT_TYPE_CSV,
                    ),
                ),
            }
            response = self._send_post_request(constants.TRAIN_API_PATH, files=files)
            return response.json()

    # TODO: add error handling logic for when something goes wrong
    @retry(exceptions=RETRY_EXCEPTIONS, tries=3, delay=1, backoff=2)
    def simulation(
        self,
        simulation_event: Dict,
        ue_data: Union[str, pd.DataFrame, None] = None,
        config: Union[str, pd.DataFrame, None] = None,
    ) -> Dict:
        logger.debug(
            f"Calling simulation API with the following simulation event:\n{json.dumps(simulation_event, indent=4)}"
        )

        # save dfs to csv files if provided
        if isinstance(ue_data, pd.DataFrame):
            df_to_csv_file(ue_data, constants.TMP_UE_DATA_FILE_NAME)
            ue_data = constants.TMP_UE_DATA_FILE_NAME
        if isinstance(config, pd.DataFrame):
            df_to_csv_file(config, constants.TMP_CONFIG_FILE_NAME)
            config = constants.TMP_CONFIG_FILE_NAME

        payload_file = json.dumps(simulation_event)

        files: Set[Any] = {
            (
                constants.POST_REQUEST_PAYLOAD_LABEL,
                (
                    constants.POST_REQUEST_PAYLOAD_LABEL,
                    payload_file,
                    constants.HTTP_CONTENT_TYPE_JSON,
                ),
            )
        }

        if not ue_data and not config:
            return self._send_post_request(constants.SIMULATION_API_PATH, files=files).json()

        if not config:
            with open(str(ue_data), "r") as ue_data_file:
                files.add(
                    (
                        constants.SIMULATION_REQUEST_UE_DATA_LABEL,
                        (
                            constants.SIMULATION_REQUEST_UE_DATA_LABEL,
                            ue_data_file,
                            constants.HTTP_CONTENT_TYPE_CSV,
                        ),
                    )
                )
                return self._send_post_request(constants.SIMULATION_API_PATH, files=files).json()

        if not ue_data:
            with open(str(config), "r") as config_file:
                files.add(
                    (
                        constants.SIMULATION_REQUEST_CONFIG_LABEL,
                        (
                            constants.SIMULATION_REQUEST_CONFIG_LABEL,
                            config_file,
                            constants.HTTP_CONTENT_TYPE_CSV,
                        ),
                    )
                )
                return self._send_post_request(constants.SIMULATION_API_PATH, files=files).json()

        with open(ue_data, "r") as ue_data_file, open(config, "r") as config_file:
            files.add(
                (
                    constants.SIMULATION_REQUEST_UE_DATA_LABEL,
                    (
                        constants.SIMULATION_REQUEST_UE_DATA_LABEL,
                        ue_data_file,
                        constants.HTTP_CONTENT_TYPE_CSV,
                    ),
                ),
            )
            files.add(
                (
                    constants.SIMULATION_REQUEST_CONFIG_LABEL,
                    (
                        constants.SIMULATION_REQUEST_CONFIG_LABEL,
                        config_file,
                        constants.HTTP_CONTENT_TYPE_CSV,
                    ),
                ),
            )
            return self._send_post_request(constants.SIMULATION_API_PATH, files=files).json()

    # TODO: add error handling logic for when something goes wrong
    @retry(exceptions=RETRY_EXCEPTIONS, tries=3, delay=1, backoff=2)
    def consume_simulation_output(self, simulation_id: str) -> pd.DataFrame:
        logger.debug(f"Calling consume_simulation_output api with simulation: '{simulation_id}'")

        path = f"{constants.CONSUME_SIMULATION_OUTPUT_API_PATH}/{simulation_id}/{constants.DOWNLOAD}"
        consume_simulation_output_url = self._get_request_url(path, {})

        # TODO: this only works for a single file in zipfile
        # we will need to update this once batching is supported
        rf_dataframe = pd.read_csv(consume_simulation_output_url, compression=constants.ZIP_COMPRESSION)
        return rf_dataframe
