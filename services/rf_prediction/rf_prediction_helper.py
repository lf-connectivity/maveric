# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

from radp.common import constants


class RFPredictionHelper:
    """Helper class for RF Prediction Service

    Contains simple methods to read data from Kafka job event and
    other helpful utilities
    """

    @staticmethod
    def get_model_parameters(job_data: Dict) -> Tuple[str, str]:
        """Helper method to return model parameters of an RF Prediction job"""
        # pull each input file path
        return (
            job_data[constants.RF_PREDICTION][constants.MODEL_ID],
            job_data[constants.RF_PREDICTION][constants.MODEL_FILE],
        )

    @staticmethod
    def get_ue_data_file_path(job_data: Dict) -> str:
        """Helper method to return UE data file path of an RF Prediction job"""
        # pull file path object
        return job_data[constants.RF_PREDICTION][constants.UE_DATA_FILE_PATH_KEY]

    @staticmethod
    def get_output_file_path(job_data: Dict) -> str:
        """Pull the output file from an RF Prediction job event"""
        return job_data[constants.RF_PREDICTION][constants.OUTPUT_FILE_PATH]

    @staticmethod
    def get_config_file_path(job_data: Dict) -> str:
        """Helper method to return config file path of an RF Prediction job"""
        # pull file path object
        return job_data[constants.RF_PREDICTION][constants.CONFIG_FILE_PATH_KEY]

    @staticmethod
    def get_topology_file_path(job_data: Dict) -> str:
        """Helper method to return topology file path of an RF Prediction job"""
        # pull file path object
        return job_data[constants.RF_PREDICTION][constants.TOPOLOGY_FILE_PATH_KEY]

    @staticmethod
    def get_simulation_id(job_data: Dict) -> str:
        """Pull the simulation id from an RF Prediction job event"""
        return job_data[constants.SIMULATION_ID]

    @staticmethod
    def get_batch_index(job_data: Dict) -> str:
        """Pull the batch index from an RF Prediction job event"""
        return job_data[constants.BATCH]
