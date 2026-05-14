# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import types
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

mock_kafka_module = types.ModuleType("confluent_kafka")
mock_kafka_module.Producer = object
mock_kafka_module.Consumer = object
sys.modules.setdefault("confluent_kafka", mock_kafka_module)

from api_manager.exceptions.validation_exception import ValidationException
from api_manager.handlers.train_handler import TrainHandler

valid_train_request_1 = {
    "model_id": "dummy_model",
    "params": {
        "maxiter": 1,
        "lr": 0.05,
        "stopping_threshold": 0.0001,
    },
}

invalid_train_request_1 = {
    "params": {},
}

invalid_train_request_2 = {
    "model_id": "dummy_model",
    "params": {"fake_param": "Bilbo Baggins"},
}

dummy_files = {
    "ue_training_data_file_path": "dummy_ue_training_data_file_path",
    "topology_file_path": "dummy_topology_file_path",
}


class TestTrainHandler(TestCase):
    @patch("api_manager.handlers.train_handler.Producer")
    def test_handle_train_request__invalid_requests(self, _mock_producer):
        with self.assertRaises(ValidationException):
            TrainHandler().handle_train_request(invalid_train_request_1, dummy_files)
        with self.assertRaises(ValidationException):
            TrainHandler().handle_train_request(invalid_train_request_2, dummy_files)

    @patch("api_manager.handlers.train_handler.TrainingRequestValidator")
    @patch("builtins.open")
    @patch("api_manager.handlers.train_handler.pd")
    @patch("api_manager.handlers.train_handler.write_feather_df")
    @patch("api_manager.handlers.train_handler.Producer")
    @patch("api_manager.handlers.train_handler.RADPFileSystemHelper")
    @patch("api_manager.handlers.train_handler.produce_object_to_kafka_topic")
    def test_handle_train_request(
        self,
        mock_produce,
        mock_file_system_helper,
        mock_producer,
        mock_write_feather_,
        mock_pd_,
        mock_open_,
        mock_validator_cls,
    ):
        mock_producer_instance = MagicMock
        mock_producer.return_value = mock_producer_instance
        mock_file_system_helper.gen_model_file_path.return_value = (
            "dummy_model_file_path"
        )

        mock_produce.return_value = "dummy_job_id"

        expected_job = {
            "job_type": "training",
            "model_update": False,
            "model_id": "dummy_model",
            "training_params": {
                "maxiter": 1,
                "lr": 0.05,
                "stopping_threshold": 0.0001,
            },
            "model_file_path": "dummy_model_file_path",
            "ue_training_data_file_path": "dummy_ue_training_data_file_path",
            "topology_file_path": "dummy_topology_file_path",
        }

        assert TrainHandler().handle_train_request(
            valid_train_request_1, files=dummy_files
        ) == {
            "job_id": "dummy_job_id",
            "model_id": "dummy_model",
        }

        mock_file_system_helper.gen_model_file_path.assert_called_once_with(
            "dummy_model"
        )
        mock_file_system_helper.save_model_metadata.assert_called_once()
        mock_file_system_helper.gen_model_topology_file_path.assert_called_once()
        mock_validator_cls.return_value.validate.assert_called_once_with(
            valid_train_request_1
        )
        mock_validator_cls.return_value.validate_training_files.assert_called_once_with(
            dummy_files
        )

        mock_produce.assert_called_once_with(
            producer=mock_producer_instance, topic="jobs", value=expected_job
        )
