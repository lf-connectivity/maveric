# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, patch

from api_manager.handlers.simulation_handler import SimulationHandler

dummy_rf_sim = {
    "simulation_time_interval_seconds": 0.01,
    "ue_tracks": {"ue_data_id": "dummy_id"},
    "rf_prediction": {"model_id": "dummy_model", "config_id": "dummy_config"},
}

dummy_files = {
    "ue_data_file_path": "dummy_ue_data_file_path",
    "config_file_path": "dummy_config_file_path",
}


class TestDescribeSimulationHandler(TestCase):
    @patch("api_manager.handlers.simulation_handler.Producer")
    @patch("api_manager.handlers.simulation_handler.RICSimulationRequestPreprocessor")
    @patch("api_manager.handlers.simulation_handler.RADPFileSystemHelper")
    @patch("api_manager.handlers.simulation_handler.os")
    @patch("api_manager.handlers.simulation_handler.produce_object_to_kafka_topic")
    def test_handle_simulation_request(
        self,
        mock_produce,
        mock_os,
        mock_file_system_helper,
        mock_preprocessor,
        mock_producer,
    ):
        mock_producer_instance = MagicMock
        mock_producer.return_value = mock_producer_instance

        mock_produce.return_value = "dummy_job_id"
        mock_preprocessor.preprocess.return_value = {
            "simulation_id": "dummy_sim_id",
        }

        expected_job = {"job_type": "orchestration", "simulation_id": "dummy_sim_id"}

        assert SimulationHandler().handle_simulation_request(
            dummy_rf_sim, dummy_files
        ) == {
            "job_id": "dummy_job_id",
            "simulation_id": "dummy_sim_id",
        }

        mock_os.path.exists.assert_called_once()
        mock_file_system_helper.save_simulation_ue_data.assert_called_once_with(
            "dummy_sim_id", ue_data_file_path="dummy_ue_data_file_path"
        )
        mock_file_system_helper.save_simulation_cell_config.assert_called_once_with(
            "dummy_sim_id", config_file_path="dummy_config_file_path"
        )

        mock_produce.assert_called_once_with(
            producer=mock_producer_instance, topic="jobs", value=expected_job
        )
