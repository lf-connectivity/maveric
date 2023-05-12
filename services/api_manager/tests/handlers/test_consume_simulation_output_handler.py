# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, patch

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.simulation_output_not_found_exception import SimulationOutputNotFoundException
from api_manager.handlers.consume_simulation_output_handler import ConsumeSimulationOutputHandler


class TestConsumeSimulationOutputHandler(TestCase):
    @patch("api_manager.handlers.consume_simulation_output_handler.os")
    @patch("api_manager.handlers.consume_simulation_output_handler.RADPFileSystemHelper")
    def test_handle_consume_simulation_output_request(self, mock_file_system_helper: MagicMock, mock_os: MagicMock):
        mock_file_system_helper.gen_sim_output_zip_file_path.return_value = "dummy_outzip_zip"
        mock_os.path.exists.return_value = True
        dummy_sim_id = "dummy_sim"

        assert (
            ConsumeSimulationOutputHandler().handle_consume_simulation_output_request(dummy_sim_id)
            == "dummy_outzip_zip"
        )
        mock_file_system_helper.gen_sim_output_zip_file_path.assert_called_once_with("dummy_sim")
        mock_os.path.exists.assert_called_once_with("dummy_outzip_zip")

    @patch("api_manager.handlers.consume_simulation_output_handler.os")
    @patch("api_manager.handlers.consume_simulation_output_handler.RADPFileSystemHelper")
    def test_handle_consume_simulation_output_request__nonexistent_model(
        self, mock_file_system_helper: MagicMock, mock_os: MagicMock
    ):
        mock_file_system_helper.gen_sim_output_zip_file_path.return_value = "dummy_outzip_zip"
        mock_os.path.exists.return_value = False

        with self.assertRaises(SimulationOutputNotFoundException):
            assert ConsumeSimulationOutputHandler().handle_consume_simulation_output_request("dummy_sim")

    def test_handle_consume_simulation_output_request__no_sim_id(self):
        with self.assertRaises(InvalidParameterException):
            ConsumeSimulationOutputHandler().handle_consume_simulation_output_request("")
