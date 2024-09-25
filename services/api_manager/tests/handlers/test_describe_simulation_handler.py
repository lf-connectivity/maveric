# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, patch

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.simulation_not_found_exception import (
    SimulationNotFoundException,
)
from api_manager.handlers.describe_simulation_handler import DescribeSimulationHandler


class TestDescribeSimulationHandler(TestCase):
    @patch("api_manager.handlers.describe_simulation_handler.RADPFileSystemHelper")
    def test_handle_describe_simulation_request(
        self, mock_file_system_helper: MagicMock
    ):
        mock_file_system_helper.load_simulation_metadata.return_value = {
            "metadata": "dummy"
        }
        assert DescribeSimulationHandler().handle_describe_simulation_request(
            "dummy_simulation"
        ) == {"metadata": "dummy"}
        mock_file_system_helper.load_simulation_metadata.assert_called_once_with(
            "dummy_simulation"
        )

    @patch("api_manager.handlers.describe_simulation_handler.RADPFileSystemHelper")
    def test_handle_describe_simulation_request__simulation_not_found(
        self, mock_file_system_helper: MagicMock
    ):
        mock_file_system_helper.side_effect
        mock_file_system_helper.load_simulation_metadata.side_effect = (
            FileNotFoundError()
        )
        with self.assertRaises(SimulationNotFoundException):
            DescribeSimulationHandler().handle_describe_simulation_request(
                "dummy_simulation"
            )

    def test_handle_describe_simulation_request__invalid_simulation_id(self):
        with self.assertRaises(InvalidParameterException):
            DescribeSimulationHandler().handle_describe_simulation_request("")
