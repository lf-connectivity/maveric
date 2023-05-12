# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, patch

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.model_not_found_exception import ModelNotFoundException
from api_manager.handlers.describe_model_handler import DescribeModelHandler


class TestDescribeModelHandler(TestCase):
    @patch("api_manager.handlers.describe_model_handler.RADPFileSystemHelper")
    def test_handle_describe_model_request(self, mock_file_system_helper: MagicMock):
        mock_file_system_helper.load_model_metadata.return_value = {"metadata": "dummy"}
        assert DescribeModelHandler().handle_describe_model_request("dummy_model") == {"metadata": "dummy"}
        mock_file_system_helper.load_model_metadata.assert_called_once_with(model_id="dummy_model")

    @patch("api_manager.handlers.describe_model_handler.RADPFileSystemHelper")
    def test_handle_describe_model_request__model_not_found(self, mock_file_system_helper: MagicMock):
        mock_file_system_helper.side_effect
        mock_file_system_helper.load_model_metadata.side_effect = FileNotFoundError()
        with self.assertRaises(ModelNotFoundException):
            DescribeModelHandler().handle_describe_model_request("dummy_model")

    def test_handle_describe_model_request__invalid_model_id(self):
        with self.assertRaises(InvalidParameterException):
            DescribeModelHandler().handle_describe_model_request("")
