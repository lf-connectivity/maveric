# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd

from api_manager.exceptions.validation_exception import (
    FileValidationException,
    ValidationException,
)
from api_manager.validators.training_validator import TrainingRequestValidator


class TestTrainingRequestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = TrainingRequestValidator()
        self.valid_request = {
            "model_id": "valid_model_1",
            "model_update": False,
            "params": {"maxiter": 10, "lr": 0.05, "stopping_threshold": 0.0001},
        }

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=False,
    )
    def test_validate_request_success(self, _mock_exists):
        self.validator.validate(self.valid_request)

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=False,
    )
    def test_validate_request_rejects_bad_params(self, _mock_exists):
        request = {
            "model_id": "bad model",
            "params": {"maxiter": 0, "lr": 2.0, "unknown": True},
        }

        with self.assertRaises(ValidationException) as ctx:
            self.validator.validate(request)

        fields = {error["field"] for error in ctx.exception.validation_errors}
        self.assertIn("model_id", fields)
        self.assertIn("params.maxiter", fields)
        self.assertIn("params.lr", fields)
        self.assertIn("params.unknown", fields)

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_request_rejects_duplicate_create(self, _mock_exists):
        with self.assertRaises(ValidationException) as ctx:
            self.validator.validate(self.valid_request)

        self.assertEqual(ctx.exception.field, "model_id")

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=False,
    )
    def test_validate_request_rejects_missing_update_target(self, _mock_exists):
        request = dict(self.valid_request)
        request["model_update"] = True

        with self.assertRaises(ValidationException):
            self.validator.validate(request)

    def test_validate_training_files_success(self):
        training_path, topology_path = self._write_training_files()
        try:
            self.validator.validate_training_files(
                {
                    "ue_training_data_file_path": training_path,
                    "topology_file_path": topology_path,
                }
            )
        finally:
            os.unlink(training_path)
            os.unlink(topology_path)

    def test_validate_training_files_rejects_ranges(self):
        training_path, topology_path = self._write_training_files(
            training_overrides={"avg_rsrp": [10], "lat": [100]}
        )
        try:
            with self.assertRaises(FileValidationException) as ctx:
                self.validator.validate_training_files(
                    {
                        "ue_training_data_file_path": training_path,
                        "topology_file_path": topology_path,
                    }
                )
            fields = {error["field"] for error in ctx.exception.validation_errors}
            self.assertEqual(fields, {"avg_rsrp", "lat"})
        finally:
            os.unlink(training_path)
            os.unlink(topology_path)

    def test_validate_training_files_rejects_missing_topology_cell(self):
        training_path, topology_path = self._write_training_files(
            training_overrides={"cell_id": ["missing_cell"]}
        )
        try:
            with self.assertRaises(ValidationException) as ctx:
                self.validator.validate_training_files(
                    {
                        "ue_training_data_file_path": training_path,
                        "topology_file_path": topology_path,
                    }
                )
            self.assertEqual(ctx.exception.validation_errors[0]["field"], "cell_id")
        finally:
            os.unlink(training_path)
            os.unlink(topology_path)

    def test_validate_training_files_rejects_duplicate_topology_cell(self):
        training_path, topology_path = self._write_training_files(
            topology_rows=[
                {
                    "cell_lat": 35.0,
                    "cell_lon": -80.0,
                    "cell_id": "cell_1",
                    "cell_az_deg": 0,
                    "cell_carrier_freq_mhz": 2100,
                },
                {
                    "cell_lat": 35.1,
                    "cell_lon": -80.1,
                    "cell_id": "cell_1",
                    "cell_az_deg": 90,
                    "cell_carrier_freq_mhz": 2100,
                },
            ]
        )
        try:
            with self.assertRaises(FileValidationException):
                self.validator.validate_training_files(
                    {
                        "ue_training_data_file_path": training_path,
                        "topology_file_path": topology_path,
                    }
                )
        finally:
            os.unlink(training_path)
            os.unlink(topology_path)

    def _write_training_files(
        self,
        training_overrides=None,
        topology_rows=None,
    ):
        training_data = {
            "cell_id": ["cell_1"],
            "avg_rsrp": [-80],
            "lon": [-80.0],
            "lat": [35.0],
            "cell_el_deg": [5],
        }
        if training_overrides:
            training_data.update(training_overrides)

        if topology_rows is None:
            topology_rows = [
                {
                    "cell_lat": 35.0,
                    "cell_lon": -80.0,
                    "cell_id": "cell_1",
                    "cell_az_deg": 0,
                    "cell_carrier_freq_mhz": 2100,
                }
            ]

        training_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        topology_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        training_file.close()
        topology_file.close()

        pd.DataFrame(training_data).to_csv(training_file.name, index=False)
        pd.DataFrame(topology_rows).to_csv(topology_file.name, index=False)
        return training_file.name, topology_file.name
