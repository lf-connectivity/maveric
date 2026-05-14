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
from api_manager.validators.simulation_validator import SimulationRequestValidator


class TestSimulationRequestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SimulationRequestValidator()
        self.valid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {"ue_data_id": "ue_data_1"},
            "rf_prediction": {"model_id": "model_1", "config_id": "config_1"},
        }

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_uploaded_data_request_success(self, _mock_exists):
        self.validator.validate(self.valid_request)

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_generation_request_success(self, _mock_exists):
        self.validator.validate(self._generation_request())

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_rejects_both_ue_sources(self, _mock_exists):
        request = dict(self.valid_request)
        request["ue_tracks"] = {
            "ue_data_id": "ue_data_1",
            "ue_tracks_generation": self._generation_request()["ue_tracks"][
                "ue_tracks_generation"
            ],
        }

        with self.assertRaises(ValidationException) as ctx:
            self.validator.validate(request)

        self.assertEqual(ctx.exception.field, "ue_tracks")

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_rejects_invalid_generation_class(self, _mock_exists):
        request = self._generation_request()
        request["ue_tracks"]["ue_tracks_generation"]["ue_class_distribution"] = {
            "hoverboard": {"count": 1, "velocity": 1, "velocity_variance": 1}
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(request)

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_rejects_invalid_boundaries(self, _mock_exists):
        request = self._generation_request()
        request["ue_tracks"]["ue_tracks_generation"]["lat_lon_boundaries"][
            "min_lat"
        ] = 36

        with self.assertRaises(ValidationException):
            self.validator.validate(request)

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=False,
    )
    def test_validate_rejects_missing_model(self, _mock_exists):
        with self.assertRaises(ValidationException) as ctx:
            self.validator.validate(self.valid_request)

        self.assertEqual(ctx.exception.field, "rf_prediction.model_id")

    @patch(
        "radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists",
        return_value=True,
    )
    def test_validate_rejects_protocol_ranges(self, _mock_exists):
        request = dict(self.valid_request)
        request["protocol_emulation"] = {"ttt_seconds": 11, "hysteresis": 21}

        with self.assertRaises(ValidationException) as ctx:
            self.validator.validate(request)

        fields = {error["field"] for error in ctx.exception.validation_errors}
        self.assertEqual(fields, {"protocol_emulation.ttt_seconds", "protocol_emulation.hysteresis"})

    def test_validate_simulation_files_success(self):
        ue_path, config_path = self._write_simulation_files()
        try:
            self.validator.validate_simulation_files(
                {"ue_data_file_path": ue_path, "config_file_path": config_path}
            )
        finally:
            os.unlink(ue_path)
            os.unlink(config_path)

    def test_validate_simulation_files_rejects_bad_ue_data(self):
        ue_path, config_path = self._write_simulation_files(
            ue_overrides={"lat": [100], "tick": [-1]}
        )
        try:
            with self.assertRaises(FileValidationException) as ctx:
                self.validator.validate_simulation_files(
                    {"ue_data_file_path": ue_path, "config_file_path": config_path}
                )
            fields = {error["field"] for error in ctx.exception.validation_errors}
            self.assertEqual(fields, {"lat", "tick"})
        finally:
            os.unlink(ue_path)
            os.unlink(config_path)

    def test_validate_simulation_files_rejects_duplicate_config_cells(self):
        ue_path, config_path = self._write_simulation_files(
            config_rows=[
                {"cell_id": "cell_1", "cell_el_deg": 5},
                {"cell_id": "cell_1", "cell_el_deg": 6},
            ]
        )
        try:
            with self.assertRaises(FileValidationException):
                self.validator.validate_simulation_files(
                    {"ue_data_file_path": ue_path, "config_file_path": config_path}
                )
        finally:
            os.unlink(ue_path)
            os.unlink(config_path)

    def _generation_request(self):
        return {
            "simulation_time_interval_seconds": 0.1,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "simulation_duration_seconds": 10,
                    "ue_class_distribution": {
                        "pedestrian": {
                            "count": 5,
                            "velocity": 1,
                            "velocity_variance": 1,
                        }
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35,
                        "max_lat": 36,
                        "min_lon": -81,
                        "max_lon": -80,
                    },
                    "gauss_markov_params": {"alpha": 0.5, "variance": 1, "rng_seed": 1},
                }
            },
            "rf_prediction": {"model_id": "model_1"},
        }

    def _write_simulation_files(self, ue_overrides=None, config_rows=None):
        ue_data = {"mock_ue_id": [1], "lon": [-80.0], "lat": [35.0], "tick": [0]}
        if ue_overrides:
            ue_data.update(ue_overrides)

        if config_rows is None:
            config_rows = [{"cell_id": "cell_1", "cell_el_deg": 5}]

        ue_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        ue_file.close()
        config_file.close()
        pd.DataFrame(ue_data).to_csv(ue_file.name, index=False)
        pd.DataFrame(config_rows).to_csv(config_file.name, index=False)
        return ue_file.name, config_file.name
