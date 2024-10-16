# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.validators.simulation_request_validator import (
    RICSimulationRequestValidator,
)


class TestRICSimulationRequestValidator(unittest.TestCase):
    def test__validate_rf_prediction(self):
        # Missing ue_tracks
        with self.assertRaises(InvalidParameterException) as ipe:
            RICSimulationRequestValidator._validate_rf_prediction({})
        self.assertEqual(
            str(ipe.exception),
            "Must provide ue_tracks section with either provide `ue_tracks_generation` "
            "or `ue_data_id`, in RIC Simulation Request spec!",
        )

        # Missing ue_tracks_generation and ue_data_id
        with self.assertRaises(InvalidParameterException) as ipe:
            RICSimulationRequestValidator._validate_rf_prediction({"ue_tracks": {}})
        self.assertEqual(
            str(ipe.exception),
            "Must provide ue_tracks section with either provide `ue_tracks_generation` "
            "or `ue_data_id`, in RIC Simulation Request spec!",
        )

        # Missing rf_prediction (but ue_tracks present and valid)
        with self.assertRaises(InvalidParameterException) as ipe:
            RICSimulationRequestValidator._validate_rf_prediction(
                {"ue_tracks": {"ue_tracks_generation": {}}}
            )
        self.assertEqual(
            str(ipe.exception),
            "Missing rf_prediction key in RIC Simulation Request spec!",
        )
        with self.assertRaises(InvalidParameterException) as ipe:
            RICSimulationRequestValidator._validate_rf_prediction(
                {"ue_tracks": {"ue_data_id": {}}}
            )
        self.assertEqual(
            str(ipe.exception),
            "Missing rf_prediction key in RIC Simulation Request spec!",
        )

        # rf_prediction present (and ue_tracks present and valid)
        RICSimulationRequestValidator._validate_rf_prediction(
            request={
                "ue_tracks": {"ue_tracks_generation": {}},
                "rf_prediction": {
                    "model_id": "detroit_10",
                },
            }
        )
        RICSimulationRequestValidator._validate_rf_prediction(
            request={
                "ue_tracks": {"ue_data_id": {}},
                "rf_prediction": {
                    "model_id": "detroit_10",
                },
            }
        )
