# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The RIC Simulation Request Validator.
"""

import logging
from typing import Dict

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException

logger = logging.getLogger(__name__)


"""

## Sample RIC Simulation Request Dictionary
## with UE tracks generation as UE tracks data source

{
   "simulation_duration_seconds": 3600,
   "simulation_time_interval_seconds": 0.01,
   "ue_tracks": {
        "ue_tracks_generation" : {
            "ue_ue_class_distribution": {
                "stationary": {
                    "count": 0,
                    "velocity": 1,
                    "velocity_variance": 1
                },
                "pedestrian": {
                    "count": 0,
                    "velocity": 1,
                    "velocity_variance": 1
                },
                "cyclist": {
                    "count": 0,
                    "velocity": 1,
                    "velocity_variance": 1
                },
                "car": {
                    "count": 0,
                    "velocity": 1,
                    "velocity_variance": 1
                }
            },
            "gauss_markov_params": {
                "alpha": 0.5,
                "variance": 0.8
            }
        }
   },
   "rf_prediction": {
       "model_id": "detroit_10"
   },
   "protocol_emulation": {
       "ttt_seconds": 1,
       "hysteresis": 2
   }
}

## Sample RIC Simulation Request Dictionary
## user-created ue_data_id as UE Tracks data source


{
   "simulation_duration": 3600,
   "simulation_time_interval": 0.01,
   "ue_tracks": {
       "ue_data_id": "<ue_data_id>",
   }
   "rf_prediction": {
       "model_id": "detroit_10"
   },
   "protocol_emulation": {
       "params": {}
   }
}


"""


class RICSimulationRequestValidator:
    @staticmethod
    def validate(request: Dict):
        """Validate RIC Simulation Request."""

        # call all the validators
        RICSimulationRequestValidator._validate_rf_prediction(request)

    @staticmethod
    def _validate_rf_prediction(request: Dict):
        """Validate `rf_prediction` component of RIC Simulation Request.

        Throws `InvalidParameterException` if invalid.
        """

        if ("ue_tracks" not in request) or (
            ("ue_tracks_generation" not in request["ue_tracks"])
            and ("ue_data_id" not in request["ue_tracks"])
        ):
            raise InvalidParameterException(
                "Must provide ue_tracks section with either provide `ue_tracks_generation` "
                "or `ue_data_id`, in RIC Simulation Request spec!"
            )

        if "rf_prediction" not in request:
            raise InvalidParameterException(
                "Missing rf_prediction key in RIC Simulation Request spec!"
            )

        return None
