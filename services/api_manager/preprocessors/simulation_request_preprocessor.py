# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The RIC Simulation Request Preprocessor : event --> RIC Simulation Metadata.
"""

import hashlib
import json
import logging
from typing import Any, Dict

from radp.common import constants

logger = logging.getLogger(__name__)


"""
## Sample RIC Simulation Event Dictionary
## with UE tracks generation as UE tracks data source

{
    "simulation_time_interval_seconds": 0.01,
    "ue_tracks": {
        "ue_tracks_generation" : {
            "ue_class_distribution": {
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
            "lat_lon_boundaries": {
                "min_lat": -90,
                "max_lat": 90,
                "min_lon": -180,
                "max_lon": 180
            },
            "gauss_markov_params": {
                "alpha": 0.5,
                "variance": 0.8,
                "rng_seed": 42
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
{
    "simulation_time_interval_seconds": 0.01,
    "ue_tracks": {
        "ue_data_id": "ue_data_1"
    },
    "rf_prediction": {
        "model_id": "test_dt",
        "config_id": "test_config_1"
    }
}
"""


def deterministic_hash_dict(kv: Dict) -> str:
    return hashlib.md5(json.dumps(kv, sort_keys=True).encode("utf-8")).hexdigest()


class RICSimulationRequestPreprocessor:
    @staticmethod
    def preprocess(
        request: Dict,
        files: Dict,
    ) -> Dict:
        """Preprocess a RIC Simulation Event spec, produced by a user-API-call,
        and produce a processed (runnable) RIC Simulation Event spec.
        """

        # start chained hash from top-level simulation parameters
        chained_hash_val = deterministic_hash_dict(
            {constants.SIMULATION_TIME_INTERVAL: request[constants.SIMULATION_TIME_INTERVAL]}
        )

        # build processed request frame (this will become to simulation metadata object)
        processed_request: Dict[str, Any] = {}

        # simulation status initialized as "planned"
        processed_request[constants.SIMULATION_STATUS] = constants.STATUS_PLANNED

        # supply simulation interval value
        processed_request[constants.SIMULATION_TIME_INTERVAL] = request[constants.SIMULATION_TIME_INTERVAL]

        # get the ue_tracks hash value
        chained_hash_val = deterministic_hash_dict(
            {
                constants.PREDECESSOR_STAGE_HASH_VAL: chained_hash_val,
                **request[constants.UE_TRACKS],
            }
        )

        # check if UE tracks generation is requested, if so, preprocess and
        # supply to the processed request
        if constants.UE_TRACKS_GENERATION in request[constants.UE_TRACKS]:
            # supply UE tracks generation object to the processed request
            processed_request[constants.UE_TRACKS_GENERATION] = {}
            processed_request[constants.UE_TRACKS_GENERATION][constants.PARAMS] = request[constants.UE_TRACKS][
                constants.UE_TRACKS_GENERATION
            ]

            # get num_ticks using division of duration by time interval
            simulation_duration = processed_request[constants.UE_TRACKS_GENERATION][constants.PARAMS][
                constants.SIMULATION_DURATION
            ]
            processed_request[constants.NUM_TICKS] = int(
                simulation_duration / processed_request[constants.SIMULATION_TIME_INTERVAL]
            )

            # set hash value in ue_tracks generation
            processed_request[constants.UE_TRACKS_GENERATION][constants.HASH_VAL] = chained_hash_val

            # add state object to UE tracks generation object
            processed_request[constants.UE_TRACKS_GENERATION][constants.STATE] = {}
            processed_request[constants.UE_TRACKS_GENERATION][constants.STATE][
                constants.STATUS
            ] = constants.STATUS_PLANNED
            processed_request[constants.UE_TRACKS_GENERATION][constants.STATE][constants.BATCHES_OUTPUTTED] = 0

        else:
            # TODO: get actual num_ticks value by scanning input file "tick" column
            # for now we'll just make it 1 since it is only used in UE tracks generation
            processed_request[constants.NUM_TICKS] = 1

        # supply number of batches (TODO : generalize later)
        processed_request[constants.NUM_BATCHES] = 1

        def _preprocess_stage(
            stage: str,
            chained_hash_val: str,
        ) -> str:
            # build request frame
            processed_request[stage] = {}
            processed_request[stage][constants.PARAMS] = request[stage]

            # chained hashing based on params and previous stage hash
            chained_hash_val = deterministic_hash_dict(
                {
                    constants.PREDECESSOR_STAGE_HASH_VAL: chained_hash_val,
                    **request[stage],
                }
            )
            processed_request[stage][constants.HASH_VAL] = chained_hash_val

            # initial empty state
            processed_request[stage][constants.STATE] = {}
            processed_request[stage][constants.STATE][constants.STATUS] = constants.STATUS_PLANNED
            processed_request[stage][constants.STATE][constants.LATEST_BATCH_WITHOUT_FAILURE] = 0
            processed_request[stage][constants.STATE][constants.LATEST_BATCH_TO_SUCCEED] = 0
            processed_request[stage][constants.STATE][constants.BATCHES_RETRYING] = []
            return chained_hash_val

        # RF Prediction
        if constants.RF_PREDICTION in request:
            chained_hash_val = _preprocess_stage(stage=constants.RF_PREDICTION, chained_hash_val=chained_hash_val)

        # Protocol Emulation
        if constants.PROTOCOL_EMULATION in request:
            chained_hash_val = _preprocess_stage(stage=constants.PROTOCOL_EMULATION, chained_hash_val=chained_hash_val)

        # Simulation ID
        def _create_simulation_id(
            processed_request: Dict,
        ):
            """Create the simulation ID from hashing everything together"""
            # pull the non-stage-specific fields
            constituent_hashes = {
                k: processed_request[k] for k in (constants.NUM_TICKS, constants.NUM_BATCHES) if k in processed_request
            }

            # pull all present stage hashes
            if constants.UE_TRACKS_GENERATION in processed_request:
                constituent_hashes[constants.UE_TRACKS_GENERATION_HASH_VAL] = processed_request[
                    constants.UE_TRACKS_GENERATION
                ][constants.HASH_VAL]

            if constants.RF_PREDICTION in processed_request:
                constituent_hashes[constants.RF_PREDICTION_HASH_VAL] = processed_request[constants.RF_PREDICTION][
                    constants.HASH_VAL
                ]

            if constants.PROTOCOL_EMULATION in processed_request:
                constituent_hashes[constants.PROTOCOL_EMULATION_HASH_VAL] = processed_request[
                    constants.PROTOCOL_EMULATION
                ][constants.HASH_VAL]

            # build the one hash
            # one hash to rule them all... throw it into the fire you fool!
            simulation_id = deterministic_hash_dict(constituent_hashes)

            processed_request[constants.SIMULATION_ID] = simulation_id

        # Simulation ID
        _create_simulation_id(
            processed_request=processed_request,
        )

        return processed_request
