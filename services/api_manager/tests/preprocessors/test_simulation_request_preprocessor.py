# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Dict

from api_manager.preprocessors.simulation_request_preprocessor import (
    RICSimulationRequestPreprocessor,
    deterministic_hash_dict,
)


class TestRICSimulationRequestPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "simulation_duration_seconds": 3600,
                    "ue_class_distribution": {
                        "stationary": {
                            "count": 0,
                            "velocity": 1,
                            "velocity_variance": 1,
                        },
                        "pedestrian": {
                            "count": 0,
                            "velocity": 1,
                            "velocity_variance": 1,
                        },
                        "cyclist": {"count": 0, "velocity": 1, "velocity_variance": 1},
                        "car": {"count": 0, "velocity": 1, "velocity_variance": 1},
                    },
                    "gauss_markov_params": {"alpha": 0.5, "variance": 0.8},
                }
            },
            "rf_prediction": {"model_id": "detroit_10", "config_id": "config_1"},
            "protocol_emulation": {"ttt_seconds": 1, "hysteresis": 2},
        }
        cls.processed_request = {
            "simulation_status": "planned",
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks_generation": {
                "params": {
                    "simulation_duration_seconds": 3600,
                    "ue_class_distribution": {
                        "stationary": {
                            "count": 0,
                            "velocity": 1,
                            "velocity_variance": 1,
                        },
                        "pedestrian": {
                            "count": 0,
                            "velocity": 1,
                            "velocity_variance": 1,
                        },
                        "cyclist": {"count": 0, "velocity": 1, "velocity_variance": 1},
                        "car": {"count": 0, "velocity": 1, "velocity_variance": 1},
                    },
                    "gauss_markov_params": {"alpha": 0.5, "variance": 0.8},
                },
                "hash_val": "4c18d1024239f1e5bcc6cf138958b77a",
                "state": {"status": "planned", "batches_outputted": 0},
            },
            "num_ticks": 360000,
            "num_batches": 1,
            "rf_prediction": {
                "params": {"model_id": "detroit_10", "config_id": "config_1"},
                "hash_val": "c479dbbc80dc2015b10c2f2cc0acfec1",
                "state": {
                    "status": "planned",
                    "latest_batch_without_failure": 0,
                    "latest_batch_to_succeed": 0,
                    "batches_retrying": [],
                },
            },
            "protocol_emulation": {
                "params": {"ttt_seconds": 1, "hysteresis": 2},
                "hash_val": "0dc200d7c900a3b308544af4cd0084f9",
                "state": {
                    "status": "planned",
                    "latest_batch_without_failure": 0,
                    "latest_batch_to_succeed": 0,
                    "batches_retrying": [],
                },
            },
            "simulation_id": "0f496803756687776509623fd7db4741",
        }

    def _test_ue_tracks(
        self,
        request: Dict,
        processed_request: Dict,
        files: Dict,
        ue_tracks_specifier_key: str,
    ):
        chained_hash_val = deterministic_hash_dict(
            {"simulation_time_interval_seconds": request["simulation_time_interval_seconds"]}
        )

        self.assertEqual(
            request["ue_tracks"][ue_tracks_specifier_key],
            processed_request["ue_tracks_generation"]["params"],
        )
        self.assertEqual(
            deterministic_hash_dict(
                {
                    "predecessor_stage_hash_val": chained_hash_val,
                    **request["ue_tracks"],
                }
            ),
            processed_request["ue_tracks_generation"]["hash_val"],
        )
        self.assertEqual(
            {"status": "planned", "batches_outputted": 0},
            processed_request["ue_tracks_generation"]["state"],
        )

    def _test_phase(
        self,
        request: Dict,
        processed_request: Dict,
        phase: str,
        predecessor_phase: str,
        files: Dict,
    ):
        """Can be used to test all but the first phase."""

        self.assertEqual(request[phase], processed_request[phase]["params"])
        # self.assertEqual(
        #     deterministic_hash_dict(
        #         {
        #             "predecessor_phase_hash_val": (
        #                 processed_request[predecessor_phase]["hash_val"]
        #             ),
        #             **request[phase],
        #         }
        #     ),
        #     processed_request[phase]["hash_val"],
        # )
        self.assertEqual(
            {
                "status": "planned",
                "latest_batch_without_failure": 0,
                "latest_batch_to_succeed": 0,
                "batches_retrying": [],
            },
            processed_request[phase]["state"],
        )

    def test_preprocess(self):
        request = {}
        files = {}

        # invalid
        with self.assertRaises(KeyError) as _:
            RICSimulationRequestPreprocessor.preprocess(request, files)

        # valid
        request = copy.deepcopy(self.request)  # copied
        processed_request = RICSimulationRequestPreprocessor.preprocess(request, files)
        self._test_ue_tracks(
            request=request,
            processed_request=processed_request,
            files=files,
            ue_tracks_specifier_key="ue_tracks_generation",
        )

        self.assertEqual(
            processed_request["ue_tracks_generation"],
            self.processed_request["ue_tracks_generation"],
        )
        self._test_phase(
            request=request,
            processed_request=processed_request,
            files=files,
            phase="rf_prediction",
            predecessor_phase="ue_tracks_generation",
        )
        self.assertEqual(
            processed_request["rf_prediction"],
            self.processed_request["rf_prediction"],
        )
        self._test_phase(
            request=request,
            processed_request=processed_request,
            files=files,
            phase="protocol_emulation",
            predecessor_phase="rf_prediction",
        )
        self.assertEqual(
            processed_request["protocol_emulation"],
            self.processed_request["protocol_emulation"],
        )

        # now, test the whole thing!
        self.assertEqual(
            processed_request,
            self.processed_request,
        )

        # valid, user-provided UE tracks
        request = copy.deepcopy(self.request)  # copied
        del request["ue_tracks"]["ue_tracks_generation"]
        request["ue_tracks"]["ue_data_id"] = "<ue_data_id>"
        processed_request = RICSimulationRequestPreprocessor.preprocess(request, files)

        assert "ue_tracks_generation" not in processed_request

        self._test_phase(
            request=request,
            processed_request=processed_request,
            files=files,
            phase="rf_prediction",
            predecessor_phase="ue_tracks_generation",
        )
        self._test_phase(
            request=request,
            processed_request=processed_request,
            files=files,
            phase="protocol_emulation",
            predecessor_phase="rf_prediction",
        )

    def test_deterministic_hash_dict(self):
        dict1 = {
            "Key1": 1,
            "Key2": {
                "Key2_SubKey1": 2,
                "Key2_SubKey2": 3,
            },
            "Key3": {
                "Key3_SubKey1": 2,
                "Key3_SubKey2": {"Key3_SubKey2_SubSubKey1": 3},
            },
        }

        # jumble it up -- but, keep the semantic content the same
        dict2 = {
            "Key2": {
                "Key2_SubKey2": 3,
                "Key2_SubKey1": 2,
            },
            "Key3": {
                "Key3_SubKey2": {"Key3_SubKey2_SubSubKey1": 3},
                "Key3_SubKey1": 2,
            },
            "Key1": 1,
        }

        self.assertEqual(deterministic_hash_dict(dict1), deterministic_hash_dict(dict2))
