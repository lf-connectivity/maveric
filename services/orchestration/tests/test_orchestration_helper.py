# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import patch

from orchestration.orchestration_helper import OrchestrationHelper
from radp.common.enums import SimulationStage

dummy_sim_metadata = {
    "ue_tracks_generation": {"params": "dummy_val_ue_tracks_generation"},
    "rf_prediction": {"params": "dummy_val_rf_prediction"},
    "protocol_emulation": {"params": "dummy_val_protocol_emulation"},
    "simulation_time_interval_seconds": "dummy_sim_time_interval_val",
    "num_ticks": "dummy_num_ticks_val",
    "num_batches": "dummy_num_batches_val",
}


class TestOrchestrationHelper(TestCase):
    def test_get_stage_params_ue(self):
        for stage in [
            SimulationStage.UE_TRACKS_GENERATION,
            SimulationStage.RF_PREDICTION,
            SimulationStage.PROTOCOL_EMULATION,
        ]:
            self.assertEqual(
                OrchestrationHelper.get_stage_params(
                    dummy_sim_metadata,
                    stage,
                ),
                f"dummy_val_{stage.value}",
            )

    def test_get_stage_params_exception(self):
        with self.assertRaises(Exception) as e:
            OrchestrationHelper.get_stage_params(
                dummy_sim_metadata,
                SimulationStage.START,
            )
        with self.assertRaises(Exception) as e2:
            OrchestrationHelper.get_stage_params(
                dummy_sim_metadata,
                SimulationStage.FINISH,
            )
        self.assertEqual(
            str(e.exception),
            "'start'",
        )
        self.assertEqual(
            str(e2.exception),
            "'finish'",
        )

    def test_get_simulation_interval(self):
        self.assertEqual(
            OrchestrationHelper.get_simulation_interval(
                dummy_sim_metadata,
            ),
            "dummy_sim_time_interval_val",
        )

    def test_get_batching_params(self):
        dummy_ticks, dummy_batches = OrchestrationHelper.get_batching_params(
            dummy_sim_metadata,
        )
        self.assertEqual(
            dummy_ticks,
            "dummy_num_ticks_val",
        )
        self.assertEqual(
            dummy_batches,
            "dummy_num_batches_val",
        )

    @patch(
        "radp.common.constants.NUM_TICKS",
        "dummy_num_ticks_key_exception",
    )
    @patch(
        "radp.common.constants.NUM_BATCHES",
        "dummy_num_batches_key_exception",
    )
    def test_get_batching_params_invalid(self):
        dummy_ticks, dummy_batches = OrchestrationHelper.get_batching_params(
            dummy_sim_metadata,
        )
        self.assertEqual(
            dummy_ticks,
            None,
        )
        self.assertEqual(
            dummy_batches,
            None,
        )

    def test_get_next_stage(self):
        self.assertEqual(
            OrchestrationHelper.get_next_stage(SimulationStage.START),
            SimulationStage.UE_TRACKS_GENERATION,
        )
        self.assertEqual(
            OrchestrationHelper.get_next_stage(SimulationStage.UE_TRACKS_GENERATION),
            SimulationStage.RF_PREDICTION,
        )
        self.assertEqual(
            OrchestrationHelper.get_next_stage(SimulationStage.RF_PREDICTION),
            SimulationStage.PROTOCOL_EMULATION,
        )
        self.assertEqual(
            OrchestrationHelper.get_next_stage(SimulationStage.PROTOCOL_EMULATION),
            SimulationStage.FINISH,
        )

    def test_get_output_stage(self):
        sim_metadata = {"ue_tracks_generation": {"dummy_params": "dummy_val_rf"}}
        self.assertEqual(
            OrchestrationHelper.get_output_stage(sim_metadata),
            SimulationStage.UE_TRACKS_GENERATION,
        )

        sim_metadata.pop("ue_tracks_generation", "invalid stage")
        sim_metadata["rf_prediction"] = {"dummy_params": "dummy_val_rf"}
        self.assertEqual(
            OrchestrationHelper.get_output_stage(sim_metadata),
            SimulationStage.RF_PREDICTION,
        )

        sim_metadata.pop("rf_prediction", "invalid stage")
        sim_metadata["protocol_emulation"] = {"dummy_params": "dummy_val_pe"}
        self.assertEqual(
            OrchestrationHelper.get_output_stage(sim_metadata),
            SimulationStage.PROTOCOL_EMULATION,
        )

    def test_get_rf_digital_twin_model_id(self):
        sim_metadata_dt = {"rf_prediction": {"params": {"model_id": "dummy_val"}}}
        self.assertEqual(
            OrchestrationHelper.get_rf_digital_twin_model_id(sim_metadata_dt),
            "dummy_val",
        )

    def test_has_stage(self):
        self.assertTrue(OrchestrationHelper.has_stage(dummy_sim_metadata, SimulationStage.RF_PREDICTION))

    def test_stage_missing(self):
        self.assertFalse(OrchestrationHelper.has_stage(dummy_sim_metadata, SimulationStage.START))

    def test_has_hash(self):
        sim_metadata_hash = {"rf_prediction": {"hash_val": "val"}}
        self.assertTrue(
            OrchestrationHelper.stage_has_hash(
                sim_metadata_hash,
                SimulationStage.RF_PREDICTION,
            )
        )

    def test_hash_missing(self):
        sim_metadata_hash = {"rf_prediction": {}}
        self.assertFalse(
            OrchestrationHelper.stage_has_hash(
                sim_metadata_hash,
                SimulationStage.RF_PREDICTION,
            )
        )

    def test_get_stage_hash_val(self):
        sim_metadata_stage_hash = {"rf_prediction": {"hash_val": "val"}}
        self.assertEqual(
            OrchestrationHelper.get_stage_hash_val(
                sim_metadata_stage_hash,
                SimulationStage.RF_PREDICTION,
            ),
            "val",
        )

    def test_generate_job_event_frame(self):
        sim_metadata_event = {"simulation_id": "dummy_id"}
        self.assertEqual(
            OrchestrationHelper.generate_job_event_frame(
                sim_metadata_event,
                SimulationStage.RF_PREDICTION,
                "batch",
            ),
            {
                "job_type": "rf_prediction",
                "simulation_id": "dummy_id",
                "batch": "batch",
                "rf_prediction": {},
            },
        )

    def test_stage_has_completed(self):
        dummy_sim_data = {
            "num_ticks": "dummy_num_ticks_val",
            "num_batches": "dummy_num_batches_val",
            "ue_tracks_generation": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
            "rf_prediction": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
            "protocol_emulation": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
        }
        num_ticks, num_batches = OrchestrationHelper.get_batching_params(dummy_sim_data)

        self.assertTrue(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.UE_TRACKS_GENERATION))
        self.assertTrue(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.RF_PREDICTION))
        self.assertTrue(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.PROTOCOL_EMULATION))

    def test_stage_has_completed_exception(self):
        dummy_sim_data = {
            "num_ticks": "dummy_num_ticks_neg",
            "num_batches": "dummy_num_batches_neg",
            "ue_tracks_generation": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
            "rf_prediction": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
            "protocol_emulation": {
                "state": {
                    "batches_outputted": "dummy_num_batches_val",
                    "latest_batch_without_failure": "dummy_num_batches_val",
                }
            },
        }
        self.assertFalse(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.UE_TRACKS_GENERATION))
        self.assertFalse(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.RF_PREDICTION))
        self.assertFalse(OrchestrationHelper.stage_has_completed(dummy_sim_data, SimulationStage.PROTOCOL_EMULATION))

    def test_update_stage_state_to_finished(self):
        dummy_sim_data = {
            "num_ticks": "dummy_num_ticks_val",
            "num_batches": "dummy_num_batches_val",
            "ue_tracks_generation": {
                "state": {
                    "latest_batch_without_failure": "dummy_num_batches_val",
                    "latest_batch_to_succeed": "dummy_num_batches_val",
                }
            },
            "rf_prediction": {"state": {"batches_outputted": "dummy_num_batches_val"}},
            "protocol_emulation": {"state": {"batches_outputted": "dummy_num_batches_val"}},
        }
        stage_ue_track_generation = SimulationStage.UE_TRACKS_GENERATION
        OrchestrationHelper.update_stage_state_to_finished(dummy_sim_data, stage_ue_track_generation)
        self.assertEqual(
            dummy_sim_data[stage_ue_track_generation.value]["state"],
            {
                "batches_outputted": "dummy_num_batches_val",
                "latest_batch_without_failure": "dummy_num_batches_val",
                "latest_batch_to_succeed": "dummy_num_batches_val",
                "status": "finished",
            },
        )
