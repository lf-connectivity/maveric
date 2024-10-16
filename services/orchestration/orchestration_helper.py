# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Tuple

from radp.common import constants
from radp.common.enums import SimulationStage, WorkflowStatus

logger = logging.getLogger(__name__)


class OrchestrationHelper:
    """Helper class to support operations on simulation metadata"""

    @staticmethod
    def get_stage_params(sim_metadata: Dict, stage: SimulationStage) -> Dict:
        """Helper method to get simulation stage parameters given the stage"""
        try:
            return sim_metadata[stage.value][constants.PARAMS]
        except Exception as e:
            logger.exception(f"Cannot get stage params for stage: {stage.value}")
            raise e

    @staticmethod
    def get_simulation_interval(sim_metadata: Dict) -> Tuple:
        """Get the simulation interval param from simulation object"""
        return sim_metadata[constants.SIMULATION_TIME_INTERVAL]

    @staticmethod
    def get_batching_params(sim_metadata: Dict) -> Tuple:
        """Method to get batch params from simulation object"""
        num_ticks = sim_metadata.get(constants.NUM_TICKS)
        num_batches = sim_metadata.get(constants.NUM_BATCHES)
        return num_ticks, num_batches

    @staticmethod
    def get_next_stage(current_stage: SimulationStage) -> SimulationStage:
        """Helper method to get the next stage in simulation pipeline

        Given the current stage, this method provides the next stage in the RIC
        Simulation pipeline. The next stage may not even be present, however it will
        be returned nonetheless.
        """

        if current_stage == SimulationStage.START:
            return SimulationStage.UE_TRACKS_GENERATION

        if current_stage == SimulationStage.UE_TRACKS_GENERATION:
            return SimulationStage.RF_PREDICTION

        if current_stage == SimulationStage.RF_PREDICTION:
            return SimulationStage.PROTOCOL_EMULATION

        # return finish if next step is not listed on object
        return SimulationStage.FINISH

    @staticmethod
    def get_output_stage(sim_metadata: Dict) -> SimulationStage:
        """Helper method to get last (output) stage from simulation metadata"""
        if sim_metadata.get(SimulationStage.PROTOCOL_EMULATION.value, {}):
            return SimulationStage.PROTOCOL_EMULATION
        elif sim_metadata.get(SimulationStage.RF_PREDICTION.value, {}):
            return SimulationStage.RF_PREDICTION
        else:
            return SimulationStage.UE_TRACKS_GENERATION

    @staticmethod
    def get_rf_digital_twin_model_id(sim_metadata: Dict) -> str:
        """Get the RF digital twin model used in simulation"""
        return sim_metadata[SimulationStage.RF_PREDICTION.value][constants.PARAMS][
            constants.MODEL_ID
        ]

    @staticmethod
    def has_stage(sim_metadata: Dict, stage: SimulationStage) -> bool:
        """Helper method to get a hash_val for a given stage"""
        # safe key lookup
        return stage.value in sim_metadata

    @staticmethod
    def stage_has_hash(sim_metadata: Dict, stage: SimulationStage) -> bool:
        """Helper method to check if a given stage has a hash_val"""
        # safe key lookup
        return constants.HASH_VAL in sim_metadata[stage.value]

    @staticmethod
    def get_stage_hash_val(sim_metadata: Dict, stage: SimulationStage) -> str:
        """Helper method to get a hash_val for a given stage"""
        return sim_metadata[stage.value][constants.HASH_VAL]

    @staticmethod
    def generate_job_event_frame(
        sim_metadata: Dict, stage: SimulationStage, batch=None
    ) -> Dict:
        """Generate a standard frame for job event given a stage"""
        job_frame: Dict[str, Any] = {}

        # each job requires at least the job type and simulation id
        job_frame[constants.JOB_TYPE] = stage.value
        job_frame[constants.SIMULATION_ID] = sim_metadata[constants.SIMULATION_ID]
        if batch is not None:
            job_frame[constants.BATCH] = batch

        # add the stage key to map to stage-specific value
        job_frame[stage.value] = {}

        return job_frame

    @staticmethod
    def stage_has_completed(sim_metadata: Dict, stage: SimulationStage):
        """Check if the stage has completed"""
        _, num_batches = OrchestrationHelper.get_batching_params(sim_metadata)
        if stage == SimulationStage.UE_TRACKS_GENERATION:
            ue_tracks_state = sim_metadata[SimulationStage.UE_TRACKS_GENERATION.value][
                constants.STATE
            ]
            return ue_tracks_state[constants.BATCHES_OUTPUTTED] == num_batches

        if stage == SimulationStage.RF_PREDICTION:
            rf_prediction_state = sim_metadata[SimulationStage.RF_PREDICTION.value][
                constants.STATE
            ]
            return (
                rf_prediction_state[constants.LATEST_BATCH_WITHOUT_FAILURE]
                == num_batches
            )

        if stage == SimulationStage.PROTOCOL_EMULATION:
            protocol_emulation_state = sim_metadata[
                SimulationStage.PROTOCOL_EMULATION.value
            ][constants.STATE]
            return (
                protocol_emulation_state[constants.LATEST_BATCH_WITHOUT_FAILURE]
                == num_batches
            )
        else:
            logger.exception("Received unexpected stage: {stage.value}")
            raise ValueError("Received unexpected stage: {stage.value}")

    @staticmethod
    def update_stage_state_to_finished(sim_metadata: Dict, stage: SimulationStage):
        """Updates stage state status and batches outputted to finished"""
        _, num_batches = OrchestrationHelper.get_batching_params(sim_metadata)
        stage_state = sim_metadata[stage.value][constants.STATE]
        stage_state[constants.STATUS] = WorkflowStatus.FINISHED.value

        # update batches outputted fields based on state
        if stage == SimulationStage.UE_TRACKS_GENERATION:
            stage_state[constants.BATCHES_OUTPUTTED] = num_batches
        else:
            stage_state[constants.LATEST_BATCH_WITHOUT_FAILURE] = num_batches
            stage_state[constants.LATEST_BATCH_TO_SUCCEED] = num_batches
