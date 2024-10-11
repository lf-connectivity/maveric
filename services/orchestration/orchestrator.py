# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from typing import Dict, Optional

from confluent_kafka import Producer

from orchestration.config.kafka import kafka_producer_config
from orchestration.orchestration_helper import OrchestrationHelper
from radp.common import constants
from radp.common.enums import DataSource, OutputStatus, SimulationStage, WorkflowStatus
from radp.common.helpers.file_system_helper import RADPFileSystemHelper
from radp.utility.kafka_utils import produce_object_to_kafka_topic

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator class handles orchestration of simulation events

    The orchestrator will handle the following:
    - orchestration job
    - output events

    Orchestration jobs are created by RADP service to kick off a simulation

    Output events are produced by the services in RIC Simulation Service Stack.
    These outputs indicate the success or failure of a simulation on a batch of
    data.

    The Orchestrator guides a simulation event using the simulation metadata,
    which stores both the simulation parameters and simulation state. This
    metadata is a json file stored in the simulation's specific folder in the
    filesystem.

    Example simulation state:
    {
        "num_batches": 1,
        "num_ticks": 1000,
        "rf_prediction": {
            "hash_val": "0f2a3981884fad1b22a8e3006f472a12",
            "params": {
                "config_id": "test_config_1",
                "model_id": "test_dt"
            },
            "state": {
                "batches_retrying": [],
                "latest_batch_to_succeed": 1,
                "latest_batch_without_failure": 1,
                "status": "finished"
            }
        },
        "simulation_id": "c70c0e05570a1afe76c08de3703e8073",
        "simulation_status": "finished",
        "simulation_time_interval_seconds": 0.01,
        "ue_tracks_generation": {
            "hash_val": "6a642f0fb23b2f1e32805af3c12f674f",
            "params": {
                "gauss_markov_params": {
                    "alpha": 0.5,
                    "lon_x_dims": 100,
                    "lon_y_dims": 100,
                    "rng_seed": 42,
                    "variance": 0.8
                },
                "lat_lon_boundaries": {
                    "max_lat": 90,
                    "max_lon": 180,
                    "min_lat": -90,
                    "min_lon": -180
                },
                "simulation_duration_seconds": 10,
                "ue_class_distribution": {
                    "cars": {
                        "count": 5,
                        "velocity": 1,
                        "velocity_variance": 1
                    },
                    "cyclists": {
                        "count": 5,
                        "velocity": 1,
                        "velocity_variance": 1
                    },
                    "pedestrian": {
                        "count": 5,
                        "velocity": 1,
                        "velocity_variance": 1
                    },
                    "stationary": {
                        "count": 5,
                        "velocity": 1,
                        "velocity_variance": 1
                    }
                }
            },
            "state": {
                "batches_outputted": 1,
                "status": "finished"
            }
        }
    }
    """

    def __init__(self):
        """Set up the orchestrator's producer instance"""
        self.producer = Producer(kafka_producer_config)

    def handle_orchestration_job(self, job_data: Dict):
        """
        Top level handler for an orchestration job.

        Starts orchestrating a simulation based on the simulation event saved

        job format:
        {
            job_type: str
            simulation_id: str
        }
        """

        logger.info(f"Handling orchestration job: {job_data}")

        # pull simulation id to find metadata
        simulation_id = job_data[constants.SIMULATION_ID]
        sim_metadata = RADPFileSystemHelper.load_simulation_metadata(simulation_id)

        job_id = job_data[constants.JOB_ID]

        # get first stage in simulation and first non-cached stage
        first_present_stage = self._get_first_present_stage(sim_metadata)
        first_non_cached_stage = self._get_first_non_cached_stage(sim_metadata)

        if first_non_cached_stage == SimulationStage.FINISH:
            # all present layers cached, wrap up simulation
            self._wrap_up_simulation(sim_metadata, simulation_id)
        else:
            # input data must be in cache if the first non-cached stage is not the first present stage
            input_data_is_cached = first_present_stage != first_non_cached_stage

            logger.info(
                f"Running first non-cached stage: {first_non_cached_stage.name}"
            )
            self._run_first_non_cached_stage(
                sim_metadata,
                simulation_id,
                first_non_cached_stage,
                input_data_is_cached,
            )

        sim_metadata[constants.JOB_ID] = job_id
        sim_metadata[constants.JOB_FINISHED_DATETIME] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        # save state after orchestration
        RADPFileSystemHelper.save_simulation_metadata(sim_metadata, simulation_id)

    def handle_output_event(self, output_event: Dict):
        """
        Top level handler for an output event

        """

        # pull simulation id to find metadata
        simulation_id = output_event[constants.SIMULATION_ID]
        sim_metadata = RADPFileSystemHelper.load_simulation_metadata(simulation_id)

        output_service = output_event[constants.SERVICE]

        if output_service == SimulationStage.UE_TRACKS_GENERATION.value:
            self._handle_ue_tracks_generation_output(
                sim_metadata, output_event, simulation_id
            )
        elif output_service == SimulationStage.RF_PREDICTION.value:
            self._handle_rf_prediction_output(sim_metadata, output_event, simulation_id)
        else:
            self._handle_protocol_emulation_output(
                sim_metadata, output_event, simulation_id
            )

        # save state after handling output
        RADPFileSystemHelper.save_simulation_metadata(sim_metadata, simulation_id)

    def _get_first_present_stage(self, sim_metadata: Dict) -> SimulationStage:
        """Get the first stage present within simulation metadata"""
        stage = SimulationStage.START

        while not OrchestrationHelper.has_stage(sim_metadata, stage):
            stage = OrchestrationHelper.get_next_stage(stage)
        return stage

    def _get_first_non_cached_stage(self, sim_metadata: Dict) -> SimulationStage:
        """Get the first stage which is present and NOT cached"""
        stage = self._get_first_present_stage(sim_metadata)

        # find first non-cached stage if one exists
        while self._stage_data_is_cached(sim_metadata, stage):
            logger.info(f"'{stage.name}' output found in cache. Skipping stage...")
            stage = OrchestrationHelper.get_next_stage(stage)

        # return stage only if it exists in simulation
        if OrchestrationHelper.has_stage(sim_metadata, stage):
            return stage
        else:
            return SimulationStage.FINISH

    def _run_first_non_cached_stage(
        self,
        sim_metadata: Dict,
        simulation_id: str,
        first_non_cached_stage: SimulationStage,
        input_data_is_cached: bool,
    ) -> None:
        """Run the first non-cached stage"""

        if first_non_cached_stage == SimulationStage.UE_TRACKS_GENERATION:
            # stage is UE Tracks Generation --> no input data, data is generated
            return self._start_ue_tracks_generation(sim_metadata, simulation_id)
        elif (
            first_non_cached_stage == SimulationStage.RF_PREDICTION
            and input_data_is_cached
        ):
            # stage is RF Prediction + input data is cached
            # --> start all jobs (which pulls from cache)
            return self._start_all_rf_prediction_jobs(sim_metadata, simulation_id)
        elif first_non_cached_stage == SimulationStage.RF_PREDICTION:
            # stage is RF Prediction + data is inputted to simulation
            # --> start single job using data in simulation folder
            return self._start_single_rf_prediction_job(
                sim_metadata, simulation_id, batch=1, data_source=DataSource.USER_INPUT
            )
        elif (
            first_non_cached_stage == SimulationStage.PROTOCOL_EMULATION
            and input_data_is_cached
        ):
            # stage is RF Prediction + input data is cached
            # --> start all jobs (which pulls from cache)
            return self._start_all_protocol_emulation_jobs(sim_metadata, simulation_id)
        else:
            # stage is Protocol Emulation + data is inputted to simulation
            # --> start single job using data in simulation folder
            return self._start_single_protocol_emulation_job(
                sim_metadata, simulation_id, batch=1, data_source=DataSource.USER_INPUT
            )

    def _stage_data_is_cached(self, sim_metadata: Dict, stage: SimulationStage) -> bool:
        """Check whether the output of a simulation event stage is cached"""
        # check if stage and hash value present
        if not OrchestrationHelper.has_stage(sim_metadata, stage=stage):
            return False
        if not OrchestrationHelper.stage_has_hash(sim_metadata, stage=stage):
            return False

        # pull the hash value for given stage
        hash_val = OrchestrationHelper.get_stage_hash_val(sim_metadata, stage=stage)
        return RADPFileSystemHelper.hash_val_found_in_output_folder(stage, hash_val)

    def _start_ue_tracks_generation(self, sim_metadata: Dict, simulation_id: str):
        """
        Starts a UE tracks generation job

        UE Tracks Generation job event:
        {
            "job_type": "ue_tracks_generation",
            "simulation_id": <simulation_id>,
            "ue_tracks_generation": {
                "output_file_prefix": "",
                "params": {
                    "simulation_duration_seconds": int,
                    "simulation_time_interval_seconds": float,
                    "num_ticks": int,
                    "num_batches": int,
                    ... <ue tracks generation specific params>
                }
            }
        }
        """
        # pull the job parameters from the simulation metadata
        ue_tracks_params = {}
        ue_tracks_params.update(
            OrchestrationHelper.get_stage_params(
                sim_metadata, stage=SimulationStage.UE_TRACKS_GENERATION
            )
        )

        # pull the interval
        time_interval = OrchestrationHelper.get_simulation_interval(sim_metadata)

        # pull the batching params
        num_ticks, num_batches = OrchestrationHelper.get_batching_params(sim_metadata)

        # pull hash_val and build output file prefix
        hash_val = OrchestrationHelper.get_stage_hash_val(
            sim_metadata, stage=SimulationStage.UE_TRACKS_GENERATION
        )
        output_file_prefix = f"{SimulationStage.UE_TRACKS_GENERATION.value}-{hash_val}"

        # build the ue tracks generation job event
        ue_tracks_params[constants.SIMULATION_TIME_INTERVAL] = time_interval
        ue_tracks_params[constants.NUM_TICKS] = num_ticks
        ue_tracks_params[constants.NUM_BATCHES] = num_batches

        # generate the kafka job frame
        ue_tracks_job = OrchestrationHelper.generate_job_event_frame(
            sim_metadata, SimulationStage.UE_TRACKS_GENERATION
        )

        # supply stage-specific fields
        ue_tracks_job[SimulationStage.UE_TRACKS_GENERATION.value].update(
            {
                constants.OUTPUT_FILE_PREFIX: output_file_prefix,
                constants.PARAMS: ue_tracks_params,
            }
        )

        # produce ue_tracks_generation job to jobs topic
        produce_object_to_kafka_topic(
            self.producer, topic=constants.KAFKA_JOBS_TOPIC_NAME, value=ue_tracks_job
        )
        logger.info(
            f"Produced UE tracks generation job for simulation: {simulation_id}"
        )

        # Update state to show UE tracks generation has started
        ue_tracks_state = sim_metadata[SimulationStage.UE_TRACKS_GENERATION.value][
            constants.STATE
        ]
        ue_tracks_state[constants.STATUS] = WorkflowStatus.IN_PROGRESS.value

    def _start_all_rf_prediction_jobs(self, sim_metadata: Dict, simulation_id: str):
        """Start and RF Prediction job for every batch"""
        # pull number of batches to start
        _, num_batches = OrchestrationHelper.get_batching_params(sim_metadata)

        logger.info(
            f"Starting all {num_batches} RF Prediction jobs for simulation: {simulation_id}"
        )
        for i in range(1, num_batches + 1):
            self._start_single_rf_prediction_job(
                sim_metadata, simulation_id, batch=i, data_source=DataSource.CACHE
            )

    def _start_single_rf_prediction_job(
        self,
        sim_metadata: Dict,
        simulation_id: str,
        batch: int,
        data_source: DataSource,
    ):
        """
        Starts a single RF prediction job

        RF Prediction job event:
        {
            "job_type": "rf_prediction",
            "simulation_id": "123",
            "batch": 3,
            "rf_prediction": {
                "ue_data_file_path": "",
                "output_file_path": "",
                "model_id": "",
                "model_file": "",
                "config_file_path": "",
                "topology_file_path": "",
                "params": {},
            }
        }
        """

        # check if input is from simulation folder or UE tracks generation output layer
        if data_source == DataSource.USER_INPUT:
            # input is in simulation folder, get its path
            ue_data_file_path = RADPFileSystemHelper.gen_simulation_ue_data_file_path(
                simulation_id
            )
        else:
            # input is cached in UE tracks generation output layer, get its path using its hash
            ue_tracks_hash_val = OrchestrationHelper.get_stage_hash_val(
                sim_metadata, SimulationStage.UE_TRACKS_GENERATION
            )
            ue_data_file_path = RADPFileSystemHelper.gen_stage_output_file_path(
                stage=SimulationStage.UE_TRACKS_GENERATION,
                hash_val=ue_tracks_hash_val,
                batch=batch,
            )

        # get output file path using the RF Prediction hash
        rf_prediction_hash_val = OrchestrationHelper.get_stage_hash_val(
            sim_metadata, SimulationStage.RF_PREDICTION
        )
        output_file_path = RADPFileSystemHelper.gen_stage_output_file_path(
            stage=SimulationStage.RF_PREDICTION,
            hash_val=rf_prediction_hash_val,
            batch=batch,
        )

        # get the model id and file path
        model_id = OrchestrationHelper.get_rf_digital_twin_model_id(sim_metadata)
        model_file_path = RADPFileSystemHelper.gen_model_file_path(model_id)

        # get the config and topology file paths
        config_file_path = RADPFileSystemHelper.gen_simulation_cell_config_file_path(
            simulation_id
        )
        topology_file_path = RADPFileSystemHelper.gen_model_topology_file_path(model_id)

        # get the RF prediction params
        rf_prediction_params = OrchestrationHelper.get_stage_params(
            sim_metadata, stage=SimulationStage.RF_PREDICTION
        )

        # build the rf prediction job
        rf_prediction_job = OrchestrationHelper.generate_job_event_frame(
            sim_metadata, SimulationStage.RF_PREDICTION, batch=batch
        )
        rf_prediction_job[SimulationStage.RF_PREDICTION.value].update(
            {
                constants.UE_DATA_FILE_PATH_KEY: ue_data_file_path,
                constants.OUTPUT_FILE_PATH: output_file_path,
                constants.MODEL_ID: model_id,
                constants.MODEL_FILE: model_file_path,
                constants.CONFIG_FILE_PATH_KEY: config_file_path,
                constants.TOPOLOGY_FILE_PATH_KEY: topology_file_path,
                constants.PARAMS: rf_prediction_params,
            }
        )

        # produce rf_prediction_job job to jobs topic
        produce_object_to_kafka_topic(
            self.producer,
            topic=constants.KAFKA_JOBS_TOPIC_NAME,
            value=rf_prediction_job,
        )
        logger.info(
            f"Produced RF Prediction job batch {batch} for simulation: {simulation_id}"
        )

        # Update state to show RF Prediction has started
        if batch == 1:
            rf_prediction_state = sim_metadata[SimulationStage.RF_PREDICTION.value][
                constants.STATE
            ]
            rf_prediction_state[constants.STATUS] = WorkflowStatus.IN_PROGRESS.value

    def _start_all_protocol_emulation_jobs(
        self,
        sim_metadata: Dict,
        simulation_id: str,
    ):
        """Start all protocol emulation jobs"""
        # TODO: implement this once protocol emulation service is implemented
        pass

    def _start_single_protocol_emulation_job(
        self,
        sim_metadata: Dict,
        simulation_id: str,
        batch: int,
        data_source: DataSource,
    ):
        """Starts a Protocol Emulation job"""
        # TODO: implement this once protocol emulation service is implemented
        pass

    def _handle_ue_tracks_generation_output(
        self, sim_metadata: Dict, ue_tracks_output: Dict, simulation_id: str
    ):
        """Handle a UE tracks generation output"""
        # TODO: handle a failed job.  We'll probably want to add the original job event to the output object
        # to make retry easier
        if ue_tracks_output[constants.STATUS] == OutputStatus.FAILURE.value:
            logger.exception(
                f"Simulation {simulation_id} failed in UE tracks generation stage: {ue_tracks_output}"
            )
            raise Exception(
                f"Simulation {simulation_id} failed in UE tracks generation stage: {ue_tracks_output}"
            )

        # pull state and batch
        ue_tracks_state = sim_metadata[SimulationStage.UE_TRACKS_GENERATION.value][
            constants.STATE
        ]
        batch = ue_tracks_output[constants.BATCH]

        # update the outputted batches field
        ue_tracks_state[constants.BATCHES_OUTPUTTED] = batch

        # check if the stage has completed, update stage status if so
        stage_completed = OrchestrationHelper.stage_has_completed(
            sim_metadata, SimulationStage.UE_TRACKS_GENERATION
        )
        if stage_completed:
            ue_tracks_state[constants.STATUS] = WorkflowStatus.FINISHED.value

        if OrchestrationHelper.has_stage(
            sim_metadata, stage=SimulationStage.RF_PREDICTION
        ):
            self._start_single_rf_prediction_job(
                sim_metadata, simulation_id, batch, data_source=DataSource.CACHE
            )
        elif stage_completed:
            # this was the last stage, move to wrap up
            self._wrap_up_simulation(sim_metadata, simulation_id)
        else:
            # this was last stage but it hasn't completed yet
            # save simulation metadata and exit
            pass

    def _handle_rf_prediction_output(
        self, sim_metadata: Dict, rf_prediction_output: Dict, simulation_id: str
    ):
        """Handle an RF prediction output"""
        # TODO: handle a failed job.  We'll probably want to add the original job event to the output object
        # to make retry easier
        if rf_prediction_output[constants.STATUS] == OutputStatus.FAILURE.value:
            logger.exception(
                f"Simulation {simulation_id} failed in RF Prediction stage: {rf_prediction_output}"
            )
            raise Exception(
                f"Simulation {simulation_id} failed in RF Prediction stage: {rf_prediction_output}"
            )

        # pull state and batch
        rf_prediction_state = sim_metadata[SimulationStage.RF_PREDICTION.value][
            constants.STATE
        ]
        batch = rf_prediction_output[constants.BATCH]

        # TODO: update this to account for past failures once
        # RIGHT NOW we're assuming everything is succeeding perfectly
        # update the outputted batches field
        rf_prediction_state[constants.LATEST_BATCH_WITHOUT_FAILURE] = batch
        rf_prediction_state[constants.LATEST_BATCH_TO_SUCCEED] = batch

        # check if the stage has completed, update stage status if so
        stage_completed = OrchestrationHelper.stage_has_completed(
            sim_metadata, SimulationStage.RF_PREDICTION
        )
        if stage_completed:
            rf_prediction_state[constants.STATUS] = WorkflowStatus.FINISHED.value

        if OrchestrationHelper.has_stage(
            sim_metadata, stage=SimulationStage.PROTOCOL_EMULATION
        ):
            self._start_single_protocol_emulation_job(
                sim_metadata,
                simulation_id,
                batch=batch,
                data_source=DataSource.CACHE,
            )
        elif stage_completed:
            # this was the last stage, move to wrap up
            self._wrap_up_simulation(sim_metadata, simulation_id)
        else:
            # this was last stage but it hasn't completed yet
            # save simulation metadata and exit
            pass

    def _handle_protocol_emulation_output(
        self,
        sim_metadata: Dict,
        output_event: Dict,
        simulation_id: str,
    ):
        """Handle a protocol eumlation output"""
        # TODO: implement this once protocol emulation service is implemented
        pass

    def _wrap_up_simulation(self, sim_metadata: Dict, simulation_id: str):
        """
        Wrap up a simulation event

        This involves the following:
        - output the last layer to consumable location
        - update the status of simulation in metadata
        - clean up any output data not from most recent successful execution
        - log that wrap up is complete
        """

        logger.info(f"Starting wrap-up of simulation event: {simulation_id}")

        # zip output files to consume folder
        self._copy_simulation_output_to_consume_folder(sim_metadata, simulation_id)

        # clean any unused output data
        self._clean_unused_output_data(sim_metadata, simulation_id)

        # update simulation status to finished
        sim_metadata[constants.SIMULATION_STATUS] = WorkflowStatus.FINISHED.value
        logger.info(f"Successfully completed simulation: {simulation_id}")

    def _copy_simulation_output_to_consume_folder(
        self, sim_metadata: Dict, simulation_id: str
    ):
        """Copy simulation output to consumable zip file

        Consumable zip file will allow the user to download data
        and system will delete afterwords.
        """
        logger.debug(
            f"Copying output from simulation: {simulation_id} to consumable zip file"
        )

        # get the output stage
        simulation_output_stage = OrchestrationHelper.get_output_stage(sim_metadata)
        hash_val = OrchestrationHelper.get_stage_hash_val(
            sim_metadata, stage=simulation_output_stage
        )

        # pull number of batches to zip
        _, num_batches = OrchestrationHelper.get_batching_params(sim_metadata)

        # zip files together into consumable output folder
        RADPFileSystemHelper.zip_output_files_to_simulation_folder_as_csvs(
            simulation_id,
            stage=simulation_output_stage,
            hash_val=hash_val,
            num_batches=num_batches,
        )

    def _clean_unused_output_data(self, sim_metadata: Dict, simulation_id: str):
        """Clean unused output data"""
        logger.info(
            f"Removing all outputs not used in recent simulation: {simulation_id}"
        )

        if OrchestrationHelper.has_stage(
            sim_metadata, stage=SimulationStage.UE_TRACKS_GENERATION
        ):
            # clean UE Tracks Generation outputs
            ue_tracks_generation_hash_val: Optional[
                str
            ] = OrchestrationHelper.get_stage_hash_val(
                sim_metadata,
                stage=SimulationStage.UE_TRACKS_GENERATION,
            )
            RADPFileSystemHelper.clear_output_data_from_stage(
                stage=SimulationStage.UE_TRACKS_GENERATION,
                save_hash_val=ue_tracks_generation_hash_val,
            )

        if OrchestrationHelper.has_stage(
            sim_metadata, stage=SimulationStage.RF_PREDICTION
        ):
            # clean RF Prediction outputs
            rf_prediction_hash_val: Optional[
                str
            ] = OrchestrationHelper.get_stage_hash_val(
                sim_metadata,
                stage=SimulationStage.RF_PREDICTION,
            )
            RADPFileSystemHelper.clear_output_data_from_stage(
                stage=SimulationStage.RF_PREDICTION,
                save_hash_val=rf_prediction_hash_val,
            )

        if OrchestrationHelper.has_stage(
            sim_metadata, stage=SimulationStage.PROTOCOL_EMULATION
        ):
            # clean Protocol Emulation outputs
            protocol_emulation_hash_val: Optional[
                str
            ] = OrchestrationHelper.get_stage_hash_val(
                sim_metadata,
                stage=SimulationStage.PROTOCOL_EMULATION,
            )
            RADPFileSystemHelper.clear_output_data_from_stage(
                stage=SimulationStage.PROTOCOL_EMULATION,
                save_hash_val=protocol_emulation_hash_val,
            )
        logger.info(
            f"Successfully removed all unused outputs for simulation: {simulation_id}"
        )
