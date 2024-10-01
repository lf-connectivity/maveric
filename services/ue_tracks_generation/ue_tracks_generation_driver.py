# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from confluent_kafka import Producer

from radp.common import constants
from radp.common.enums import OutputStatus
from radp.digital_twin.mobility.ue_tracks import MobilityClass, UETracksGenerator
from radp.digital_twin.utils.gis_tools import GISTools
from radp.utility.kafka_utils import produce_object_to_kafka_topic
from radp.utility.pandas_utils import write_feather_df
from ue_tracks_generation.config.kafka import kafka_producer_config
from ue_tracks_generation.ue_tracks_generation_helper import UETracksGenerationHelper

logger = logging.getLogger(__name__)


class UETracksGenerationDriver:
    """
    The UETracksGenerationDriver Class handles execution of the UE Tracks Generation jobs

    The UE Tracks Generation Service will take in as input an UE Tracks Generation job
    with the following format:

    {
        "job_type": "ue_tracks_generation",
        "simulation_id": "simulation_1",
        "ue_tracks_generation": {
            "output_file_prefix": "",
            "params": {
                "simulation_duration": 3600,
                "simulation_time_interval": 0.01,
                "num_ticks": 100,
                "num_batches": 10,
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
                    "rng_seed": 42,
                    "lon_x_dims": 100,
                    "lon_y_dims": 100
                    "// TODO": "Account for supporting the user choosing the anchor_loc and cov_around_anchor.",
                    "// Current implementation": "the UE Tracks generator will not be using these values.",
                    "// anchor_loc": {},
                    "// cov_around_anchor": {}
                }
            }
        }
    }

    The UE Tracks Generation service will output the following:
    - write output file(s) with generated UE Tracks for each batch
    - produce "output" event to the "outputs" topci to signal success/failure

    The UETracksGeneration class will do the following:
    1. Extract all the data from the job data which will be used to generate mobility
    2. Calculate the parameters required such as mobility class distribution, velocities and velocity variances
       from data extracted in the previous step
    3. Use the above parameters calculated to generate UE tracks
    4. Get each batch of mobility data in form of DataFrames and save them in files (.fea)
       with file name format {output_file_prefix}-{batch}.{ext}

    """

    def __init__(self):
        """Set up the orchestrator's producer instance"""
        self.producer = Producer(kafka_producer_config)

    def handle_ue_tracks_generation_job(self, job_data=None):
        """Handle an UE tracks generation job, start to finish"""

        logger.info(f"Handling UE Tracks generation job: {job_data}")

        # Extract all the required information from the job_data in order to generate UE tracks
        ue_tracks_generation_params = UETracksGenerationHelper.get_ue_tracks_generation_parameters(job_data)

        simulation_time_interval = UETracksGenerationHelper.get_simulation_time_interval(ue_tracks_generation_params)
        num_ticks = UETracksGenerationHelper.get_num_ticks(ue_tracks_generation_params)
        num_batches = UETracksGenerationHelper.get_num_batches(ue_tracks_generation_params)

        # Get the total number of UEs from the UE class distribution and add them up
        (
            stationary_count,
            pedestrian_count,
            cyclist_count,
            car_count,
        ) = UETracksGenerationHelper.get_ue_class_distribution_count(ue_tracks_generation_params)

        num_UEs = stationary_count + pedestrian_count + cyclist_count + car_count

        # Calculate the mobility class distribution as provided
        stationary_distribution = stationary_count / num_UEs
        pedestrian_distribution = pedestrian_count / num_UEs
        cyclist_distribution = cyclist_count / num_UEs
        car_distribution = car_count / num_UEs

        mobility_class_distribution = {
            MobilityClass.stationary: stationary_distribution,
            MobilityClass.pedestrian: pedestrian_distribution,
            MobilityClass.cyclist: cyclist_distribution,
            MobilityClass.car: car_distribution,
        }

        # Calculate the velocity class for each UE class
        # Each velocity class will be calculated according to the simulation_time_interval provided by the user,
        # which indicates the unit of time in seconds.
        # Each grid here defined in the mobility model is assumed to be 1 meter
        # Hence the velocity will have a unit of m/s (meter/second)
        (
            stationary_velocity,
            pedestrian_velocity,
            cyclist_velocity,
            car_velocity,
        ) = UETracksGenerationHelper.get_ue_class_distribution_velocity(
            ue_tracks_generation_params, simulation_time_interval
        )

        mobility_class_velocities = {
            MobilityClass.stationary: stationary_velocity,
            MobilityClass.pedestrian: pedestrian_velocity,
            MobilityClass.cyclist: cyclist_velocity,
            MobilityClass.car: car_velocity,
        }

        # Calculate the velocity variance for each UE class
        # Each velocity variance will be calculated according to the simulation_time_interval provided by the user,
        # which indicates the unit of time in seconds.
        # Each grid here defined in the mobility model is assumed to be 1 meter
        # Hence the velocity will have a unit of m/s (meter/second)
        (
            stationary_velocity_variance,
            pedestrian_velocity_variance,
            cyclist_velocity_variance,
            car_velocity_variances,
        ) = UETracksGenerationHelper.get_ue_class_distribution_velocity_variances(
            ue_tracks_generation_params, simulation_time_interval
        )

        mobility_class_velocity_variances = {
            MobilityClass.stationary: stationary_velocity_variance,
            MobilityClass.pedestrian: pedestrian_velocity_variance,
            MobilityClass.cyclist: cyclist_velocity_variance,
            MobilityClass.car: car_velocity_variances,
        }

        # latitude-longitude boundaries in which to generate UE tracks
        (
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        ) = UETracksGenerationHelper.get_lat_lon_boundaries(ue_tracks_generation_params)

        # Gauss Markov params
        alpha = UETracksGenerationHelper.get_gauss_markov_alpha(ue_tracks_generation_params)
        variance = UETracksGenerationHelper.get_gauss_markov_variance(ue_tracks_generation_params)
        rng_seed = UETracksGenerationHelper.get_gauss_markov_rng_seed(ue_tracks_generation_params)

        lon_x_dims, lon_y_dims = UETracksGenerationHelper.get_gauss_markov_xy_dims(ue_tracks_generation_params)

        # Use the above parameters extracted from the job data to generate mobility
        # Get each batch of mobility data in form of DataFrames

        simulation_id = UETracksGenerationHelper.get_simulation_id(job_data)

        current_batch = 1
        for ue_tracks_generation_current_batch_df in UETracksGenerator.generate_as_lon_lat_points(
            rng_seed=rng_seed,
            lon_x_dims=lon_x_dims,
            lon_y_dims=lon_y_dims,
            num_ticks=num_ticks,
            num_batches=num_batches,
            num_UEs=num_UEs,
            alpha=alpha,
            variance=variance,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            mobility_class_distribution=mobility_class_distribution,
            mobility_class_velocities=mobility_class_velocities,
            mobility_class_velocity_variances=mobility_class_velocity_variances,
        ):
            # save output to file with format {output_file_prefix}-{batch}.fea
            output_file_prefix = UETracksGenerationHelper.get_output_file_prefix(job_data)
            output_file_name = f"{output_file_prefix}-{current_batch}.{constants.DF_FILE_EXTENSION}"
            output_file_path = os.path.join(constants.UE_TRACK_GENERATION_OUTPUTS_FOLDER, output_file_name)

            write_feather_df(output_file_path, ue_tracks_generation_current_batch_df)
            logger.info(f"Saved UE Tracks batch {current_batch} output DF to {output_file_path}")

            # Once each batch has been processed and written to the output file, we can indicate that the job
            # has done successfully and produce output event to outputs topic

            output_event = {
                constants.SIMULATION_ID: simulation_id,
                constants.SERVICE: constants.UE_TRACKS_GENERATION,
                constants.BATCH: current_batch,
                constants.STATUS: OutputStatus.SUCCESS.value,
            }
            produce_object_to_kafka_topic(
                self.producer,
                topic=constants.KAFKA_OUTPUTS_TOPIC_NAME,
                value=output_event,
            )
            logger.info(f"Produced successful output event to topic: {output_event}")

            # Increment the batch number for the next batch
            current_batch += 1
