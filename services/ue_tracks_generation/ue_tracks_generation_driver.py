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
        for ue_tracks_generation_current_batch_df in self._mobility_data_generation(
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

    def _mobility_data_generation(
        self,
        rng_seed: int,
        lon_x_dims: int,
        lon_y_dims: int,
        num_ticks: int,
        num_batches: int,
        num_UEs: int,
        alpha: int,
        variance: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
    ) -> pd.DataFrame:
        """
        The mobility data generation method takes in all the parameters required to generate UE tracks
        for a specified number of batches

        The UETracksGenerator uses the Gauss-Markov Mobility Model to yields batch of tracks for UEs,
        corresponding to `num_ticks` number of simulation ticks, and the number of UEs
        the user wants to simulate.

        Using the UETracksGenerator, the UE tracks are returned in form of a dataframe
        The Dataframe is arranged as follows:

        +------------+------------+-----------+------+
        | mock_ue_id | lon        | lat       | tick |
        +============+============+===========+======+
        |   0        | 102.219377 | 33.674572 |   0  |
        |   1        | 102.415954 | 33.855534 |   0  |
        |   2        | 102.545935 | 33.878075 |   0  |
        |   0        | 102.297766 | 33.575942 |   1  |
        |   1        | 102.362725 | 33.916477 |   1  |
        |   2        | 102.080675 | 33.832793 |   1  |
        +------------+------------+-----------+------+
        """

        ue_tracks_generator = UETracksGenerator(
            rng=np.random.default_rng(rng_seed),
            lon_x_dims=lon_x_dims,
            lon_y_dims=lon_y_dims,
            num_ticks=num_ticks,
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
        )

        for _num_batches, xy_batches in enumerate(ue_tracks_generator.generate()):
            ue_tracks_dataframe_dict: Dict[Any, Any] = {}

            # Extract the xy (lon, lat) points from each batch to use it in the mobility dataframe
            # mock_ue_id, tick, lat, lon
            mock_ue_id = []
            ticks = []
            lon: List[float] = []
            lat: List[float] = []

            tick = 0
            for xy_batch in xy_batches:
                lon_lat_pairs = GISTools.converting_xy_points_into_lonlat_pairs(
                    xy_points=xy_batch,
                    x_dim=lon_x_dims,
                    y_dim=lon_y_dims,
                    min_longitude=min_lon,
                    max_longitude=max_lon,
                    min_latitude=min_lat,
                    max_latitude=max_lat,
                )

                # Build list for each column/row for the UE Tracks dataframe
                lon.extend(xy_points[0] for xy_points in lon_lat_pairs)
                lat.extend(xy_points[1] for xy_points in lon_lat_pairs)
                mock_ue_id.extend([i for i in range(num_UEs)])
                ticks.extend(list(itertools.repeat(tick, num_UEs)))
                tick += 1

            # Build dict for each column/row for the UE Tracks dataframe
            ue_tracks_dataframe_dict[constants.MOCK_UE_ID] = mock_ue_id
            ue_tracks_dataframe_dict[constants.LONGITUDE] = lon
            ue_tracks_dataframe_dict[constants.LATITUDE] = lat
            ue_tracks_dataframe_dict[constants.TICK] = ticks

            # Yield each batch as a dataframe
            yield pd.DataFrame(ue_tracks_dataframe_dict)

            num_batches -= 1
            if num_batches == 0:
                break
