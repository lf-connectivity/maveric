# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

from radp.common import constants


class UETracksGenerationHelper:
    """Helper class for UE Tracks Generation Service
    Contains simple methods to read data from Kafka job event and
    other helpful utilities
    """

    @staticmethod
    def get_simulation_id(job_data: Dict) -> str:
        """Pull the simulation id from an RF Prediction job event"""
        return job_data[constants.SIMULATION_ID]

    @staticmethod
    def get_ue_tracks_generation_parameters(job_data: Dict) -> Dict:
        """Helper method to return model parameters of an UE Tracks Generation job"""
        return job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS]

    @staticmethod
    def get_output_file_prefix(job_data: Dict) -> str:
        """Helper method to return output file prefix for the UE Tracks Generation job"""
        return job_data[constants.UE_TRACKS_GENERATION][constants.OUTPUT_FILE_PREFIX]

    @staticmethod
    def get_simulation_time_interval(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return simulation_time_interval of an UE Tracks Generation job"""
        return ue_tracks_generation_params[constants.SIMULATION_TIME_INTERVAL]

    @staticmethod
    def get_num_ticks(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return num_ticks of an UE Tracks Generation job"""
        return ue_tracks_generation_params[constants.NUM_TICKS]

    @staticmethod
    def get_num_batches(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return num_batches of an UE Tracks Generation job"""
        return ue_tracks_generation_params[constants.NUM_BATCHES]

    @staticmethod
    def get_ue_class_distribution_count(ue_tracks_generation_params: Dict):
        """Helper method to return ue_class_distribution of an UE Tracks Generation job"""
        stationary_count = ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
            constants.STATIONARY
        ][constants.COUNT]
        pedestrian_count = ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
            constants.PEDESTRIAN
        ][constants.COUNT]
        cyclist_count = ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
            constants.CYCLIST
        ][constants.COUNT]
        car_count = ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
            constants.CAR
        ][constants.COUNT]
        return stationary_count, pedestrian_count, cyclist_count, car_count

    @staticmethod
    def get_ue_class_distribution_velocity(
        ue_tracks_generation_params: Dict, simulation_time_interval: int
    ):
        """
        Helper method to return ue_class_distribution_velocity of an UE Tracks Generation job

        Calculate the velocity class for each UE class
        Each velocity class will be calculated according to the simulation_time_interval provided by the user,
        which indicates the unit of time in seconds.
        Each grid here defined in the mobility model is assumed to be 1 meter
        Hence the velocity will have a unit of m/s (meter/second)
        """

        stationary_velocity = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.STATIONARY
            ][constants.VELOCITY]
            * simulation_time_interval
        )
        pedestrian_velocity = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.PEDESTRIAN
            ][constants.VELOCITY]
            * simulation_time_interval
        )
        cyclist_velocity = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.CYCLIST
            ][constants.VELOCITY]
            * simulation_time_interval
        )
        car_velocity = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][constants.CAR][
                constants.VELOCITY
            ]
            * simulation_time_interval
        )
        return (
            stationary_velocity,
            pedestrian_velocity,
            cyclist_velocity,
            car_velocity,
        )

    @staticmethod
    def get_ue_class_distribution_velocity_variances(
        ue_tracks_generation_params: Dict, simulation_time_interval: int
    ):
        """
        Helper method to return ue_class_distribution_velocity_variances of an UE Tracks Generation job

        Calculate the velocity class for each UE class
        Each velocity variance will be calculated according to the simulation_time_interval provided by the user,
        which indicates the unit of time in seconds.
        Each grid here defined in the mobility model is assumed to be 1 meter
        Hence the velocity will have a unit of m/s (meter/second)
        """

        stationary_velocity_variance = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.STATIONARY
            ][constants.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        pedestrian_velocity_variance = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.PEDESTRIAN
            ][constants.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        cyclist_velocity_variance = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][
                constants.CYCLIST
            ][constants.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        car_velocity_variances = (
            ue_tracks_generation_params[constants.UE_CLASS_DISTRIBUTION][constants.CAR][
                constants.VELOCITY_VARIANCE
            ]
            * simulation_time_interval
        )

        return (
            stationary_velocity_variance,
            pedestrian_velocity_variance,
            cyclist_velocity_variance,
            car_velocity_variances,
        )

    @staticmethod
    def get_lat_lon_boundaries(ue_tracks_generation_params: Dict):
        """
        Helper method to return latitude-longitude boundaries in which to generate UE tracks
        """

        min_lat = ue_tracks_generation_params[constants.LON_LAT_BOUNDARIES][
            constants.MIN_LAT
        ]
        max_lat = ue_tracks_generation_params[constants.LON_LAT_BOUNDARIES][
            constants.MAX_LAT
        ]
        min_lon = ue_tracks_generation_params[constants.LON_LAT_BOUNDARIES][
            constants.MIN_LON
        ]
        max_lon = ue_tracks_generation_params[constants.LON_LAT_BOUNDARIES][
            constants.MAX_LON
        ]

        return min_lat, max_lat, min_lon, max_lon

    @staticmethod
    def get_gauss_markov_alpha(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return gauss_markov alpha of an UE Tracks Generation job"""
        alpha = ue_tracks_generation_params[constants.GAUSS_MARKOV_PARAMS][
            constants.ALPHA
        ]
        return alpha

    @staticmethod
    def get_gauss_markov_variance(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return gauss_markov variance of an UE Tracks Generation job"""
        variance = ue_tracks_generation_params[constants.GAUSS_MARKOV_PARAMS][
            constants.VARIANCE
        ]
        return variance

    @staticmethod
    def get_gauss_markov_rng_seed(ue_tracks_generation_params: Dict) -> str:
        """Helper method to return gauss_markov rng_seed of an UE Tracks Generation job"""
        rng_seed = ue_tracks_generation_params[constants.GAUSS_MARKOV_PARAMS][
            constants.RNG_SEED
        ]
        return rng_seed

    @staticmethod
    def get_gauss_markov_xy_dims(ue_tracks_generation_params: Dict) -> Tuple[str, str]:
        """Helper method to return gauss_markov lon_x_dims and lon_y_dims of a UE Tracks Generation job"""
        lon_x_dims = ue_tracks_generation_params[constants.GAUSS_MARKOV_PARAMS][
            constants.LON_X_DIMS
        ]
        lon_y_dims = ue_tracks_generation_params[constants.GAUSS_MARKOV_PARAMS][
            constants.LON_Y_DIMS
        ]
        return lon_x_dims, lon_y_dims
