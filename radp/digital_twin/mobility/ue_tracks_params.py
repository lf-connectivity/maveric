from typing import Dict

from radp.common import constants
from radp.digital_twin.mobility.ue_tracks import MobilityClass
from services.ue_tracks_generation.ue_tracks_generation_helper import (
    UETracksGenerationHelper,
)


class UETracksGenerationParams:

    """
    The UETracksGenerationParams Class handles execution of the UE Tracks Generation parameters
    and generates the mobility distribution for the User Equipment (UE) instances.

    The UETracksGenerationParams will take in as input an UE Tracks Generation params
    with the following format:


    "ue_tracks_generation": {
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

    Attributes:
        rng_seed (int): Seed for the random number generator.
        num_batches (int): Number of batches to generate.
        lon_x_dims (int): Longitudinal dimension for x-coordinates.
        lon_y_dims (int): Longitudinal dimension for y-coordinates.
        num_ticks (int): Number of ticks per batch.
        num_UEs (int): Number of User Equipment (UE) instances.
        alpha (float): Alpha parameter for the Gauss-Markov mobility model.
        variance (float): Variance parameter for the Gauss-Markov mobility model.
        min_lat (float): Minimum latitude boundary.
        max_lat (float): Maximum latitude boundary.
        min_lon (float): Minimum longitude boundary.
        max_lon (float): Maximum longitude boundary.
        mobility_class_distribution (Dict[MobilityClass, float]): Distribution of mobility classes.
        mobility_class_velocities (Dict[MobilityClass, float]): Average velocities for each mobility class.
        mobility_class_velocity_variances (Dict[MobilityClass, float]): Variance of velocities for each mobility class.

    """

    def __init__(self, params: Dict):
        self.params = params[constants.UE_TRACKS_GENERATION][constants.PARAMS]
        self.rng_seed = self.params[constants.GAUSS_MARKOV_PARAMS][constants.RNG_SEED]
        self.num_batches = self.params[constants.NUM_BATCHES]
        self.lon_x_dims = self.params[constants.GAUSS_MARKOV_PARAMS][
            constants.LON_X_DIMS
        ]
        self.lon_y_dims = self.params[constants.GAUSS_MARKOV_PARAMS][
            constants.LON_Y_DIMS
        ]
        self.num_ticks = self.params[constants.NUM_TICKS]
        self.num_UEs = self.extract_ue_class_distribution()
        self.alpha = self.params[constants.GAUSS_MARKOV_PARAMS][constants.ALPHA]
        self.variance = self.params[constants.GAUSS_MARKOV_PARAMS][constants.VARIANCE]
        self.min_lat = self.params[constants.LON_LAT_BOUNDARIES][constants.MIN_LAT]
        self.max_lat = self.params[constants.LON_LAT_BOUNDARIES][constants.MAX_LAT]
        self.min_lon = self.params[constants.LON_LAT_BOUNDARIES][constants.MIN_LON]
        self.max_lon = self.params[constants.LON_LAT_BOUNDARIES][constants.MAX_LON]
        self.extract_ue_class_distribution()  # Initialize the method to extract the UE class distribution

    def extract_ue_class_distribution(self):
        """
        Processes and calculates UE class distribution, velocities, and variances from the parameters.
        """

        simulation_time_interval = self.params[constants.SIMULATION_TIME_INTERVAL]

        # Get the total number of UEs from the UE class distribution and add them up
        (
            stationary_count,
            pedestrian_count,
            cyclist_count,
            car_count,
        ) = UETracksGenerationHelper.get_ue_class_distribution_count(self.params)

        self.num_UEs = stationary_count + pedestrian_count + cyclist_count + car_count

        # Calculate the mobility class distribution as provided
        stationary_distribution = stationary_count / self.num_UEs
        pedestrian_distribution = pedestrian_count / self.num_UEs
        cyclist_distribution = cyclist_count / self.num_UEs
        car_distribution = car_count / self.num_UEs

        # Create the mobility class distribution dictionary
        self.mobility_class_distribution = {
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
            self.params, simulation_time_interval
        )

        self.mobility_class_velocities = {
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
            self.params, simulation_time_interval
        )

        self.mobility_class_velocity_variances = {
            MobilityClass.stationary: stationary_velocity_variance,
            MobilityClass.pedestrian: pedestrian_velocity_variance,
            MobilityClass.cyclist: cyclist_velocity_variance,
            MobilityClass.car: car_velocity_variances,
        }
