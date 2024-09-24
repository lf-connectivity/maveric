import unittest
from services.ue_tracks_generation.ue_tracks_generation_helper import (
    UETracksGenerationHelper,
)
from radp.common import constants


class TestUETracksGenerationHelper(unittest.TestCase):
    """
    Unit tests for the UETracksGenerationHelper class.

    This test suite validates key functionalities of the UETracksGenerationHelper
    class, which handles mobility data generation for user equipment (UE).
    The tests focus on verifying:

    - Correct retrieval of simulation parameters such as simulation ID, number
    of ticks, and batches.
    - Proper generation of output file prefixes.
    - Accurate calculation of UE class distributions (counts and velocities).
    - Correct geographic boundary values for latitude and longitude.
    - Accuracy of Gauss-Markov model parameters, specifically the alpha value.

    Mock data is used to simulate job configurations and expected outputs.

    How to Run:
    ------------
    To run these tests, execute the following command in your terminal:
    ```
    python3 -m unittest radp/digital_twin/mobility/tests/test_ue_tracks_generation_helper.py
    ```
    """

    def setUp(self):
        self.job_data = {
            constants.SIMULATION_ID: "1234",
            constants.UE_TRACKS_GENERATION: {
                constants.PARAMS: {
                    constants.SIMULATION_TIME_INTERVAL: 5,
                    constants.NUM_TICKS: 100,
                    constants.NUM_BATCHES: 10,
                    constants.UE_CLASS_DISTRIBUTION: {
                        constants.STATIONARY: {
                            constants.COUNT: 10,
                            constants.VELOCITY: 0.0,
                            constants.VELOCITY_VARIANCE: 0.0,
                        },
                        constants.PEDESTRIAN: {
                            constants.COUNT: 20,
                            constants.VELOCITY: 1.2,
                            constants.VELOCITY_VARIANCE: 0.1,
                        },
                        constants.CYCLIST: {
                            constants.COUNT: 15,
                            constants.VELOCITY: 5.5,
                            constants.VELOCITY_VARIANCE: 0.5,
                        },
                        constants.CAR: {
                            constants.COUNT: 5,
                            constants.VELOCITY: 20.0,
                            constants.VELOCITY_VARIANCE: 1.0,
                        },
                    },
                    constants.LON_LAT_BOUNDARIES: {
                        constants.MIN_LAT: -90.0,
                        constants.MAX_LAT: 90.0,
                        constants.MIN_LON: -180.0,
                        constants.MAX_LON: 180.0,
                    },
                    constants.GAUSS_MARKOV_PARAMS: {
                        constants.ALPHA: 0.8,
                        constants.VARIANCE: 0.1,
                        constants.RNG_SEED: 42,
                        constants.LON_X_DIMS: "100",
                        constants.LON_Y_DIMS: "100",
                    },
                },
                constants.OUTPUT_FILE_PREFIX: "sim_output_",
            },
        }

    def test_get_simulation_id(self):
        """
        - Validates retrieval of the simulation ID.
        """
        self.assertEqual(
            UETracksGenerationHelper.get_simulation_id(self.job_data), "1234"
        )

    def test_get_ue_tracks_generation_parameters(self):
        """
        - Ensures correct retrieval of ticks and batches.
        - Verifies consistency across batches and ticks.
        """
        params = UETracksGenerationHelper.get_ue_tracks_generation_parameters(
            self.job_data
        )
        self.assertEqual(params[constants.NUM_TICKS], 100)
        self.assertEqual(params[constants.NUM_BATCHES], 10)

    def test_get_output_file_prefix(self):
        """
        - Validates retrieval of the output file prefix.
        """
        self.assertEqual(
            UETracksGenerationHelper.get_output_file_prefix(self.job_data),
            "sim_output_",
        )

    def test_get_ue_class_distribution_count(self):
        """
        - Validates that the class distribution counts are correctly retrieved.
        """
        counts = UETracksGenerationHelper.get_ue_class_distribution_count(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS]
        )
        self.assertEqual(counts, (10, 20, 15, 5))

    def test_get_ue_class_distribution_velocity(self):
        """
        - Validates that the class distribution velocities are correctly calculated.
        """
        velocities = UETracksGenerationHelper.get_ue_class_distribution_velocity(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS], 5
        )
        self.assertEqual(velocities, (0.0, 6.0, 27.5, 100.0))

    def test_get_lat_lon_boundaries(self):
        """
        - Validates that the latitude and longitude boundaries are correctly retrieved.
        """
        boundaries = UETracksGenerationHelper.get_lat_lon_boundaries(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS]
        )
        self.assertEqual(boundaries, (-90.0, 90.0, -180.0, 180.0))

    def test_get_gauss_markov_alpha(self):
        """
        - Validates that the Gauss-Markov alpha value is correctly retrieved.
        """
        alpha = UETracksGenerationHelper.get_gauss_markov_alpha(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS]
        )
        self.assertEqual(alpha, 0.8)
