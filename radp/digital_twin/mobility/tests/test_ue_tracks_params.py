import unittest
from radp.digital_twin.mobility.ue_tracks_params import UETracksGenerationParams
from services.ue_tracks_generation.ue_tracks_generation_helper import (
    UETracksGenerationHelper,
)
from radp.common import constants
from radp.digital_twin.mobility.ue_tracks import MobilityClass
from unittest.mock import patch


class TestUETracksParams(unittest.TestCase):
    """
    Unit tests for the UETracksGenerationParams class.

    Tests the initialization and attribute extraction of UETracksGenerationParams
    from valid parameter configurations, ensuring correct handling of mobility class
    distributions and velocities.

    How to Run:
    ------------
    To run these tests, execute the following command in your terminal:
    ```
    python3 -m unittest radp/digital_twin/mobility/tests/test_ue_tracks_params.py
    ```
    """

    def setUp(self) -> None:
        self.valid_params = {
            constants.UE_TRACKS_GENERATION: {
                constants.PARAMS: {
                    constants.SIMULATION_DURATION: 3600,
                    constants.SIMULATION_TIME_INTERVAL: 0.01,
                    constants.NUM_TICKS: 100,
                    constants.NUM_BATCHES: 10,
                    constants.UE_CLASS_DISTRIBUTION: {
                        constants.STATIONARY: {
                            constants.COUNT: 10,
                            constants.VELOCITY: 1,
                            constants.VELOCITY_VARIANCE: 1,
                        },
                        constants.PEDESTRIAN: {
                            constants.COUNT: 10,
                            constants.VELOCITY: 1,
                            constants.VELOCITY_VARIANCE: 1,
                        },
                        constants.CYCLIST: {
                            constants.COUNT: 10,
                            constants.VELOCITY: 1,
                            constants.VELOCITY_VARIANCE: 1,
                        },
                        constants.CAR: {
                            constants.COUNT: 10,
                            constants.VELOCITY: 1,
                            constants.VELOCITY_VARIANCE: 1,
                        },
                    },
                    constants.LON_LAT_BOUNDARIES: {
                        constants.MIN_LAT: -90,
                        constants.MAX_LAT: 90,
                        constants.MIN_LON: -180,
                        constants.MAX_LON: 180,
                    },
                    constants.GAUSS_MARKOV_PARAMS: {
                        constants.ALPHA: 0.5,
                        constants.VARIANCE: 0.8,
                        constants.RNG_SEED: 42,
                        constants.LON_X_DIMS: 100,
                        constants.LON_Y_DIMS: 100,
                    },
                }
            }
        }

    @patch.object(UETracksGenerationHelper, "get_ue_class_distribution_count")
    @patch.object(UETracksGenerationHelper, "get_ue_class_distribution_velocity")
    @patch.object(
        UETracksGenerationHelper, "get_ue_class_distribution_velocity_variances"
    )
    def test_initialization_and_extraction(
        self, mock_velocity_variances, mock_velocity, mock_count
    ) -> None:
        """
        Test the initialization and attribute extraction of UETracksGenerationParams.

        Args:
            mock_velocity_variances: Mock for velocity variances method.
            mock_velocity: Mock for velocity method.
            mock_count: Mock for count method.

        Returns:
            None
        """
        # Set up the mock return values
        mock_count.return_value = (10, 10, 10, 10)
        mock_velocity.return_value = (1, 1, 1, 1)
        mock_velocity_variances.return_value = (1, 1, 1, 1)

        # Initialize the UETracksGenerationParams object
        params = UETracksGenerationParams(self.valid_params)

        # Assert attributes are set correctly
        self.assertEqual(params.rng_seed, 42)
        self.assertEqual(params.num_batches, 10)
        self.assertEqual(params.lon_x_dims, 100)
        self.assertEqual(params.lon_y_dims, 100)
        self.assertEqual(params.num_ticks, 100)
        self.assertEqual(params.num_UEs, 40)  # 10 + 10 + 10 + 10
        self.assertEqual(params.alpha, 0.5)
        self.assertEqual(params.variance, 0.8)
        self.assertEqual(params.min_lat, -90)
        self.assertEqual(params.max_lat, 90)
        self.assertEqual(params.min_lon, -180)
        self.assertEqual(params.max_lon, 180)

        # Assert mobility class distributions using the MobilityClass Enum
        self.assertAlmostEqual(
            params.mobility_class_distribution[MobilityClass.stationary], 10 / 40
        )
        self.assertAlmostEqual(
            params.mobility_class_distribution[MobilityClass.pedestrian], 10 / 40
        )
        self.assertAlmostEqual(
            params.mobility_class_distribution[MobilityClass.cyclist], 10 / 40
        )
        self.assertAlmostEqual(
            params.mobility_class_distribution[MobilityClass.car], 10 / 40
        )

        # Assert mobility class velocities
        self.assertEqual(params.mobility_class_velocities[MobilityClass.stationary], 1)
        self.assertEqual(params.mobility_class_velocities[MobilityClass.pedestrian], 1)
        self.assertEqual(params.mobility_class_velocities[MobilityClass.cyclist], 1)
        self.assertEqual(params.mobility_class_velocities[MobilityClass.car], 1)

        # Assert mobility class velocity variances
        self.assertEqual(
            params.mobility_class_velocity_variances[MobilityClass.stationary], 1
        )
        self.assertEqual(
            params.mobility_class_velocity_variances[MobilityClass.pedestrian], 1
        )
        self.assertEqual(
            params.mobility_class_velocity_variances[MobilityClass.cyclist], 1
        )
        self.assertEqual(params.mobility_class_velocity_variances[MobilityClass.car], 1)