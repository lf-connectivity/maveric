import unittest
from digital_twin.mobility.ue_tracks_generation_helper import UETracksGenerationHelper
from common import constants

# Mock constants to simulate the actual constants used in the UETracksGenerationHelper class.
class constants:
    SIMULATION_ID = "simulation_id"
    UE_TRACKS_GENERATION = "ue_tracks_generation"
    PARAMS = "params"
    OUTPUT_FILE_PREFIX = "output_file_prefix"
    SIMULATION_TIME_INTERVAL = "simulation_time_interval_seconds"
    NUM_TICKS = "num_ticks"
    NUM_BATCHES = "num_batches"
    UE_CLASS_DISTRIBUTION = "ue_class_distribution"
    STATIONARY = "stationary"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    CAR = "car"
    COUNT = "count"
    VELOCITY = "velocity"
    VELOCITY_VARIANCE = "velocity_variance"
    LON_LAT_BOUNDARIES = "lat_lon_boundaries"
    MIN_LAT = "min_lat"
    MAX_LAT = "max_lat"
    MIN_LON = "min_lon"
    MAX_LON = "max_lon"
    GAUSS_MARKOV_PARAMS = "gauss_markov_params"
    ALPHA = "alpha"
    VARIANCE = "variance"
    RNG_SEED = "rng_seed"
    LON_X_DIMS = "lon_x_dims"
    LON_Y_DIMS = "lon_y_dims"


class TestUETracksGenerationHelper(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.job_data = {
            constants.SIMULATION_ID: "1234",
            constants.UE_TRACKS_GENERATION: {
                constants.PARAMS: {
                    constants.SIMULATION_TIME_INTERVAL: 5,
                    constants.NUM_TICKS: 100,
                    constants.NUM_BATCHES: 10,
                    constants.UE_CLASS_DISTRIBUTION: {
                        constants.STATIONARY: {constants.COUNT: 10, constants.VELOCITY: 0.0, constants.VELOCITY_VARIANCE: 0.0},
                        constants.PEDESTRIAN: {constants.COUNT: 20, constants.VELOCITY: 1.2, constants.VELOCITY_VARIANCE: 0.1},
                        constants.CYCLIST: {constants.COUNT: 15, constants.VELOCITY: 5.5, constants.VELOCITY_VARIANCE: 0.5},
                        constants.CAR: {constants.COUNT: 5, constants.VELOCITY: 20.0, constants.VELOCITY_VARIANCE: 1.0},
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
                constants.OUTPUT_FILE_PREFIX: "sim_output_"
            }
        }

    def test_get_simulation_id(self):
        self.assertEqual(UETracksGenerationHelper.get_simulation_id(self.job_data), "1234")

    def test_get_ue_tracks_generation_parameters(self):
        params = UETracksGenerationHelper.get_ue_tracks_generation_parameters(self.job_data)
        self.assertEqual(params[constants.NUM_TICKS], 100)
        self.assertEqual(params[constants.NUM_BATCHES], 10)

    def test_get_output_file_prefix(self):
        self.assertEqual(UETracksGenerationHelper.get_output_file_prefix(self.job_data), "sim_output_")

    def test_get_ue_class_distribution_count(self):
        counts = UETracksGenerationHelper.get_ue_class_distribution_count(self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS])
        self.assertEqual(counts, (10, 20, 15, 5))

    def test_get_ue_class_distribution_velocity(self):
        velocities = UETracksGenerationHelper.get_ue_class_distribution_velocity(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS], 5
        )
        self.assertEqual(velocities, (0.0, 6.0, 27.5, 100.0))

    def test_get_lat_lon_boundaries(self):
        boundaries = UETracksGenerationHelper.get_lat_lon_boundaries(
            self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS]
        )
        self.assertEqual(boundaries, (-90.0, 90.0, -180.0, 180.0))

    def test_get_gauss_markov_alpha(self):
        alpha = UETracksGenerationHelper.get_gauss_markov_alpha(self.job_data[constants.UE_TRACKS_GENERATION][constants.PARAMS])
        self.assertEqual(alpha, 0.8)

'''
if __name__ == '__main__':
    unittest.main()
'''