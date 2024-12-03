import unittest
import pandas as pd
import numpy as np
from apps.mobility_robustness_optimization.mobility_robustness_optimization import (
    calculate_mro_metric,
    reattach_columns,
    count_handovers,
    MobilityRobustnessOptimization as MRO,
)


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def setUp(self):
        # Define the data with 5 ticks and 2 distinct UE ids
        self.data = {
            'loc_x': [-22.625309, 119.764151, 72.095437, -67.548009, 59.867089,  # tick 0
                    -22.625309, 119.764151, 72.095437, -67.548009, 59.867089,  # tick 1
                    -22.625309, 119.764151, 72.095437, -67.548009, 59.867089,  # tick 2
                    -22.625309, 119.764151, 72.095437, -67.548009, 59.867089,  # tick 3
                    -22.625309, 119.764151, 72.095437, -67.548009, 59.867089], # tick 4
            'loc_y': [59.806764, 54.857584, -20.253892, -38.100941, -83.103930,  # tick 0
                    59.806764, 54.857584, -20.253892, -38.100941, -83.103930,  # tick 1
                    59.806764, 54.857584, -20.253892, -38.100941, -83.103930,  # tick 2
                    59.806764, 54.857584, -20.253892, -38.100941, -83.103930,  # tick 3
                    59.806764, 54.857584, -20.253892, -38.100941, -83.103930], # tick 4
            'cell_id': [3.0, 3.0, 1.0, 1.0, 1.0,  # tick 0
                        3.0, 3.0, 1.0, 1.0, 1.0,  # tick 1
                        3.0, 3.0, 1.0, 1.0, 1.0,  # tick 2
                        3.0, 3.0, 1.0, 1.0, 1.0,  # tick 3
                        3.0, 3.0, 1.0, 1.0, 1.0],  # tick 4
            'sinr_db': [-2.379967, -2.327379, -2.879403, -2.681959, -1.272086,  # tick 0
                        -2.379967, -2.327379, -2.879403, -2.681959, -1.272086,  # tick 1
                        -2.379967, -2.327379, -2.879403, -2.681959, -1.272086,  # tick 2
                        -2.379967, -2.327379, -2.879403, -2.681959, -1.272086,  # tick 3
                        -2.379967, -2.327379, -2.879403, -2.681959, -1.272086],  # tick 4
            'rsrp_dbm': [-99.439212, -99.529860, -99.914970, -99.750036, -98.454310,  # tick 0
                        -99.439212, -99.529860, -99.914970, -99.750036, -98.454310,  # tick 1
                        -99.439212, -99.529860, -99.914970, -99.750036, -98.454310,  # tick 2
                        -99.439212, -99.529860, -99.914970, -99.750036, -98.454310,  # tick 3
                        -99.439212, -99.529860, -99.914970, -99.750036, -98.454310],  # tick 4
            'ue_id': [0, 1, 0, 1, 0,  # tick 0
                    0, 1, 0, 1, 0,  # tick 1
                    0, 1, 0, 1, 0,  # tick 2
                    0, 1, 0, 1, 0,  # tick 3
                    0, 1, 0, 1, 0],  # tick 4
            'tick': [0, 0, 0, 0, 0,  # tick 0
                    1, 1, 1, 1, 1,  # tick 1
                    2, 2, 2, 2, 2,  # tick 2
                    3, 3, 3, 3, 3,  # tick 3
                    4, 4, 4, 4, 4],  # tick 4
        }
        # Create the DataFrame
        self.df = pd.DataFrame(self.data)

        self.dummy_topology = {
            "cell_id": ["cell_001", "cell_002"],
            "cell_lat": [45.0, 46.0],
            "cell_lon": [-73.0, -74.0]
        }

        self.dummy_topology = pd.DataFrame(self.dummy_topology)

    def test_update(self):        # TODO: Implement AFTER PR
        pass
    def test_solve(self):        # TODO: Implement AFTER PR
        pass

    def test_format_ue_data_and_topology(self):
        formatted_ue, formatted_topology = MRO._format_ue_data_and_topology(self.data, self.dummy_topology)
        expected_topology = pd.DataFrame({
            "cell_id": [1, 2],
            "loc_y": [45.0, 46.0],
            "loc_x": [-73.0, -74.0]
        })
        pd.testing.assert_frame_equal(formatted_topology, expected_topology)
        # Verify ue_data column renaming
        self.assertIn("loc_x", formatted_ue.columns)
        self.assertIn("loc_y", formatted_ue.columns)
        self.assertNotIn("lat", formatted_ue.columns)
        self.assertNotIn("lon", formatted_ue.columns)

    def test_training(self):
        # TODO: Implement
        pass

    def test_predictions(self):
        # TODO: Implement
        pass

    def test_prepare_all_UEs_from_all_cells_df(self):
        # TODO: Implement
        pass

    def _calculate_received_power(self):
        dummy_distance = 1
        dummy_freq = 1800
        expected_fspl = 20 * np.log10(dummy_distance) + 20 * np.log10(dummy_freq) - 27.55
        expected_power = 23 - expected_fspl
        power = MRO._calculate_received_power(dummy_distance,dummy_freq)
        self.assertEqual(expected_power,power)

    def test_preprocess_ue_topology_data(self):
        # TODO: Implement
        pass

    def test_preprocess_ue_simulation_data(self):
        # TODO: Implement
        pass

    def test_preprocess_ue_training_data(self):
        # TODO: Implement
        pass

    def test_preprocess_ue_update_data(self):
        # TODO: Implement
        pass

    def test_preprocess_prediction_data(self):
        # TODO: Implement
        pass

    def test_count_handovers(self):
        ns_handover_count, nf_handover_count, no_change = count_handovers(self.df)
        expected_ns = 18
        expected_nf = 0
        expected_no_change = 7
        self.assertEqual(ns_handover_count, expected_ns)
        self.assertEqual(nf_handover_count, expected_nf)
        self.assertEqual(no_change, expected_no_change)

    def test_reattach_columns(self):
        self.predicted_df = pd.DataFrame({
            'tick': [1, 2, 3, 4],
            'loc_x': [10, 20, 30, 40],
            'loc_y': [5, 6, 7, 8]
        })
        self.full_prediction_df = pd.DataFrame({
            'mock_ue_id': [100, 101, 102, 103],
            'tick': [1, 2, 3, 4],
            'loc_x': [10, 20, 30, 40],
            'loc_y': [5, 6, 7, 8]
        })
        full_prediction_df_no_match = pd.DataFrame({
            'mock_ue_id': [200, 201, 202, 203],
            'tick': [1, 2, 3, 4],
            'loc_x': [50, 60, 70, 80],  # No matching loc_x values
            'loc_y': [9, 10, 11, 12]    # No matching loc_y values
        })

        result = reattach_columns(self.predicted_df, self.full_prediction_df)

        self.assertTrue('ue_id' in result.columns)  # Ensure 'ue_id' column exists
        self.assertFalse('mock_ue_id' in result.columns)  # Ensure 'mock_ue_id' column is removed
        self.assertEqual(result.shape[0], self.predicted_df.shape[0]) # Ensure size matches predicted_df
        self.assertEqual(result.loc[0, 'ue_id'], 100)
        self.assertEqual(result.loc[1, 'ue_id'], 101)

        result = reattach_columns(self.predicted_df, full_prediction_df_no_match)
        self.assertTrue(result['ue_id'].isna().all())

    def test_calculate_mro_metric(self):
        data = {
            "tick": [1, 2, 3, 4, 5] * 10,  # 5 unique ticks, each repeated 10 times
            "loc_x": range(50),  # Dummy x coordinates (you can modify as needed)
            "loc_y": range(50),  # Dummy y coordinates (you can modify as needed)
            "ue_id": range(50),  # Dummy UE IDs
            "sinr_db": [-3.0, -1.5, 1.0, -2.0, 0.5] * 10,  # Dummy SINR values
        }
        dummy_pred_data = pd.DataFrame(data)
        ns_count = 1
        nf_count = 2

        d = calculate_mro_metric(ns_count, nf_count, dummy_pred_data)
        real_d = 5 - (ns_count * (50 / 1000) + nf_count * (1000 / 1000))
        self.assertEqual(d, real_d)