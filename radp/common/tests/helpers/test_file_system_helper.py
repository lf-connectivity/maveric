# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from unittest import TestCase
from unittest.mock import mock_open, patch

from radp.common.enums import ModelStatus, ModelType, SimulationStage
from radp.common.helpers.file_system_helper import RADPFileSystemHelper

mocked_sim_data = {"test_key": "test_val"}
json_mocked_sim_data = json.dumps(mocked_sim_data)


class TestRADPFileSystemHelper(TestCase):
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    def test_gen_simulation_directory(self):
        simulation_directory = RADPFileSystemHelper.gen_simulation_directory(simulation_id="dummy_simulation")
        self.assertEqual(simulation_directory, "/dummy_simulation_data_folder_path/dummy_simulation")

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_METADATA_FILE_NAME",
        "dummy_metadata_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_METADATA_FILE_EXTENSION",
        "dummy_metadata_file_extension",
    )
    def test_gen_simulation_metadata_file_path(self):
        sim_metadata_file_path = RADPFileSystemHelper.gen_simulation_metadata_file_path(
            simulation_id="dummy_simulation"
        )
        self.assertEqual(
            sim_metadata_file_path,
            "/dummy_simulation_data_folder_path/dummy_simulation/"
            "dummy_metadata_file_name.dummy_metadata_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.UE_DATA_FILE_NAME",
        "dummy_ue_data_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.DF_FILE_EXTENSION",
        "dummy_df_file_extension",
    )
    def test_gen_simulation_ue_data_file_path(self):
        simulation_ue_data_file_path = RADPFileSystemHelper.gen_simulation_ue_data_file_path(
            simulation_id="dummy_simulation"
        )
        self.assertEqual(
            simulation_ue_data_file_path,
            "/dummy_simulation_data_folder_path/dummy_simulation/dummy_ue_data_file_name.dummy_df_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.CONFIG_FILE_NAME",
        "dummy_config_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.DF_FILE_EXTENSION",
        "dummy_df_file_extension",
    )
    def test_gen_simulation_cell_config_file_path(self):
        simulation_cell_config_file_path = RADPFileSystemHelper.gen_simulation_cell_config_file_path(
            simulation_id="dummy_simulation"
        )
        self.assertEqual(
            simulation_cell_config_file_path,
            "/dummy_simulation_data_folder_path/dummy_simulation/dummy_config_file_name.dummy_df_file_extension",
        )

    # replace builtins.open with a mocked open operation
    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_simulation_metadata_file_path")
    def test_load_simulation_metadata(self, mocked_metadata_file_path):
        mocked_metadata_file_path.return_value = "dummy_sim_data_file_path"
        self.assertEqual(
            RADPFileSystemHelper.gen_simulation_metadata_file_path("dummy_simulation"),
            "dummy_sim_data_file_path",
        )
        self.assertEqual(
            RADPFileSystemHelper.load_simulation_metadata("dummy_simulation"),
            mocked_sim_data,
        )

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_simulation_metadata_file_path")
    def test_load_simulation_metadata_exception(self, mocked_metadata_file_path):
        mocked_metadata_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.load_simulation_metadata("dummy_simulation")
        self.assertEqual(str(e.exception), "dummy exception!")

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.atomic_write")
    @patch("radp.common.helpers.file_system_helper.json")
    def test_save_simulation_metadata(self, mocked_json, mocked_atomic_write):
        RADPFileSystemHelper.save_simulation_metadata(mocked_sim_data, "dummy_simulation")
        mocked_atomic_write.assert_called_once()
        mocked_json.dump.assert_called_once()

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_simulation_metadata_file_path")
    def test_save_simulation_metadata_exception(self, mocked_metadata_file_path):
        mocked_metadata_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.save_simulation_metadata(mocked_sim_data, "dummy_simulation")
        self.assertEqual(str(e.exception), "dummy exception!")

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.pd.read_csv")
    @patch("radp.common.helpers.file_system_helper.write_feather_df")
    def test_save_simulation_ue_data(self, mocked_pandas_read_csv, mocked_write_feather_df):
        RADPFileSystemHelper.save_simulation_ue_data("dummy_simulation", "dummy_config_file_path")
        mocked_pandas_read_csv.assert_called_once()
        mocked_write_feather_df.assert_called_once()

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_simulation_ue_data_file_path")
    @patch("radp.common.helpers.file_system_helper.write_feather_df")
    def test_save_simulation_ue_data_exception(self, mocked_simulation_ue_data_file_path, mocked_write_feather_df):
        mocked_simulation_ue_data_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.save_simulation_ue_data("dummy_simulation", "dummy_config_file_path")
        mocked_write_feather_df.assert_called_once()
        self.assertEqual(str(e.exception), "dummy exception!")

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch("radp.utility.pandas_utils.atomic_write")
    @patch("radp.common.helpers.file_system_helper.pd.read_csv")
    def test_save_simulation_cell_config(self, mocked_pandas_read_csv, mocked_atomic_write):
        RADPFileSystemHelper.save_simulation_cell_config("dummy_simulation", "dummy_config_file_path")
        mocked_atomic_write.assert_called_once()
        mocked_pandas_read_csv.assert_called_once()

    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    @patch(
        """radp.common.helpers.file_system_helper.\
RADPFileSystemHelper.gen_simulation_cell_config_file_path"""
    )
    def test_save_simulation_cell_config_exception(self, mocked_simulation_cell_config_file_path):
        mocked_simulation_cell_config_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.save_simulation_cell_config("dummy_simulation", "dummy_config_file_path")
        self.assertEqual(str(e.exception), "dummy exception!")

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_OUTPUTS_FOLDER",
        "/dummy_simulation_outputs_folder",
    )
    @patch(
        "radp.common.enums.SimulationStage",
        "dummy_stage",
    )
    @patch("os.listdir")
    def test_hash_val_found_in_output_folder(self, mocked_listdir):
        mocked_listdir.return_value = ["dummy_dir_1", "dummy_dir_2"]
        dummy_stage = SimulationStage.UE_TRACKS_GENERATION
        self.assertTrue(RADPFileSystemHelper.hash_val_found_in_output_folder(dummy_stage, "dummy_dir"))

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_OUTPUTS_FOLDER",
        "/dummy_simulation_outputs_folder",
    )
    @patch(
        "radp.common.enums.SimulationStage",
        "dummy_stage",
    )
    @patch("os.listdir")
    def test_hash_val_found_in_output_folder_neg(self, mocked_listdir):
        mocked_listdir.return_value = ["dummy_dir_1", "dummy_dir_2"]
        dummy_stage = SimulationStage.UE_TRACKS_GENERATION
        self.assertFalse(RADPFileSystemHelper.hash_val_found_in_output_folder(dummy_stage, "dummy_other_str"))

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_OUTPUTS_FOLDER",
        "dummy_simulation_outputs_folder",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.DF_FILE_EXTENSION",
        "dummy_df_file_extension",
    )
    def test_gen_stage_output_file_path(self):
        dummy_stage = SimulationStage.UE_TRACKS_GENERATION
        dummy_hash_val = "dummy_hash_val"
        dummy_batch = 1
        self.assertEqual(
            RADPFileSystemHelper.gen_stage_output_file_path(dummy_stage, dummy_hash_val, dummy_batch),
            "/dummy_simulation_data_folder_path/dummy_simulation_outputs_folder/"
            "ue_tracks_generation/ue_tracks_generation-dummy_hash_val-1.dummy_df_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_OUTPUT_FILE_SUFFIX",
        "dummy_output_file_suffix",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_OUTPUT_FILE_EXTENSION",
        "dummy_output_file_extension",
    )
    def test_gen_sim_output_zip_file_path(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_sim_output_zip_file_path("dummy_simulation", True),
            "/dummy_simulation_data_folder_path/dummy_simulation/"
            "dummy_simulation-dummy_output_file_suffix.dummy_output_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_OUTPUT_FILE_SUFFIX",
        "dummy_output_file_suffix",
    )
    def test_gen_sim_output_zip_file_path_neg(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_sim_output_zip_file_path("dummy_simulation", False),
            "/dummy_simulation_data_folder_path/dummy_simulation/dummy_simulation-dummy_output_file_suffix",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIM_OUTPUT_DIRECTORY",
        "dummy_output_directory",
    )
    def test_gen_sim_output_directory(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_sim_output_directory("dummy_simulation"),
            "/dummy_simulation_data_folder_path/dummy_simulation/dummy_output_directory",
        )

    # TODO: Skipped zip_output_files_to_simulation_folder_as_csvs()

    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_DATA_FOLDER",
        "/dummy_simulation_data_folder_path",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.SIMULATION_OUTPUTS_FOLDER",
        "dummy_simulation_outputs_folder",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.DF_FILE_EXTENSION",
        "dummy_df_file_extension",
    )
    def test_get_stage_output_file_paths(self):
        dummy_stage = SimulationStage.START
        dummy_hash_val = "dummy_hash_val"
        dummy_num_batches = 2
        self.assertEqual(
            RADPFileSystemHelper.get_stage_output_file_paths(
                dummy_stage,
                dummy_hash_val,
                dummy_num_batches,
            ),
            [
                "/dummy_simulation_data_folder_path/dummy_simulation_outputs_folder/"
                "start/start-dummy_hash_val-1.dummy_df_file_extension",
                "/dummy_simulation_data_folder_path/dummy_simulation_outputs_folder/"
                "start/start-dummy_hash_val-2.dummy_df_file_extension",
            ],
        )

    @patch("os.listdir")
    @patch("os.remove")
    def test_clear_output_data_from_stage_no_save_hash_val(self, mocked_listdir, mocked_remove):
        mocked_listdir.return_value = ["dummy_file_1"]
        dummy_stage = SimulationStage.START
        RADPFileSystemHelper.clear_output_data_from_stage(dummy_stage, None)
        mocked_remove.assert_called_once()

    @patch("os.listdir")
    @patch("os.remove")
    def test_clear_output_data_from_stage_with_hash_val(self, mocked_listdir, mocked_remove):
        mocked_listdir.return_value = ["dummy_file_1", "dummy_file_2"]
        dummy_stage = SimulationStage.START
        RADPFileSystemHelper.clear_output_data_from_stage(dummy_stage, "_2")
        mocked_remove.assert_called_once()

    @patch("os.listdir")
    def test_clear_output_data_from_stage_exception(self, mocked_listdir):
        mocked_listdir.side_effect = Exception("dummy exception!")
        dummy_stage = SimulationStage.START
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.clear_output_data_from_stage(dummy_stage, None)
        self.assertEqual(str(e.exception), "dummy exception!")

    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_ID",
        "dummy_model_id",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_TYPE",
        "dummy_model_type",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.STATUS",
        "dummy_status",
    )
    def test_gen_model_metadata_frame(self):
        dummy_model_id = "dummy_model_id"
        dummy_model_type = ModelType.RF_DIGITAL_TWIN
        dummy_status = ModelStatus.TRAINED
        dummy_model_specific_params = {"dummy_key": "dummy_val"}
        self.assertEqual(
            RADPFileSystemHelper.gen_model_metadata_frame(
                dummy_model_id,
                dummy_model_type,
                dummy_status,
                dummy_model_specific_params,
            ),
            {
                "dummy_model_id": dummy_model_id,
                "dummy_model_type": dummy_model_type.value,
                "dummy_status": dummy_status.value,
                dummy_model_type.value: dummy_model_specific_params,
            },
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.MODELS_FOLDER",
        "/dummy_models_folder",
    )
    def test_gen_model_folder_path(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_model_folder_path("dummy_model_id"),
            "/dummy_models_folder/dummy_model_id",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.MODELS_FOLDER",
        "/dummy_models_folder",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_FILE_NAME",
        "dummy_model_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_FILE_EXTENSION",
        "dummy_model_file_extension",
    )
    def test_gen_model_file_path(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_model_file_path(
                "dummy_model_id",
            ),
            "/dummy_models_folder/dummy_model_id/dummy_model_file_name.dummy_model_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.MODELS_FOLDER",
        "/dummy_models_folder",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_METADATA_FILE_NAME",
        "dummy_model_metadata_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_METADATA_FILE_EXTENSION",
        "dummy_model_metadata_file_extension",
    )
    def test_gen_model_metadata_file_path(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_model_metadata_file_path(
                "dummy_model_id",
            ),
            "/dummy_models_folder/dummy_model_id/dummy_model_metadata_file_name.dummy_model_metadata_file_extension",
        )

    @patch(
        "radp.common.helpers.file_system_helper.constants.MODELS_FOLDER",
        "/dummy_models_folder",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.TOPOLOGY_FILE_NAME",
        "dummy_topology_file_name",
    )
    @patch(
        "radp.common.helpers.file_system_helper.constants.DF_FILE_EXTENSION",
        "dummy_df_file_extension",
    )
    def test_gen_model_topology_file_path(self):
        self.assertEqual(
            RADPFileSystemHelper.gen_model_topology_file_path(
                "dummy_model_id",
            ),
            "/dummy_models_folder/dummy_model_id/dummy_topology_file_name.dummy_df_file_extension",
        )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_metadata_file_path")
    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    def test_load_model_metadata(self, mocked_model_metadata_file_path):
        mocked_model_metadata_file_path.return_value = "dummy_model_metadata_file_path"
        self.assertEqual(
            RADPFileSystemHelper.gen_model_metadata_file_path(
                "dummy_model_id",
            ),
            "dummy_model_metadata_file_path",
        )
        self.assertEqual(
            RADPFileSystemHelper.load_model_metadata(
                "dummy_model_id",
            ),
            mocked_sim_data,
        )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_metadata_file_path")
    @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    def test_load_model_metadata_exception(self, mocked_model_metadata_file_path):
        mocked_model_metadata_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.load_model_metadata(
                "dummy_simulation",
            )
        self.assertEqual(
            str(e.exception),
            "dummy exception!",
        )

    # @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    # @patch("radp.common.helpers.file_system_helper.json.dump")
    # @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_folder_path")
    # @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_metadata_file_path")
    # def test_save_model_metadata(self, mocked_json, mocked_model_folder_path, mocked_model_metadata_file_path):
    #     mocked_model_folder_path.return_value = "dummy_folder_path"
    #     mocked_model_metadata_file_path.return_value = "dummy_file_path"
    #     RADPFileSystemHelper.save_model_metadata(
    #         "dummy_simulation",
    #         mocked_sim_data,
    #     )
    #     mocked_json.assert_called_once()

    # @patch("builtins.open", mock_open(read_data=json_mocked_sim_data))
    # @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_folder_path")
    # @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_metadata_file_path")
    # def test_save_model_metadata_exception(self, mocked_model_folder_path, mocked_model_metadata_file_path):
    #     mocked_model_folder_path.return_value = "dummy_folder_path"
    #     mocked_model_metadata_file_path.side_effect = Exception("dummy exception!")
    #     with self.assertRaises(Exception) as e:
    #         RADPFileSystemHelper.save_model_metadata(
    #             "dummy_simulation",
    #             mocked_sim_data,
    #         )
    #     self.assertEqual(
    #         str(e.exception),
    #         "dummy exception!",
    #     )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_file_path")
    def test_check_model_exists(self, mocked_model_file_path):
        model_file_path = RADPFileSystemHelper.gen_model_file_path(
            "dummy_model_id",
        )
        mocked_model_file_path.return_value = model_file_path
        self.assertTrue(
            RADPFileSystemHelper.check_model_exists(
                "dummy_model_id",
            )
        )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.gen_model_file_path")
    def test_check_model_exists_neg(self, mocked_model_file_path):
        mocked_model_file_path.side_effect = Exception("dummy exception!")
        with self.assertRaises(Exception) as e:
            RADPFileSystemHelper.check_model_exists(
                "dummy_model_id",
            )
        self.assertEqual(
            str(e.exception),
            "dummy exception!",
        )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.load_model_metadata")
    @patch(
        "radp.common.helpers.file_system_helper.constants.STATUS",
        "dummy_status",
    )
    def test_get_model_status(self, mocked_model_metadata):
        mocked_model_metadata.return_value = {"dummy_status": "TRAINED"}
        self.assertEqual(
            RADPFileSystemHelper.get_model_status(
                "dummy_model_id",
            ).value,
            "trained",
        )

    @patch("radp.common.helpers.file_system_helper.RADPFileSystemHelper.load_model_metadata")
    @patch(
        "radp.common.helpers.file_system_helper.constants.MODEL_TYPE",
        "dummy_model_type",
    )
    def test_get_model_type(self, mocked_model_metadata):
        mocked_model_metadata.return_value = {"dummy_model_type": "RF_DIGITAL_TWIN"}
        self.assertEqual(
            RADPFileSystemHelper.get_model_type(
                "dummy_model_id",
            ).value,
            "rf_digital_twin",
        )
