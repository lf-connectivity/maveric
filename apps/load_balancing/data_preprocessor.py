# This module is intended for the preprocessing of raw UE data CSV files
# to ensure compatibility with the Gym environment.
# The longitude and latitude columns are renamed to loc_x and loc_y, respectively, by the implemented class and methods.

import glob
import logging
import os
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UEDataPreprocessor:
    """
    The preprocessing of raw UE data CSV files is performed by this class
    to ensure compatibility with the Gym environment.
    The longitude and latitude columns are renamed to loc_x and loc_y, respectively.
    """

    def __init__(self, base_data_dir: str):
        self.base_data_dir = base_data_dir
        self.col_lon_input = "lon"
        self.col_lat_input = "lat"
        self.col_loc_x_output = "loc_x"
        self.col_loc_y_output = "loc_y"

    def run(self, days: List[int]):
        """
        The UE data is processed for a specified list of day numbers by this method.
        """
        for day_num in days:
            input_dir = os.path.join(self.base_data_dir, f"Day_{day_num}", "ue_data_per_tick")
            output_dir = os.path.join(self.base_data_dir, f"Day_{day_num}", "ue_data_gym_ready")

            os.makedirs(output_dir, exist_ok=True)

            input_file_pattern = os.path.join(input_dir, "generated_ue_data_for_cco_*.csv")
            ue_csv_files = glob.glob(input_file_pattern)

            if not ue_csv_files:
                logger.warning(f"No UE data CSV files found in '{input_dir}' for Day_{day_num}.")
                continue

            logger.info(f"Processing {len(ue_csv_files)} files for Day_{day_num}...")
            processed_count = 0
            for filepath in ue_csv_files:
                try:
                    filename = os.path.basename(filepath)
                    df = pd.read_csv(filepath)

                    if self.col_lon_input not in df.columns or self.col_lat_input not in df.columns:
                        logger.warning(
                            f"Skipping {filename}: Missing '{self.col_lon_input}' or '{self.col_lat_input}' column."
                        )
                        continue

                    df.rename(
                        columns={self.col_lon_input: self.col_loc_x_output, self.col_lat_input: self.col_loc_y_output},
                        inplace=True,
                    )

                    output_filepath = os.path.join(output_dir, filename)
                    df.to_csv(output_filepath, index=False)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {filepath}: {e}")
            logger.info(
                f"Finished preprocessing for Day_{day_num}. Processed {processed_count} files into '{output_dir}'."
            )
