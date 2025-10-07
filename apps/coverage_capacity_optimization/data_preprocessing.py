# data_preprocessing.py

import logging
import os
from typing import List

import pandas as pd

# Logging is configured for the module, and a logger instance is created.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UEDataPreprocessor:
    """
    The preprocessing of raw UE data CSV files is performed by this class to ensure compatibility
    with the CCO environment.
    The columns representing longitude and latitude are renamed to loc_x and loc_y, respectively.
    """

    def __init__(self, base_data_dir: str = None):
        self.base_data_dir = base_data_dir
        self.col_lon_input = "lon"
        self.col_lat_input = "lat"
        self.col_loc_x_output = "loc_x"
        self.col_loc_y_output = "loc_y"

    def load_and_process_ue_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process UE data from a single CSV file.
        
        Args:
            file_path: Path to the UE data CSV file
            
        Returns:
            Processed DataFrame with coordinates renamed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"UE data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded UE data from {file_path}: {len(df)} records")
            
            # Check if required columns exist
            if self.col_lon_input not in df.columns or self.col_lat_input not in df.columns:
                logger.warning(f"Missing required columns in {file_path}. Available columns: {list(df.columns)}")
                return df
            
            # Rename longitude and latitude columns to loc_x and loc_y
            df.rename(
                columns={self.col_lon_input: self.col_loc_x_output, self.col_lat_input: self.col_loc_y_output},
                inplace=True,
            )
            
            logger.info(f"Processed UE data: renamed {self.col_lon_input}->{self.col_loc_x_output}, {self.col_lat_input}->{self.col_loc_y_output}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def process_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Process multiple UE data files and combine them.
        
        Args:
            file_paths: List of paths to UE data CSV files
            
        Returns:
            Combined processed DataFrame
        """
        dataframes = []
        
        for file_path in file_paths:
            try:
                df = self.load_and_process_ue_data(file_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No data files were successfully processed")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} files: {len(combined_df)} total records")
        
        return combined_df

    def run(self, days: List[int] = None):
        """
        The processing of UE data for the specified list of day numbers is conducted by this method.
        For CCO data, this method is kept for compatibility but doesn't use days.
        
        Args:
            days: List of day numbers (not used in CCO context, kept for compatibility)
        """
        if self.base_data_dir is None:
            logger.warning("No base data directory specified. Use load_and_process_ue_data() or process_multiple_files() instead.")
            return
        
        logger.info("CCO data preprocessing - days parameter not applicable")
        logger.info("Use load_and_process_ue_data() or process_multiple_files() for CCO data processing")