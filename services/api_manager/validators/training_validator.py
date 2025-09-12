# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Any
import pandas as pd

from api_manager.validators.base_validator import BaseValidator, SchemaValidator, FileValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException
from radp.common import constants


class TrainingRequestValidator(BaseValidator):
    """Validator for training API requests"""
    
    # Define training request schema
    TRAINING_SCHEMA = {
        "model_id": {
            "required": True,
            "type": str,
            "min_length": 1,
            "max_length": 100,
            "pattern": r"^[a-zA-Z0-9_-]+$"  # alphanumeric, underscore, hyphen only
        },
        "model_update": {
            "required": False,
            "type": bool
        },
        "params": {
            "required": True,
            "type": dict
        }
    }
    
    # Define training params schema
    TRAINING_PARAMS_SCHEMA = {
        "maxiter": {
            "required": False,
            "type": int,
            "min": 1,
            "max": 10000
        },
        "lr": {
            "required": False,
            "type": float,
            "min": 0.0001,
            "max": 1.0
        },
        "stopping_threshold": {
            "required": False,
            "type": float,
            "min": 1e-8,
            "max": 1e-1
        }
    }
    
    def __init__(self):
        self.schema_validator = SchemaValidator(self.TRAINING_SCHEMA)
        self.params_validator = SchemaValidator(self.TRAINING_PARAMS_SCHEMA)
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate training request data"""
        # Validate main request structure
        self.schema_validator.validate(data)
        
        # Validate training parameters
        params = data.get("params", {})
        self.params_validator.validate(params)
        
        # Custom validation for model_id uniqueness if not updating
        self._validate_model_id_availability(data.get("model_id"), data.get("model_update", False))
    
    def _validate_model_id_availability(self, model_id: str, is_update: bool) -> None:
        """Check if model_id is available for new models or exists for updates"""
        from radp.common.helpers.file_system_helper import RADPFileSystemHelper
        
        model_exists = RADPFileSystemHelper.check_model_exists(model_id)
        
        if not is_update and model_exists:
            raise ValidationException(
                f"Model '{model_id}' already exists. Use model_update=true to update existing model.",
                field="model_id"
            )
        elif is_update and not model_exists:
            raise ValidationException(
                f"Model '{model_id}' does not exist. Cannot update non-existent model.",
                field="model_id"
            )
    
    def validate_training_files(self, files: Dict[str, str]) -> None:
        """Validate training data files"""
        required_file_keys = [constants.UE_TRAINING_DATA_FILE_PATH_KEY, constants.TOPOLOGY_FILE_PATH_KEY]
        
        # Check required files are provided
        for file_key in required_file_keys:
            if file_key not in files:
                raise ValidationException(f"Missing required file: {file_key}")
            
            file_path = files[file_key]
            if not os.path.exists(file_path):
                raise FileValidationException(f"File not found: {file_path}")
        
        # Validate UE training data file
        self._validate_ue_training_data(files[constants.UE_TRAINING_DATA_FILE_PATH_KEY])
        
        # Validate topology file
        self._validate_topology_file(files[constants.TOPOLOGY_FILE_PATH_KEY])
        
        # Cross-validate that cells in training data exist in topology
        self._validate_cell_consistency(
            files[constants.UE_TRAINING_DATA_FILE_PATH_KEY],
            files[constants.TOPOLOGY_FILE_PATH_KEY]
        )
    
    def _validate_ue_training_data(self, file_path: str) -> pd.DataFrame:
        """Validate UE training data CSV file"""
        required_columns = ["cell_id", "avg_rsrp", "lon", "lat", "cell_el_deg"]
        
        # Validate file format and columns
        df = FileValidator.validate_csv_file(file_path, required_columns, "ue_training_data.csv")
        
        # Validate file size (max 500MB for training data)
        FileValidator.validate_file_size(file_path, max_size_mb=500, filename="ue_training_data.csv")
        
        # Validate data types and ranges
        errors = []
        
        # Check RSRP values are in reasonable range (-150 to 0 dBm)
        invalid_rsrp = df[(df["avg_rsrp"] < -150) | (df["avg_rsrp"] > 0)]
        if not invalid_rsrp.empty:
            errors.append({
                "field": "avg_rsrp",
                "error": f"RSRP values must be between -150 and 0 dBm. Found {len(invalid_rsrp)} invalid values."
            })
        
        # Check latitude values are valid (-90 to 90)
        invalid_lat = df[(df["lat"] < -90) | (df["lat"] > 90)]
        if not invalid_lat.empty:
            errors.append({
                "field": "lat", 
                "error": f"Latitude values must be between -90 and 90. Found {len(invalid_lat)} invalid values."
            })
        
        # Check longitude values are valid (-180 to 180)
        invalid_lon = df[(df["lon"] < -180) | (df["lon"] > 180)]
        if not invalid_lon.empty:
            errors.append({
                "field": "lon",
                "error": f"Longitude values must be between -180 and 180. Found {len(invalid_lon)} invalid values."
            })
        
        # Check electrical tilt is reasonable (0 to 15 degrees)
        invalid_tilt = df[(df["cell_el_deg"] < 0) | (df["cell_el_deg"] > 15)]
        if not invalid_tilt.empty:
            errors.append({
                "field": "cell_el_deg",
                "error": f"Electrical tilt must be between 0 and 15 degrees. Found {len(invalid_tilt)} invalid values."
            })
        
        if errors:
            raise FileValidationException(
                "UE training data validation failed",
                filename="ue_training_data.csv",
                validation_errors=errors
            )
        
        return df
    
    def _validate_topology_file(self, file_path: str) -> pd.DataFrame:
        """Validate topology CSV file"""
        required_columns = ["cell_lat", "cell_lon", "cell_id", "cell_az_deg", "cell_carrier_freq_mhz"]
        
        # Validate file format and columns  
        df = FileValidator.validate_csv_file(file_path, required_columns, "topology.csv")
        
        # Validate file size (max 10MB for topology)
        FileValidator.validate_file_size(file_path, max_size_mb=10, filename="topology.csv")
        
        # Validate data ranges
        errors = []
        
        # Check latitude values
        invalid_lat = df[(df["cell_lat"] < -90) | (df["cell_lat"] > 90)]
        if not invalid_lat.empty:
            errors.append({
                "field": "cell_lat",
                "error": f"Cell latitude must be between -90 and 90. Found {len(invalid_lat)} invalid values."
            })
        
        # Check longitude values
        invalid_lon = df[(df["cell_lon"] < -180) | (df["cell_lon"] > 180)]
        if not invalid_lon.empty:
            errors.append({
                "field": "cell_lon", 
                "error": f"Cell longitude must be between -180 and 180. Found {len(invalid_lon)} invalid values."
            })
        
        # Check azimuth values (0 to 360 degrees)
        invalid_az = df[(df["cell_az_deg"] < 0) | (df["cell_az_deg"] >= 360)]
        if not invalid_az.empty:
            errors.append({
                "field": "cell_az_deg",
                "error": f"Cell azimuth must be between 0 and 359 degrees. Found {len(invalid_az)} invalid values."
            })
        
        # Check frequency values (reasonable cellular frequencies)
        invalid_freq = df[(df["cell_carrier_freq_mhz"] < 400) | (df["cell_carrier_freq_mhz"] > 6000)]
        if not invalid_freq.empty:
            errors.append({
                "field": "cell_carrier_freq_mhz",
                "error": f"Carrier frequency must be between 400 and 6000 MHz. Found {len(invalid_freq)} invalid values."
            })
        
        # Check for duplicate cell IDs
        duplicate_cells = df[df.duplicated(subset=["cell_id"])]
        if not duplicate_cells.empty:
            errors.append({
                "field": "cell_id",
                "error": f"Found duplicate cell IDs: {duplicate_cells['cell_id'].tolist()}"
            })
        
        if errors:
            raise FileValidationException(
                "Topology file validation failed",
                filename="topology.csv",
                validation_errors=errors
            )
        
        return df
    
    def _validate_cell_consistency(self, training_file_path: str, topology_file_path: str) -> None:
        """Validate that cells in training data exist in topology"""
        try:
            training_df = pd.read_csv(training_file_path)
            topology_df = pd.read_csv(topology_file_path)
            
            training_cells = set(training_df["cell_id"].unique())
            topology_cells = set(topology_df["cell_id"].unique())
            
            missing_cells = training_cells - topology_cells
            if missing_cells:
                raise ValidationException(
                    f"Training data contains cells not found in topology: {', '.join(missing_cells)}"
                )
        except Exception as e:
            if isinstance(e, ValidationException):
                raise
            raise FileValidationException(f"Error validating cell consistency: {str(e)}")