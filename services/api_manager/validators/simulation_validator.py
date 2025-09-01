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


class SimulationRequestValidator(BaseValidator):
    """Validator for simulation API requests"""
    
    # Define simulation request schema
    SIMULATION_SCHEMA = {
        "simulation_time_interval_seconds": {
            "required": True,
            "type": float,
            "min": 0.001,  # 1ms minimum
            "max": 60.0    # 1 minute maximum
        },
        "ue_tracks": {
            "required": True,
            "type": dict
        },
        "rf_prediction": {
            "required": True,
            "type": dict
        },
        "protocol_emulation": {
            "required": False,
            "type": dict
        }
    }
    
    # Schema for UE tracks generation
    UE_TRACKS_GENERATION_SCHEMA = {
        "ue_class_distribution": {
            "required": True,
            "type": dict
        },
        "lat_lon_boundaries": {
            "required": True,
            "type": dict
        },
        "gauss_markov_params": {
            "required": True,
            "type": dict
        }
    }
    
    # Schema for UE class parameters
    UE_CLASS_PARAMS_SCHEMA = {
        "count": {
            "required": True,
            "type": int,
            "min": 0,
            "max": 10000
        },
        "velocity": {
            "required": True,
            "type": (int, float),
            "min": 0.0,
            "max": 200.0  # Max 200 m/s
        },
        "velocity_variance": {
            "required": True,
            "type": (int, float),
            "min": 0.0,
            "max": 50.0
        }
    }
    
    # Schema for lat/lon boundaries
    LAT_LON_BOUNDARIES_SCHEMA = {
        "min_lat": {
            "required": True,
            "type": (int, float),
            "min": -90.0,
            "max": 90.0
        },
        "max_lat": {
            "required": True,
            "type": (int, float),
            "min": -90.0,
            "max": 90.0
        },
        "min_lon": {
            "required": True,
            "type": (int, float),
            "min": -180.0,
            "max": 180.0
        },
        "max_lon": {
            "required": True,
            "type": (int, float),
            "min": -180.0,
            "max": 180.0
        }
    }
    
    # Schema for Gauss-Markov parameters
    GAUSS_MARKOV_SCHEMA = {
        "alpha": {
            "required": True,
            "type": (int, float),
            "min": 0.0,
            "max": 1.0
        },
        "variance": {
            "required": True,
            "type": (int, float),
            "min": 0.0,
            "max": 100.0
        },
        "rng_seed": {
            "required": False,
            "type": int,
            "min": 0,
            "max": 2**31 - 1
        }
    }
    
    # Schema for RF prediction
    RF_PREDICTION_SCHEMA = {
        "model_id": {
            "required": True,
            "type": str,
            "min_length": 1,
            "max_length": 100,
            "pattern": r"^[a-zA-Z0-9_-]+$"
        }
    }
    
    def __init__(self):
        self.schema_validator = SchemaValidator(self.SIMULATION_SCHEMA)
        self.ue_tracks_gen_validator = SchemaValidator(self.UE_TRACKS_GENERATION_SCHEMA)
        self.ue_class_validator = SchemaValidator(self.UE_CLASS_PARAMS_SCHEMA)
        self.boundaries_validator = SchemaValidator(self.LAT_LON_BOUNDARIES_SCHEMA)
        self.gauss_markov_validator = SchemaValidator(self.GAUSS_MARKOV_SCHEMA)
        self.rf_prediction_validator = SchemaValidator(self.RF_PREDICTION_SCHEMA)
        
        # Valid UE classes
        self.valid_ue_classes = {constants.STATIONARY, constants.PEDESTRIAN, constants.CYCLIST, constants.CAR}
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate simulation request data"""
        # Validate main request structure
        self.schema_validator.validate(data)
        
        # Validate UE tracks component
        ue_tracks = data.get("ue_tracks", {})
        self._validate_ue_tracks(ue_tracks)
        
        # Validate RF prediction component
        rf_prediction = data.get("rf_prediction", {})
        self._validate_rf_prediction(rf_prediction)
        
        # Validate protocol emulation if present
        protocol_emulation = data.get("protocol_emulation")
        if protocol_emulation:
            self._validate_protocol_emulation(protocol_emulation)
        
        # Cross-validation
        self._validate_simulation_constraints(data)
    
    def _validate_ue_tracks(self, ue_tracks: Dict[str, Any]) -> None:
        """Validate UE tracks configuration"""
        has_generation = constants.UE_TRACKS_GENERATION in ue_tracks
        has_data_id = "ue_data_id" in ue_tracks
        
        if not has_generation and not has_data_id:
            raise ValidationException(
                "UE tracks must contain either 'ue_tracks_generation' or 'ue_data_id'",
                field="ue_tracks"
            )
        
        if has_generation and has_data_id:
            raise ValidationException(
                "UE tracks cannot contain both 'ue_tracks_generation' and 'ue_data_id'",
                field="ue_tracks"
            )
        
        if has_generation:
            self._validate_ue_tracks_generation(ue_tracks[constants.UE_TRACKS_GENERATION])
        
        if has_data_id:
            self._validate_ue_data_id(ue_tracks["ue_data_id"])
    
    def _validate_ue_tracks_generation(self, generation_config: Dict[str, Any]) -> None:
        """Validate UE tracks generation configuration"""
        self.ue_tracks_gen_validator.validate(generation_config)
        
        # Validate UE class distribution
        ue_classes = generation_config.get("ue_class_distribution", {})
        self._validate_ue_class_distribution(ue_classes)
        
        # Validate lat/lon boundaries
        boundaries = generation_config.get("lat_lon_boundaries", {})
        self._validate_lat_lon_boundaries(boundaries)
        
        # Validate Gauss-Markov parameters
        gauss_markov = generation_config.get("gauss_markov_params", {})
        self.gauss_markov_validator.validate(gauss_markov)
    
    def _validate_ue_class_distribution(self, ue_classes: Dict[str, Any]) -> None:
        """Validate UE class distribution"""
        if not ue_classes:
            raise ValidationException("At least one UE class must be specified", field="ue_class_distribution")
        
        # Check for invalid UE class names
        invalid_classes = set(ue_classes.keys()) - self.valid_ue_classes
        if invalid_classes:
            raise ValidationException(
                f"Invalid UE classes: {', '.join(invalid_classes)}. Valid classes: {', '.join(self.valid_ue_classes)}",
                field="ue_class_distribution"
            )
        
        # Validate each UE class parameters
        total_ues = 0
        for class_name, class_params in ue_classes.items():
            try:
                self.ue_class_validator.validate(class_params)
                total_ues += class_params.get("count", 0)
            except ValidationException as e:
                # Add class context to error
                e.field = f"ue_class_distribution.{class_name}"
                raise e
        
        # Check total UE count is reasonable
        if total_ues == 0:
            raise ValidationException("Total UE count cannot be zero", field="ue_class_distribution")
        elif total_ues > 50000:  # Reasonable limit
            raise ValidationException(
                f"Total UE count ({total_ues}) exceeds maximum limit (50,000)",
                field="ue_class_distribution"
            )
    
    def _validate_lat_lon_boundaries(self, boundaries: Dict[str, Any]) -> None:
        """Validate lat/lon boundaries"""
        self.boundaries_validator.validate(boundaries)
        
        # Logical validation
        min_lat = boundaries.get("min_lat")
        max_lat = boundaries.get("max_lat")
        min_lon = boundaries.get("min_lon")
        max_lon = boundaries.get("max_lon")
        
        if min_lat >= max_lat:
            raise ValidationException("min_lat must be less than max_lat", field="lat_lon_boundaries")
        
        if min_lon >= max_lon:
            raise ValidationException("min_lon must be less than max_lon", field="lat_lon_boundaries")
        
        # Check area is reasonable (not too small)
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        
        if lat_span < 0.001 or lon_span < 0.001:  # ~100m minimum
            raise ValidationException("Geographic area is too small (minimum ~100m)", field="lat_lon_boundaries")
    
    def _validate_ue_data_id(self, ue_data_id: str) -> None:
        """Validate UE data ID"""
        if not isinstance(ue_data_id, str) or not ue_data_id.strip():
            raise ValidationException("ue_data_id must be a non-empty string", field="ue_data_id")
        
        if len(ue_data_id) > 100:
            raise ValidationException("ue_data_id must be 100 characters or less", field="ue_data_id")
    
    def _validate_rf_prediction(self, rf_prediction: Dict[str, Any]) -> None:
        """Validate RF prediction configuration"""
        self.rf_prediction_validator.validate(rf_prediction)
        
        # Check if model exists
        model_id = rf_prediction.get("model_id")
        if model_id:
            self._validate_model_exists(model_id)
    
    def _validate_model_exists(self, model_id: str) -> None:
        """Check if RF prediction model exists"""
        from radp.common.helpers.file_system_helper import RADPFileSystemHelper
        
        if not RADPFileSystemHelper.check_model_exists(model_id):
            raise ValidationException(
                f"RF prediction model '{model_id}' does not exist. Train the model first.",
                field="rf_prediction.model_id"
            )
    
    def _validate_protocol_emulation(self, protocol_emulation: Dict[str, Any]) -> None:
        """Validate protocol emulation configuration"""
        # Basic validation - can be expanded based on specific requirements
        if not isinstance(protocol_emulation, dict):
            raise ValidationException("protocol_emulation must be an object", field="protocol_emulation")
        
        # Validate common protocol parameters if present
        if "ttt_seconds" in protocol_emulation:
            ttt = protocol_emulation["ttt_seconds"]
            if not isinstance(ttt, (int, float)) or ttt < 0 or ttt > 10:
                raise ValidationException(
                    "ttt_seconds must be between 0 and 10 seconds",
                    field="protocol_emulation.ttt_seconds"
                )
        
        if "hysteresis" in protocol_emulation:
            hysteresis = protocol_emulation["hysteresis"]
            if not isinstance(hysteresis, (int, float)) or hysteresis < 0 or hysteresis > 20:
                raise ValidationException(
                    "hysteresis must be between 0 and 20 dB",
                    field="protocol_emulation.hysteresis"
                )
    
    def _validate_simulation_constraints(self, data: Dict[str, Any]) -> None:
        """Validate overall simulation constraints and limits"""
        time_interval = data.get("simulation_time_interval_seconds", 0)
        
        # For UE tracks generation, estimate simulation size
        ue_tracks = data.get("ue_tracks", {})
        if constants.UE_TRACKS_GENERATION in ue_tracks:
            ue_classes = ue_tracks[constants.UE_TRACKS_GENERATION].get("ue_class_distribution", {})
            total_ues = sum(params.get("count", 0) for params in ue_classes.values())
            
            # Estimate memory and processing requirements
            # Simple heuristic: time_interval determines tick count indirectly
            if total_ues > 1000 and time_interval < 0.01:
                raise ValidationException(
                    "High UE count with very small time intervals may cause performance issues. "
                    "Consider reducing UE count or increasing time interval.",
                    field="simulation_constraints"
                )
    
    def validate_simulation_files(self, files: Dict[str, str]) -> None:
        """Validate simulation data files if provided"""
        # Validate UE data file if provided
        if constants.UE_DATA_FILE_PATH_KEY in files:
            self._validate_ue_data_file(files[constants.UE_DATA_FILE_PATH_KEY])
        
        # Validate config file if provided
        if constants.CONFIG_FILE_PATH_KEY in files:
            self._validate_config_file(files[constants.CONFIG_FILE_PATH_KEY])
    
    def _validate_ue_data_file(self, file_path: str) -> pd.DataFrame:
        """Validate UE data CSV file"""
        required_columns = ["mock_ue_id", "lon", "lat", "tick"]
        optional_columns = ["cell_id"]
        
        # Validate file format
        df = FileValidator.validate_csv_file(file_path, required_columns, "ue_data.csv")
        
        # Validate file size (max 1GB for UE data)
        FileValidator.validate_file_size(file_path, max_size_mb=1000, filename="ue_data.csv")
        
        # Validate data ranges and consistency
        errors = []
        
        # Check latitude/longitude ranges
        invalid_lat = df[(df["lat"] < -90) | (df["lat"] > 90)]
        if not invalid_lat.empty:
            errors.append({
                "field": "lat",
                "error": f"Latitude must be between -90 and 90. Found {len(invalid_lat)} invalid values."
            })
        
        invalid_lon = df[(df["lon"] < -180) | (df["lon"] > 180)]
        if not invalid_lon.empty:
            errors.append({
                "field": "lon", 
                "error": f"Longitude must be between -180 and 180. Found {len(invalid_lon)} invalid values."
            })
        
        # Check tick values are non-negative integers
        invalid_tick = df[df["tick"] < 0]
        if not invalid_tick.empty:
            errors.append({
                "field": "tick",
                "error": f"Tick values must be non-negative. Found {len(invalid_tick)} invalid values."
            })
        
        # Check UE ID consistency (should be consistent across ticks)
        ue_tick_counts = df.groupby("mock_ue_id")["tick"].nunique()
        max_ticks = ue_tick_counts.max()
        min_ticks = ue_tick_counts.min()
        
        if max_ticks != min_ticks:
            errors.append({
                "field": "mock_ue_id",
                "error": f"Inconsistent tick counts across UEs (min: {min_ticks}, max: {max_ticks})"
            })
        
        if errors:
            raise FileValidationException(
                "UE data file validation failed",
                filename="ue_data.csv",
                validation_errors=errors
            )
        
        return df
    
    def _validate_config_file(self, file_path: str) -> pd.DataFrame:
        """Validate cell configuration CSV file"""
        required_columns = ["cell_id", "cell_el_deg"]
        
        # Validate file format
        df = FileValidator.validate_csv_file(file_path, required_columns, "config.csv")
        
        # Validate file size (max 10MB for config)
        FileValidator.validate_file_size(file_path, max_size_mb=10, filename="config.csv")
        
        # Validate electrical tilt values
        invalid_tilt = df[(df["cell_el_deg"] < 0) | (df["cell_el_deg"] > 15)]
        if not invalid_tilt.empty:
            raise FileValidationException(
                f"Electrical tilt must be between 0 and 15 degrees. Found {len(invalid_tilt)} invalid values.",
                filename="config.csv"
            )
        
        # Check for duplicate cell IDs
        duplicate_cells = df[df.duplicated(subset=["cell_id"])]
        if not duplicate_cells.empty:
            raise FileValidationException(
                f"Found duplicate cell IDs in config: {duplicate_cells['cell_id'].tolist()}",
                filename="config.csv"
            )
        
        return df