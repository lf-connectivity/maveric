# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List

import pandas as pd

from api_manager.exceptions.validation_exception import (
    FileValidationException,
    ValidationException,
)
from api_manager.validators.base_validator import BaseValidator, SchemaValidator
from api_manager.validators.file_validator import FileValidator
from radp.common import constants


class SimulationRequestValidator(BaseValidator):
    """Validator for RIC simulation API requests."""

    MODEL_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
    VALID_UE_CLASSES = {
        constants.STATIONARY,
        constants.PEDESTRIAN,
        constants.CYCLIST,
        constants.CAR,
    }

    SIMULATION_SCHEMA = {
        constants.SIMULATION_TIME_INTERVAL: {
            "required": True,
            "type": (int, float),
            "min": 0.001,
            "max": 60.0,
        },
        constants.UE_TRACKS: {"required": True, "type": dict},
        constants.RF_PREDICTION: {"required": True, "type": dict},
        constants.PROTOCOL_EMULATION: {"required": False, "type": dict},
    }

    UE_TRACKS_GENERATION_SCHEMA = {
        constants.SIMULATION_DURATION: {
            "required": True,
            "type": (int, float),
            "min": 0.001,
            "max": 86400,
        },
        constants.UE_CLASS_DISTRIBUTION: {"required": True, "type": dict},
        constants.LON_LAT_BOUNDARIES: {"required": True, "type": dict},
        constants.GAUSS_MARKOV_PARAMS: {"required": True, "type": dict},
    }

    UE_CLASS_SCHEMA = {
        constants.COUNT: {"required": True, "type": int, "min": 0, "max": 10000},
        constants.VELOCITY: {
            "required": True,
            "type": (int, float),
            "min": 0,
            "max": 200,
        },
        constants.VELOCITY_VARIANCE: {
            "required": True,
            "type": (int, float),
            "min": 0,
            "max": 50,
        },
    }

    BOUNDARIES_SCHEMA = {
        constants.MIN_LAT: {
            "required": True,
            "type": (int, float),
            "min": -90,
            "max": 90,
        },
        constants.MAX_LAT: {
            "required": True,
            "type": (int, float),
            "min": -90,
            "max": 90,
        },
        constants.MIN_LON: {
            "required": True,
            "type": (int, float),
            "min": -180,
            "max": 180,
        },
        constants.MAX_LON: {
            "required": True,
            "type": (int, float),
            "min": -180,
            "max": 180,
        },
    }

    GAUSS_MARKOV_SCHEMA = {
        constants.ALPHA: {
            "required": True,
            "type": (int, float),
            "min": 0,
            "max": 1,
        },
        constants.VARIANCE: {
            "required": True,
            "type": (int, float),
            "min": 0,
            "max": 100,
        },
        constants.RNG_SEED: {
            "required": False,
            "type": int,
            "min": 0,
            "max": 2**31 - 1,
        },
    }

    RF_PREDICTION_SCHEMA = {
        constants.MODEL_ID: {
            "required": True,
            "type": str,
            "min_length": 1,
            "max_length": 100,
            "pattern": MODEL_ID_PATTERN,
        }
    }

    def __init__(self):
        self.schema_validator = SchemaValidator(self.SIMULATION_SCHEMA)
        self.ue_tracks_gen_validator = SchemaValidator(
            self.UE_TRACKS_GENERATION_SCHEMA,
            field_prefix=f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.",
        )
        self.ue_class_validator = SchemaValidator(self.UE_CLASS_SCHEMA, allow_unknown=False)
        self.boundaries_validator = SchemaValidator(
            self.BOUNDARIES_SCHEMA,
            allow_unknown=False,
            field_prefix=f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.{constants.LON_LAT_BOUNDARIES}.",
        )
        self.gauss_markov_validator = SchemaValidator(
            self.GAUSS_MARKOV_SCHEMA,
            field_prefix=f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.{constants.GAUSS_MARKOV_PARAMS}.",
        )
        self.rf_prediction_validator = SchemaValidator(
            self.RF_PREDICTION_SCHEMA,
            field_prefix=f"{constants.RF_PREDICTION}.",
        )

    def validate(self, data: Dict[str, Any]) -> None:
        self.schema_validator.validate(data)
        self._validate_ue_tracks(data.get(constants.UE_TRACKS, {}))
        self._validate_rf_prediction(data.get(constants.RF_PREDICTION, {}))
        if constants.PROTOCOL_EMULATION in data:
            self._validate_protocol_emulation(data[constants.PROTOCOL_EMULATION])
        self._validate_simulation_constraints(data)

    def validate_simulation_files(self, files: Dict[str, str]) -> None:
        if constants.UE_DATA_FILE_PATH_KEY in files:
            self._validate_ue_data_file(files[constants.UE_DATA_FILE_PATH_KEY])
        if constants.CONFIG_FILE_PATH_KEY in files:
            self._validate_config_file(files[constants.CONFIG_FILE_PATH_KEY])

    def _validate_ue_tracks(self, ue_tracks: Dict[str, Any]) -> None:
        has_generation = constants.UE_TRACKS_GENERATION in ue_tracks
        has_data_id = "ue_data_id" in ue_tracks

        if not has_generation and not has_data_id:
            raise ValidationException(
                "UE tracks must contain either 'ue_tracks_generation' or 'ue_data_id'",
                field=constants.UE_TRACKS,
            )
        if has_generation and has_data_id:
            raise ValidationException(
                "UE tracks cannot contain both 'ue_tracks_generation' and 'ue_data_id'",
                field=constants.UE_TRACKS,
            )
        if has_generation:
            self._validate_ue_tracks_generation(ue_tracks[constants.UE_TRACKS_GENERATION])
        if has_data_id:
            self._validate_ue_data_id(ue_tracks["ue_data_id"])

    def _validate_ue_tracks_generation(self, config: Dict[str, Any]) -> None:
        self.ue_tracks_gen_validator.validate(config)
        self._validate_ue_class_distribution(config[constants.UE_CLASS_DISTRIBUTION])
        self._validate_lat_lon_boundaries(config[constants.LON_LAT_BOUNDARIES])
        self.gauss_markov_validator.validate(config[constants.GAUSS_MARKOV_PARAMS])

    def _validate_ue_class_distribution(self, ue_classes: Dict[str, Any]) -> None:
        if not ue_classes:
            raise ValidationException(
                "At least one UE class must be specified",
                field=f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.{constants.UE_CLASS_DISTRIBUTION}",
            )

        invalid_classes = sorted(set(ue_classes.keys()) - self.VALID_UE_CLASSES)
        if invalid_classes:
            raise ValidationException(
                f"Invalid UE classes: {', '.join(invalid_classes)}",
                field=f"{constants.UE_CLASS_DISTRIBUTION}",
            )

        total_ues = 0
        errors: List[Dict[str, str]] = []
        for class_name, params in ue_classes.items():
            validator = SchemaValidator(
                self.UE_CLASS_SCHEMA,
                allow_unknown=False,
                field_prefix=f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.{constants.UE_CLASS_DISTRIBUTION}.{class_name}.",
            )
            try:
                validator.validate(params)
            except ValidationException as exc:
                errors.extend(exc.validation_errors)
            if isinstance(params, dict):
                total_ues += params.get(constants.COUNT, 0)

        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)
        if total_ues == 0:
            raise ValidationException(
                "Total UE count cannot be zero",
                field=constants.UE_CLASS_DISTRIBUTION,
            )
        if total_ues > 50000:
            raise ValidationException(
                f"Total UE count ({total_ues}) exceeds maximum limit (50000)",
                field=constants.UE_CLASS_DISTRIBUTION,
            )

    def _validate_lat_lon_boundaries(self, boundaries: Dict[str, Any]) -> None:
        self.boundaries_validator.validate(boundaries)

        min_lat = boundaries[constants.MIN_LAT]
        max_lat = boundaries[constants.MAX_LAT]
        min_lon = boundaries[constants.MIN_LON]
        max_lon = boundaries[constants.MAX_LON]

        errors: List[Dict[str, str]] = []
        field = f"{constants.UE_TRACKS}.{constants.UE_TRACKS_GENERATION}.{constants.LON_LAT_BOUNDARIES}"
        if min_lat >= max_lat:
            errors.append({"field": field, "error": "min_lat must be less than max_lat"})
        if min_lon >= max_lon:
            errors.append({"field": field, "error": "min_lon must be less than max_lon"})
        if (max_lat - min_lat) < 0.001 or (max_lon - min_lon) < 0.001:
            errors.append(
                {
                    "field": field,
                    "error": "Geographic area is too small (minimum 0.001 degrees in each dimension)",
                }
            )
        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)

    def _validate_ue_data_id(self, ue_data_id: Any) -> None:
        if not isinstance(ue_data_id, str) or not ue_data_id.strip():
            raise ValidationException(
                "ue_data_id must be a non-empty string", field="ue_tracks.ue_data_id"
            )
        if len(ue_data_id) > 100:
            raise ValidationException(
                "ue_data_id must be 100 characters or less", field="ue_tracks.ue_data_id"
            )

    def _validate_rf_prediction(self, rf_prediction: Dict[str, Any]) -> None:
        self.rf_prediction_validator.validate(rf_prediction)
        self._validate_model_exists(rf_prediction[constants.MODEL_ID])

    def _validate_model_exists(self, model_id: str) -> None:
        from radp.common.helpers.file_system_helper import RADPFileSystemHelper

        if not RADPFileSystemHelper.check_model_exists(model_id):
            raise ValidationException(
                f"RF prediction model '{model_id}' does not exist. Train the model first.",
                field=f"{constants.RF_PREDICTION}.{constants.MODEL_ID}",
            )

    def _validate_protocol_emulation(self, protocol_emulation: Dict[str, Any]) -> None:
        errors: List[Dict[str, str]] = []
        if "ttt_seconds" in protocol_emulation:
            value = protocol_emulation["ttt_seconds"]
            if not self._is_number(value) or value < 0 or value > 10:
                errors.append(
                    {
                        "field": f"{constants.PROTOCOL_EMULATION}.ttt_seconds",
                        "error": "must be between 0 and 10 seconds",
                    }
                )
        if "hysteresis" in protocol_emulation:
            value = protocol_emulation["hysteresis"]
            if not self._is_number(value) or value < 0 or value > 20:
                errors.append(
                    {
                        "field": f"{constants.PROTOCOL_EMULATION}.hysteresis",
                        "error": "must be between 0 and 20 dB",
                    }
                )
        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)

    def _validate_simulation_constraints(self, data: Dict[str, Any]) -> None:
        interval = data[constants.SIMULATION_TIME_INTERVAL]
        ue_tracks = data[constants.UE_TRACKS]
        if constants.UE_TRACKS_GENERATION not in ue_tracks:
            return

        generation = ue_tracks[constants.UE_TRACKS_GENERATION]
        total_ues = sum(
            params.get(constants.COUNT, 0)
            for params in generation[constants.UE_CLASS_DISTRIBUTION].values()
        )
        duration = generation[constants.SIMULATION_DURATION]
        tick_count = duration / interval

        if total_ues > 1000 and interval < 0.01:
            raise ValidationException(
                "High UE count with very small time intervals may cause performance issues. "
                "Reduce UE count or increase simulation_time_interval_seconds.",
                field="simulation_constraints",
            )
        if total_ues * tick_count > 10_000_000:
            raise ValidationException(
                "Requested simulation is too large. Reduce UE count, duration, or tick frequency.",
                field="simulation_constraints",
            )

    def _validate_ue_data_file(self, file_path: str) -> pd.DataFrame:
        filename = "ue_data.csv"
        FileValidator.validate_file_size(file_path, max_size_mb=1000, filename=filename)
        df = FileValidator.validate_csv_file(
            file_path=file_path,
            required_columns=["mock_ue_id", "lon", "lat", "tick"],
            filename=filename,
            max_rows=1_000_000,
        )
        errors: List[Dict[str, str]] = []
        errors.extend(self._range_errors(df, "lat", -90, 90))
        errors.extend(self._range_errors(df, "lon", -180, 180))
        errors.extend(self._range_errors(df, "tick", 0, float("inf")))
        if errors:
            raise FileValidationException(
                "UE data file validation failed",
                filename=filename,
                validation_errors=errors,
            )
        return df

    def _validate_config_file(self, file_path: str) -> pd.DataFrame:
        filename = "config.csv"
        FileValidator.validate_file_size(file_path, max_size_mb=10, filename=filename)
        df = FileValidator.validate_csv_file(
            file_path=file_path,
            required_columns=["cell_id", "cell_el_deg"],
            filename=filename,
            max_rows=100_000,
        )
        errors: List[Dict[str, str]] = []
        errors.extend(self._range_errors(df, "cell_el_deg", 0, 15))
        duplicate_cells = sorted(
            df.loc[df.duplicated(subset=["cell_id"]), "cell_id"].astype(str).unique()
        )
        if duplicate_cells:
            errors.append(
                {
                    "field": "cell_id",
                    "error": f"Found duplicate cell IDs in config: {', '.join(duplicate_cells)}",
                }
            )
        if errors:
            raise FileValidationException(
                "Config file validation failed",
                filename=filename,
                validation_errors=errors,
            )
        return df

    @staticmethod
    def _range_errors(
        df: pd.DataFrame, column: str, lower: float, upper: float
    ) -> Iterable[Dict[str, str]]:
        values = pd.to_numeric(df[column], errors="coerce")
        errors: List[Dict[str, str]] = []
        invalid_type_count = int(values.isna().sum())
        if invalid_type_count:
            errors.append(
                {
                    "field": column,
                    "error": f"must be numeric. Found {invalid_type_count} non-numeric values.",
                }
            )
        invalid_range = values.notna() & ((values < lower) | (values > upper))
        invalid_count = int(invalid_range.sum())
        if invalid_count:
            errors.append(
                {
                    "field": column,
                    "error": f"must be between {lower} and {upper}. Found {invalid_count} invalid values.",
                }
            )
        return errors

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
