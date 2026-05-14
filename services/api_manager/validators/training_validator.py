# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Iterable, List

import pandas as pd

from api_manager.exceptions.validation_exception import (
    FileValidationException,
    ValidationException,
)
from api_manager.validators.base_validator import BaseValidator, SchemaValidator
from api_manager.validators.file_validator import FileValidator
from radp.common import constants


class TrainingRequestValidator(BaseValidator):
    """Validator for RF digital twin training API requests."""

    MODEL_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
    UE_TRAINING_COLUMNS = ["cell_id", "avg_rsrp", "lon", "lat", "cell_el_deg"]
    TOPOLOGY_COLUMNS = [
        "cell_lat",
        "cell_lon",
        "cell_id",
        "cell_az_deg",
        "cell_carrier_freq_mhz",
    ]

    TRAINING_SCHEMA = {
        "model_id": {
            "required": True,
            "type": str,
            "min_length": 1,
            "max_length": 100,
            "pattern": MODEL_ID_PATTERN,
        },
        "model_update": {"required": False, "type": bool},
        "params": {"required": True, "type": dict},
    }

    TRAINING_PARAMS_SCHEMA = {
        "maxiter": {"required": False, "type": int, "min": 1, "max": 10000},
        "lr": {"required": False, "type": (int, float), "min": 0.0001, "max": 1.0},
        "stopping_threshold": {
            "required": False,
            "type": (int, float),
            "min": 1e-8,
            "max": 1e-1,
        },
    }

    def __init__(self):
        self.schema_validator = SchemaValidator(
            self.TRAINING_SCHEMA, allow_unknown=False
        )
        self.params_validator = SchemaValidator(
            self.TRAINING_PARAMS_SCHEMA, allow_unknown=False, field_prefix="params."
        )

    def validate(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            self.schema_validator.validate(data)
            return

        errors: List[Dict[str, str]] = []
        try:
            self.schema_validator.validate(data)
        except ValidationException as exc:
            errors.extend(exc.validation_errors or [{"field": exc.field or "request", "error": exc.message}])

        params = data.get("params", {})
        if isinstance(params, dict):
            try:
                self.params_validator.validate(params)
            except ValidationException as exc:
                errors.extend(exc.validation_errors or [{"field": exc.field or "params", "error": exc.message}])

        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)

        self._validate_model_id_availability(
            model_id=data["model_id"],
            is_update=data.get("model_update", False),
        )

    def validate_training_files(self, files: Dict[str, str]) -> None:
        required_file_keys = [
            constants.UE_TRAINING_DATA_FILE_PATH_KEY,
            constants.TOPOLOGY_FILE_PATH_KEY,
        ]
        missing = [file_key for file_key in required_file_keys if file_key not in files]
        if missing:
            raise ValidationException(
                "Validation failed",
                validation_errors=[
                    {"field": file_key, "error": "required training file is missing"}
                    for file_key in missing
                ],
            )

        for file_key in required_file_keys:
            if not os.path.exists(files[file_key]):
                raise FileValidationException(
                    f"File not found: {files[file_key]}",
                    filename=os.path.basename(files[file_key]),
                )

        training_df = self._validate_ue_training_data(
            files[constants.UE_TRAINING_DATA_FILE_PATH_KEY]
        )
        topology_df = self._validate_topology_file(
            files[constants.TOPOLOGY_FILE_PATH_KEY]
        )
        self._validate_cell_consistency(training_df, topology_df)

    def _validate_model_id_availability(self, model_id: str, is_update: bool) -> None:
        from radp.common.helpers.file_system_helper import RADPFileSystemHelper

        model_exists = RADPFileSystemHelper.check_model_exists(model_id)
        if not is_update and model_exists:
            raise ValidationException(
                f"Model '{model_id}' already exists. Use model_update=true to update existing model.",
                field="model_id",
            )
        if is_update and not model_exists:
            raise ValidationException(
                f"Model '{model_id}' does not exist. Cannot update non-existent model.",
                field="model_id",
            )

    def _validate_ue_training_data(self, file_path: str) -> pd.DataFrame:
        filename = "ue_training_data.csv"
        FileValidator.validate_file_size(file_path, max_size_mb=500, filename=filename)
        df = FileValidator.validate_csv_file(
            file_path=file_path,
            required_columns=self.UE_TRAINING_COLUMNS,
            filename=filename,
            max_rows=1_000_000,
        )

        errors: List[Dict[str, str]] = []
        errors.extend(self._range_errors(df, "avg_rsrp", -150, 0))
        errors.extend(self._range_errors(df, "lat", -90, 90))
        errors.extend(self._range_errors(df, "lon", -180, 180))
        errors.extend(self._range_errors(df, "cell_el_deg", 0, 15))

        if errors:
            raise FileValidationException(
                "UE training data validation failed",
                filename=filename,
                validation_errors=errors,
            )
        return df

    def _validate_topology_file(self, file_path: str) -> pd.DataFrame:
        filename = "topology.csv"
        FileValidator.validate_file_size(file_path, max_size_mb=10, filename=filename)
        df = FileValidator.validate_csv_file(
            file_path=file_path,
            required_columns=self.TOPOLOGY_COLUMNS,
            filename=filename,
            max_rows=100_000,
        )

        errors: List[Dict[str, str]] = []
        errors.extend(self._range_errors(df, "cell_lat", -90, 90))
        errors.extend(self._range_errors(df, "cell_lon", -180, 180))
        errors.extend(self._range_errors(df, "cell_az_deg", 0, 360, upper_inclusive=False))
        errors.extend(self._range_errors(df, "cell_carrier_freq_mhz", 400, 6000))

        duplicate_cells = sorted(df.loc[df.duplicated(subset=["cell_id"]), "cell_id"].astype(str).unique())
        if duplicate_cells:
            errors.append(
                {
                    "field": "cell_id",
                    "error": f"Found duplicate cell IDs: {', '.join(duplicate_cells)}",
                }
            )

        if errors:
            raise FileValidationException(
                "Topology file validation failed",
                filename=filename,
                validation_errors=errors,
            )
        return df

    def _validate_cell_consistency(
        self, training_df: pd.DataFrame, topology_df: pd.DataFrame
    ) -> None:
        training_cells = set(training_df["cell_id"].astype(str).unique())
        topology_cells = set(topology_df["cell_id"].astype(str).unique())
        missing_cells = sorted(training_cells - topology_cells)
        if missing_cells:
            raise ValidationException(
                "Training data contains cells not found in topology",
                validation_errors=[
                    {
                        "field": "cell_id",
                        "error": f"Cells missing from topology: {', '.join(missing_cells)}",
                    }
                ],
            )

    @staticmethod
    def _range_errors(
        df: pd.DataFrame,
        column: str,
        lower: float,
        upper: float,
        upper_inclusive: bool = True,
    ) -> Iterable[Dict[str, str]]:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid_type_count = int(values.isna().sum())
        if upper_inclusive:
            invalid_range = values.notna() & ((values < lower) | (values > upper))
            upper_msg = f"{upper}"
        else:
            invalid_range = values.notna() & ((values < lower) | (values >= upper))
            upper_msg = f"< {upper}"

        errors: List[Dict[str, str]] = []
        if invalid_type_count:
            errors.append(
                {
                    "field": column,
                    "error": f"must be numeric. Found {invalid_type_count} non-numeric values.",
                }
            )
        invalid_count = int(invalid_range.sum())
        if invalid_count:
            if upper_inclusive:
                range_msg = f"between {lower} and {upper_msg}"
            else:
                range_msg = f">= {lower} and {upper_msg}"
            errors.append(
                {
                    "field": column,
                    "error": f"must be {range_msg}. Found {invalid_count} invalid values.",
                }
            )
        return errors
