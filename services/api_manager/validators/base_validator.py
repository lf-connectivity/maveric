# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from api_manager.exceptions.validation_exception import ValidationException


class BaseValidator(ABC):
    """Base class for request validators."""

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate data or raise ValidationException."""


class SchemaValidator(BaseValidator):
    """Small schema validator for request dictionaries.

    Supported field config keys:
    required, type, min, max, min_length, max_length, pattern, enum, validator.
    """

    def __init__(
        self,
        schema: Dict[str, Dict[str, Any]],
        allow_unknown: bool = True,
        field_prefix: str = "",
    ):
        self.schema = schema
        self.allow_unknown = allow_unknown
        self.field_prefix = field_prefix

    def validate(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValidationException(
                "Validation failed",
                validation_errors=[
                    {
                        "field": self.field_prefix.rstrip(".") or "request",
                        "error": "must be an object",
                    }
                ],
            )

        errors: List[Dict[str, Any]] = []

        if not self.allow_unknown:
            for field_name in set(data.keys()) - set(self.schema.keys()):
                errors.append(
                    {
                        "field": self._field_path(field_name),
                        "error": "is not an allowed field",
                    }
                )

        for field_name, field_config in self.schema.items():
            errors.extend(self._validate_field(data, field_name, field_config))

        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)

    def _field_path(self, field_name: str) -> str:
        return f"{self.field_prefix}{field_name}" if self.field_prefix else field_name

    def _validate_field(
        self,
        data: Dict[str, Any],
        field_name: str,
        field_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        field_path = self._field_path(field_name)
        value = data.get(field_name)
        errors: List[Dict[str, Any]] = []

        if field_config.get("required") and value is None:
            return [{"field": field_path, "error": "is required"}]

        if value is None:
            return []

        expected_type = field_config.get("type")
        if expected_type and (
            not isinstance(value, expected_type)
            or self._bool_is_invalid_numeric(value, expected_type)
        ):
            errors.append(
                {
                    "field": field_path,
                    "error": self._type_error(expected_type, type(value)),
                }
            )
            return errors

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = field_config.get("min")
            maximum = field_config.get("max")
            if minimum is not None and value < minimum:
                errors.append({"field": field_path, "error": f"must be >= {minimum}"})
            if maximum is not None and value > maximum:
                errors.append({"field": field_path, "error": f"must be <= {maximum}"})

        if isinstance(value, str):
            min_length = field_config.get("min_length")
            max_length = field_config.get("max_length")
            if min_length is not None and len(value) < min_length:
                errors.append(
                    {"field": field_path, "error": f"must be at least {min_length} characters"}
                )
            if max_length is not None and len(value) > max_length:
                errors.append(
                    {"field": field_path, "error": f"must be at most {max_length} characters"}
                )

        pattern = field_config.get("pattern")
        if pattern and isinstance(value, str) and not re.fullmatch(pattern, value):
            errors.append({"field": field_path, "error": "does not match required pattern"})

        allowed_values = field_config.get("enum")
        if allowed_values is not None and value not in allowed_values:
            errors.append(
                {
                    "field": field_path,
                    "error": f"must be one of {sorted(allowed_values)}",
                }
            )

        custom_validator: Optional[Callable[[Any, str], None]] = field_config.get(
            "validator"
        )
        if custom_validator:
            try:
                custom_validator(value, field_path)
            except ValidationException as exc:
                if exc.validation_errors:
                    errors.extend(exc.validation_errors)
                else:
                    errors.append(
                        {"field": exc.field or field_path, "error": exc.message}
                    )

        return errors

    @staticmethod
    def _bool_is_invalid_numeric(value: Any, expected_type: Any) -> bool:
        if not isinstance(value, bool):
            return False
        if expected_type is bool:
            return False
        if isinstance(expected_type, tuple) and bool in expected_type:
            return False
        numeric_types = (int, float)
        if expected_type in numeric_types:
            return True
        return isinstance(expected_type, tuple) and any(
            expected in numeric_types for expected in expected_type
        )

    @staticmethod
    def _type_error(expected_type: Any, actual_type: type) -> str:
        if isinstance(expected_type, tuple):
            expected_names = ", ".join(t.__name__ for t in expected_type)
        else:
            expected_names = expected_type.__name__
        return f"must be of type {expected_names}, got {actual_type.__name__}"
