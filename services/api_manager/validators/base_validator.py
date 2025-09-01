# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

from api_manager.exceptions.validation_exception import ValidationException, FileValidationException


class BaseValidator(ABC):
    """Base class for all validators"""
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate the provided data. Raises ValidationException if invalid."""
        pass


class SchemaValidator(BaseValidator):
    """Generic schema validator with common validation patterns"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate data against schema"""
        errors = []
        
        for field_name, field_config in self.schema.items():
            try:
                self._validate_field(data, field_name, field_config)
            except ValidationException as e:
                errors.extend(e.validation_errors if e.validation_errors else [{"field": field_name, "error": e.message}])
        
        if errors:
            raise ValidationException("Validation failed", validation_errors=errors)
    
    def _validate_field(self, data: Dict[str, Any], field_name: str, field_config: Dict[str, Any]) -> None:
        """Validate a single field"""
        value = data.get(field_name)
        
        # Check required fields
        if field_config.get("required", False) and value is None:
            raise ValidationException(f"Field '{field_name}' is required", field=field_name)
        
        # Skip validation for optional fields that are None
        if value is None and not field_config.get("required", False):
            return
            
        # Type validation
        expected_type = field_config.get("type")
        if expected_type and not isinstance(value, expected_type):
            raise ValidationException(
                f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                field=field_name
            )
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = field_config.get("min")
            max_val = field_config.get("max")
            
            if min_val is not None and value < min_val:
                raise ValidationException(f"Field '{field_name}' must be >= {min_val}", field=field_name)
            if max_val is not None and value > max_val:
                raise ValidationException(f"Field '{field_name}' must be <= {max_val}", field=field_name)
        
        # String length validation
        if isinstance(value, str):
            min_length = field_config.get("min_length", 0)
            max_length = field_config.get("max_length")
            
            if len(value) < min_length:
                raise ValidationException(f"Field '{field_name}' must be at least {min_length} characters", field=field_name)
            if max_length and len(value) > max_length:
                raise ValidationException(f"Field '{field_name}' must be at most {max_length} characters", field=field_name)
        
        # Pattern validation
        pattern = field_config.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                raise ValidationException(f"Field '{field_name}' does not match required pattern", field=field_name)
        
        # Enum validation
        allowed_values = field_config.get("enum")
        if allowed_values and value not in allowed_values:
            raise ValidationException(f"Field '{field_name}' must be one of {allowed_values}", field=field_name)
        
        # Custom validation function
        custom_validator = field_config.get("validator")
        if custom_validator:
            custom_validator(value, field_name)


class FileValidator:
    """Validator for uploaded files"""
    
    @staticmethod
    def validate_csv_file(file_path: str, required_columns: List[str], filename: str = None) -> pd.DataFrame:
        """Validate CSV file format and required columns"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise FileValidationException(f"Invalid CSV file format: {str(e)}", filename=filename)
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise FileValidationException(
                f"Missing required columns: {', '.join(missing_columns)}",
                filename=filename
            )
        
        # Check for empty dataframe
        if df.empty:
            raise FileValidationException("CSV file cannot be empty", filename=filename)
        
        return df
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: int = 100, filename: str = None) -> None:
        """Validate file size"""
        import os
        
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                raise FileValidationException(
                    f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)",
                    filename=filename
                )
        except OSError as e:
            raise FileValidationException(f"Error reading file: {str(e)}", filename=filename)