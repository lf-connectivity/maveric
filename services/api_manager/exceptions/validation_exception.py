# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from api_manager.exceptions.base_api_exception import APIException


class ValidationException(APIException):
    """Exception raised when request validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.code = 400
        self.message = message
        self.field = field
        self.validation_errors = validation_errors or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation failure details to an API response dictionary."""
        response: Dict[str, Any] = {
            "error": self.message,
            "error_type": "validation_error",
            "status_code": self.code,
        }
        if self.field:
            response["field"] = self.field
        if self.validation_errors:
            response["validation_errors"] = self.validation_errors
        return response


class FileValidationException(ValidationException):
    """Exception raised when uploaded file validation fails."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        line_number: Optional[int] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message=message, validation_errors=validation_errors)
        self.filename = filename
        self.line_number = line_number

    def to_dict(self) -> Dict[str, Any]:
        """Convert file validation failure details to an API response dictionary."""
        response = super().to_dict()
        response["error_type"] = "file_validation_error"
        if self.filename:
            response["filename"] = self.filename
        if self.line_number:
            response["line_number"] = self.line_number
        return response
