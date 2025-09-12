# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Any
from api_manager.exceptions.base_api_exception import APIException


class ValidationException(APIException):
    """Exception raised when request validation fails"""
    
    def __init__(self, message: str, field: str = None, validation_errors: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = 400
        self.field = field
        self.validation_errors = validation_errors or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API response"""
        result = {
            "error": self.message,
            "error_type": "validation_error", 
            "status_code": self.code
        }
        
        if self.field:
            result["field"] = self.field
            
        if self.validation_errors:
            result["validation_errors"] = self.validation_errors
            
        return result


class FileValidationException(ValidationException):
    """Exception raised when file validation fails"""
    
    def __init__(self, message: str, filename: str = None, line_number: int = None, validation_errors: List[Dict[str, Any]] = None):
        super().__init__(message, validation_errors=validation_errors)
        self.filename = filename
        self.line_number = line_number
        
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["error_type"] = "file_validation_error"
        
        if self.filename:
            result["filename"] = self.filename
            
        if self.line_number:
            result["line_number"] = self.line_number
            
        return result