# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import mimetypes
from typing import List, Optional
import pandas as pd
from werkzeug.datastructures import FileStorage

from api_manager.exceptions.validation_exception import FileValidationException


class UploadedFileValidator:
    """Validator for uploaded files through Flask requests"""
    
    ALLOWED_EXTENSIONS = {'.csv', '.json'}
    MAX_FILENAME_LENGTH = 255
    
    @staticmethod
    def validate_uploaded_file(file: FileStorage, expected_extension: str = None) -> None:
        """Validate uploaded file basic properties"""
        if not file:
            raise FileValidationException("No file provided")
        
        if not file.filename:
            raise FileValidationException("File must have a filename")
        
        # Check filename length
        if len(file.filename) > UploadedFileValidator.MAX_FILENAME_LENGTH:
            raise FileValidationException(
                f"Filename too long (max {UploadedFileValidator.MAX_FILENAME_LENGTH} characters)",
                filename=file.filename
            )
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in UploadedFileValidator.ALLOWED_EXTENSIONS:
            raise FileValidationException(
                f"Invalid file extension '{file_ext}'. Allowed: {', '.join(UploadedFileValidator.ALLOWED_EXTENSIONS)}",
                filename=file.filename
            )
        
        if expected_extension and file_ext != expected_extension:
            raise FileValidationException(
                f"Expected {expected_extension} file, got {file_ext}",
                filename=file.filename
            )
        
        # Check MIME type for additional security
        UploadedFileValidator._validate_mime_type(file)
    
    @staticmethod
    def _validate_mime_type(file: FileStorage) -> None:
        """Validate MIME type matches extension"""
        if not file.filename:
            return
            
        file_ext = os.path.splitext(file.filename)[1].lower()
        expected_mime_types = {
            '.csv': ['text/csv', 'text/plain', 'application/csv'],
            '.json': ['application/json', 'text/json', 'text/plain']
        }
        
        if file_ext in expected_mime_types:
            # Read a small portion to check MIME type
            file.seek(0)
            sample = file.read(1024)
            file.seek(0)  # Reset for later reading
            
            detected_type, _ = mimetypes.guess_type(file.filename)
            
            if detected_type and detected_type not in expected_mime_types[file_ext]:
                # This is a warning rather than an error since MIME detection can be unreliable
                pass  # Could add logging here
    
    @staticmethod
    def validate_csv_structure(file: FileStorage, required_columns: List[str], 
                             max_rows: int = 1000000) -> None:
        """Validate CSV file structure without saving to disk"""
        try:
            # Read CSV directly from FileStorage
            file.seek(0)
            df = pd.read_csv(file)
            file.seek(0)  # Reset for later reading
            
            # Check required columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise FileValidationException(
                    f"Missing required columns: {', '.join(missing_columns)}",
                    filename=file.filename
                )
            
            # Check row count
            if len(df) > max_rows:
                raise FileValidationException(
                    f"CSV file has too many rows ({len(df)}). Maximum allowed: {max_rows}",
                    filename=file.filename
                )
            
            # Check for completely empty columns
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                raise FileValidationException(
                    f"Columns are completely empty: {', '.join(empty_columns)}",
                    filename=file.filename
                )
                
        except pd.errors.ParserError as e:
            raise FileValidationException(
                f"CSV parsing error: {str(e)}",
                filename=file.filename
            )
        except Exception as e:
            if isinstance(e, FileValidationException):
                raise
            raise FileValidationException(
                f"Error reading CSV file: {str(e)}",
                filename=file.filename
            )


class ContentValidator:
    """Validator for file content security and safety"""
    
    DANGEROUS_PATTERNS = [
        r'<script\s*.*?>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'vbscript:',      # VBScript URLs
        r'on\w+\s*=',      # Event handlers
        r'eval\s*\(',      # eval() calls
        r'exec\s*\(',      # exec() calls
    ]
    
    @staticmethod
    def validate_content_safety(content: str, filename: str = None) -> None:
        """Validate content doesn't contain dangerous patterns"""
        import re
        
        content_lower = content.lower()
        
        for pattern in ContentValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                raise FileValidationException(
                    f"File contains potentially dangerous content matching pattern: {pattern}",
                    filename=filename
                )
    
    @staticmethod
    def validate_json_structure(file: FileStorage, required_keys: List[str] = None) -> dict:
        """Validate JSON file structure"""
        import json
        
        try:
            file.seek(0)
            content = file.read().decode('utf-8')
            file.seek(0)
            
            # Check for dangerous content
            ContentValidator.validate_content_safety(content, file.filename)
            
            # Parse JSON
            data = json.loads(content)
            
            # Check required keys if specified
            if required_keys:
                missing_keys = set(required_keys) - set(data.keys())
                if missing_keys:
                    raise FileValidationException(
                        f"Missing required JSON keys: {', '.join(missing_keys)}",
                        filename=file.filename
                    )
            
            return data
            
        except json.JSONDecodeError as e:
            raise FileValidationException(
                f"Invalid JSON format: {str(e)}",
                filename=file.filename
            )
        except UnicodeDecodeError as e:
            raise FileValidationException(
                f"File encoding error: {str(e)}",
                filename=file.filename
            )