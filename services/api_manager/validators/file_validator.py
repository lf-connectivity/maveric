# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from werkzeug.datastructures import FileStorage

from api_manager.exceptions.validation_exception import FileValidationException


class ContentValidator:
    """Content safety checks for user-supplied uploads."""

    DANGEROUS_PATTERNS = [
        r"<script\b",
        r"javascript:",
        r"vbscript:",
        r"\bon[a-z]+\s*=",
        r"\beval\s*\(",
        r"\bexec\s*\(",
    ]

    @staticmethod
    def validate_content_safety(content: str, filename: Optional[str] = None) -> None:
        for pattern in ContentValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                raise FileValidationException(
                    f"File contains potentially dangerous content matching pattern: {pattern}",
                    filename=filename,
                )


class FileValidator:
    """Validators for already-saved CSV files."""

    @staticmethod
    def validate_csv_file(
        file_path: str,
        required_columns: Iterable[str],
        filename: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileValidationException(
                f"File not found: {file_path}", filename=filename
            )

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise FileValidationException("CSV file cannot be empty", filename=filename)
        except pd.errors.ParserError as exc:
            raise FileValidationException(
                f"CSV parsing error: {str(exc)}", filename=filename
            )
        except Exception as exc:
            raise FileValidationException(
                f"Invalid CSV file format: {str(exc)}", filename=filename
            )

        FileValidator.validate_dataframe(
            df=df,
            required_columns=required_columns,
            filename=filename,
            max_rows=max_rows,
        )
        return df

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Iterable[str],
        filename: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        required = set(required_columns)
        missing = required - set(df.columns)
        if missing:
            raise FileValidationException(
                f"Missing required columns: {', '.join(sorted(missing))}",
                filename=filename,
            )
        if df.empty:
            raise FileValidationException("CSV file cannot be empty", filename=filename)
        if max_rows is not None and len(df) > max_rows:
            raise FileValidationException(
                f"CSV file has too many rows ({len(df)}). Maximum allowed: {max_rows}",
                filename=filename,
            )

        empty_columns = [column for column in required if df[column].isnull().all()]
        if empty_columns:
            raise FileValidationException(
                f"Columns are completely empty: {', '.join(sorted(empty_columns))}",
                filename=filename,
            )

    @staticmethod
    def validate_file_size(
        file_path: str,
        max_size_mb: int,
        filename: Optional[str] = None,
    ) -> None:
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            raise FileValidationException(
                f"Error reading file: {str(exc)}", filename=filename
            )

        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise FileValidationException(
                f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)",
                filename=filename,
            )


class UploadedFileValidator:
    """Validators for Flask FileStorage uploads."""

    ALLOWED_EXTENSIONS = {".csv", ".json"}
    MAX_FILENAME_LENGTH = 255

    @staticmethod
    def validate_uploaded_file(
        file_storage: FileStorage,
        expected_extension: Optional[str] = None,
    ) -> None:
        if not file_storage:
            raise FileValidationException("No file provided")
        if not file_storage.filename:
            raise FileValidationException("File must have a filename")
        if len(file_storage.filename) > UploadedFileValidator.MAX_FILENAME_LENGTH:
            raise FileValidationException(
                f"Filename too long (max {UploadedFileValidator.MAX_FILENAME_LENGTH} characters)",
                filename=file_storage.filename,
            )

        extension = os.path.splitext(file_storage.filename)[1].lower()
        if expected_extension == ".json" and extension == "":
            return

        if extension not in UploadedFileValidator.ALLOWED_EXTENSIONS:
            raise FileValidationException(
                f"Invalid file extension '{extension}'. Allowed: {', '.join(sorted(UploadedFileValidator.ALLOWED_EXTENSIONS))}",
                filename=file_storage.filename,
            )
        if expected_extension and extension != expected_extension:
            raise FileValidationException(
                f"Expected {expected_extension} file, got {extension}",
                filename=file_storage.filename,
            )

    @staticmethod
    def validate_json_structure(
        file_storage: FileStorage,
        required_keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        UploadedFileValidator.validate_uploaded_file(file_storage, ".json")

        try:
            file_storage.stream.seek(0)
            raw_content = file_storage.stream.read()
            file_storage.stream.seek(0)
            if isinstance(raw_content, bytes):
                content = raw_content.decode("utf-8")
            else:
                content = raw_content
        except UnicodeDecodeError as exc:
            raise FileValidationException(
                f"File encoding error: {str(exc)}", filename=file_storage.filename
            )

        ContentValidator.validate_content_safety(content, file_storage.filename)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise FileValidationException(
                f"Invalid JSON format: {str(exc)}", filename=file_storage.filename
            )

        if required_keys:
            missing = set(required_keys) - set(payload.keys())
            if missing:
                raise FileValidationException(
                    f"Missing required JSON keys: {', '.join(sorted(missing))}",
                    filename=file_storage.filename,
                )
        return payload

    @staticmethod
    def validate_csv_structure(
        file_storage: FileStorage,
        required_columns: Iterable[str],
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        UploadedFileValidator.validate_uploaded_file(file_storage, ".csv")
        try:
            file_storage.stream.seek(0)
            raw_content = file_storage.stream.read()
            file_storage.stream.seek(0)
            if isinstance(raw_content, bytes):
                content = raw_content.decode("utf-8")
            else:
                content = raw_content
        except UnicodeDecodeError as exc:
            raise FileValidationException(
                f"File encoding error: {str(exc)}", filename=file_storage.filename
            )

        ContentValidator.validate_content_safety(content, file_storage.filename)

        try:
            df = pd.read_csv(file_storage)
            file_storage.stream.seek(0)
        except pd.errors.EmptyDataError:
            file_storage.stream.seek(0)
            raise FileValidationException(
                "CSV file cannot be empty", filename=file_storage.filename
            )
        except pd.errors.ParserError as exc:
            file_storage.stream.seek(0)
            raise FileValidationException(
                f"CSV parsing error: {str(exc)}", filename=file_storage.filename
            )
        except Exception as exc:
            file_storage.stream.seek(0)
            raise FileValidationException(
                f"Error reading CSV file: {str(exc)}", filename=file_storage.filename
            )

        FileValidator.validate_dataframe(
            df=df,
            required_columns=required_columns,
            filename=file_storage.filename,
            max_rows=max_rows,
        )
        return df
