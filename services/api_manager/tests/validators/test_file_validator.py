# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
from werkzeug.datastructures import FileStorage

from api_manager.exceptions.validation_exception import FileValidationException
from api_manager.validators.file_validator import FileValidator, UploadedFileValidator


class TestUploadedFileValidator(unittest.TestCase):
    def _file(self, filename: str, content: bytes) -> FileStorage:
        return FileStorage(stream=BytesIO(content), filename=filename)

    def test_validate_json_structure(self):
        payload = UploadedFileValidator.validate_json_structure(
            self._file("payload", b'{"model_id": "m1"}'),
            required_keys=["model_id"],
        )

        self.assertEqual(payload["model_id"], "m1")

    def test_validate_json_rejects_dangerous_content(self):
        with self.assertRaises(FileValidationException):
            UploadedFileValidator.validate_json_structure(
                self._file("payload.json", b'{"value": "<script>alert(1)</script>"}')
            )

    def test_validate_csv_structure(self):
        df = UploadedFileValidator.validate_csv_structure(
            self._file("ue_data.csv", b"mock_ue_id,lon,lat,tick\n1,1,2,0\n"),
            ["mock_ue_id", "lon", "lat", "tick"],
        )

        self.assertEqual(len(df), 1)

    def test_validate_csv_rejects_missing_columns(self):
        with self.assertRaises(FileValidationException) as ctx:
            UploadedFileValidator.validate_csv_structure(
                self._file("ue_data.csv", b"mock_ue_id,lon\n1,2\n"),
                ["mock_ue_id", "lon", "lat", "tick"],
            )

        self.assertIn("Missing required columns", ctx.exception.message)

    def test_validate_uploaded_file_rejects_extension(self):
        with self.assertRaises(FileValidationException):
            UploadedFileValidator.validate_uploaded_file(
                self._file("payload.exe", b"content")
            )


class TestFileValidator(unittest.TestCase):
    def test_validate_csv_file_rejects_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
            file_path = file.name

        try:
            with self.assertRaises(FileValidationException):
                FileValidator.validate_csv_file(file_path, ["cell_id"], "empty.csv")
        finally:
            os.unlink(file_path)

    def test_validate_file_size(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
            file.write("cell_id\ncell_1\n")
            file_path = file.name

        try:
            FileValidator.validate_file_size(file_path, max_size_mb=1, filename="x.csv")
        finally:
            os.unlink(file_path)

    def test_validate_dataframe_rejects_all_empty_required_column(self):
        df = pd.DataFrame({"cell_id": [None]})

        with self.assertRaises(FileValidationException):
            FileValidator.validate_dataframe(df, ["cell_id"], "x.csv")
