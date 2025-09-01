# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
from io import BytesIO
from werkzeug.datastructures import FileStorage

from api_manager.validators.file_validator import UploadedFileValidator, ContentValidator
from api_manager.exceptions.validation_exception import FileValidationException


class TestUploadedFileValidator(unittest.TestCase):
    
    def test_valid_csv_file(self):
        """Test validation of valid CSV file"""
        csv_content = b"col1,col2\nval1,val2\n"
        file_storage = FileStorage(
            stream=BytesIO(csv_content),
            filename="test.csv",
            content_type="text/csv"
        )
        
        UploadedFileValidator.validate_uploaded_file(file_storage, ".csv")
    
    def test_invalid_file_extension(self):
        """Test validation fails for invalid file extension"""
        content = b"some content"
        file_storage = FileStorage(
            stream=BytesIO(content),
            filename="test.txt",  # Invalid extension
            content_type="text/plain"
        )
        
        with self.assertRaises(FileValidationException) as context:
            UploadedFileValidator.validate_uploaded_file(file_storage)
        
        self.assertIn("Invalid file extension", str(context.exception))
    
    def test_missing_filename(self):
        """Test validation fails when filename is missing"""
        content = b"some content"
        file_storage = FileStorage(
            stream=BytesIO(content),
            filename=None
        )
        
        with self.assertRaises(FileValidationException):
            UploadedFileValidator.validate_uploaded_file(file_storage)
    
    def test_filename_too_long(self):
        """Test validation fails for excessively long filenames"""
        content = b"some content"
        long_filename = "a" * 300 + ".csv"  # > 255 chars
        file_storage = FileStorage(
            stream=BytesIO(content),
            filename=long_filename
        )
        
        with self.assertRaises(FileValidationException):
            UploadedFileValidator.validate_uploaded_file(file_storage)
    
    def test_validate_csv_structure_success(self):
        """Test successful CSV structure validation"""
        csv_content = b"col1,col2,col3\nval1,val2,val3\nval4,val5,val6\n"
        file_storage = FileStorage(
            stream=BytesIO(csv_content),
            filename="test.csv"
        )
        
        UploadedFileValidator.validate_csv_structure(
            file_storage, 
            required_columns=["col1", "col2"]
        )
    
    def test_validate_csv_structure_missing_columns(self):
        """Test CSV structure validation fails for missing columns"""
        csv_content = b"col1,col2\nval1,val2\n"
        file_storage = FileStorage(
            stream=BytesIO(csv_content),
            filename="test.csv"
        )
        
        with self.assertRaises(FileValidationException) as context:
            UploadedFileValidator.validate_csv_structure(
                file_storage,
                required_columns=["col1", "col2", "col3"]  # col3 missing
            )
        
        self.assertIn("Missing required columns", str(context.exception))
    
    def test_validate_csv_structure_too_many_rows(self):
        """Test CSV structure validation fails for too many rows"""
        # Create CSV content with many rows
        csv_lines = ["col1,col2\n"]
        csv_lines.extend([f"val{i},val{i}\n" for i in range(1001)])  # 1001 rows
        csv_content = "".join(csv_lines).encode()
        
        file_storage = FileStorage(
            stream=BytesIO(csv_content),
            filename="test.csv"
        )
        
        with self.assertRaises(FileValidationException) as context:
            UploadedFileValidator.validate_csv_structure(
                file_storage,
                required_columns=["col1"],
                max_rows=1000
            )
        
        self.assertIn("too many rows", str(context.exception))


class TestContentValidator(unittest.TestCase):
    
    def test_safe_content(self):
        """Test validation passes for safe content"""
        safe_content = "col1,col2\nvalue1,value2\n"
        ContentValidator.validate_content_safety(safe_content)
    
    def test_dangerous_script_content(self):
        """Test validation fails for content with script tags"""
        dangerous_content = "col1,col2\n<script>alert('xss')</script>,value2\n"
        
        with self.assertRaises(FileValidationException):
            ContentValidator.validate_content_safety(dangerous_content, "test.csv")
    
    def test_dangerous_javascript_url(self):
        """Test validation fails for JavaScript URLs"""
        dangerous_content = "col1,col2\njavascript:alert('xss'),value2\n"
        
        with self.assertRaises(FileValidationException):
            ContentValidator.validate_content_safety(dangerous_content, "test.csv")
    
    def test_validate_json_structure_success(self):
        """Test successful JSON structure validation"""
        json_content = b'{"key1": "value1", "key2": "value2"}'
        file_storage = FileStorage(
            stream=BytesIO(json_content),
            filename="test.json"
        )
        
        result = ContentValidator.validate_json_structure(
            file_storage,
            required_keys=["key1"]
        )
        
        self.assertEqual(result["key1"], "value1")
    
    def test_validate_json_structure_missing_keys(self):
        """Test JSON structure validation fails for missing keys"""
        json_content = b'{"key1": "value1"}'
        file_storage = FileStorage(
            stream=BytesIO(json_content),
            filename="test.json"
        )
        
        with self.assertRaises(FileValidationException) as context:
            ContentValidator.validate_json_structure(
                file_storage,
                required_keys=["key1", "key2"]  # key2 missing
            )
        
        self.assertIn("Missing required JSON keys", str(context.exception))
    
    def test_validate_json_structure_invalid_json(self):
        """Test JSON structure validation fails for invalid JSON"""
        invalid_json_content = b'{"key1": "value1"'  # Missing closing brace
        file_storage = FileStorage(
            stream=BytesIO(invalid_json_content),
            filename="test.json"
        )
        
        with self.assertRaises(FileValidationException) as context:
            ContentValidator.validate_json_structure(file_storage)
        
        self.assertIn("Invalid JSON format", str(context.exception))


if __name__ == '__main__':
    unittest.main()