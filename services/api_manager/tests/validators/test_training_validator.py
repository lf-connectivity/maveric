# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import pandas as pd

from api_manager.validators.training_validator import TrainingRequestValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException


class TestTrainingRequestValidator(unittest.TestCase):
    
    def setUp(self):
        self.validator = TrainingRequestValidator()
    
    def test_valid_training_request(self):
        """Test validation of valid training request"""
        valid_request = {
            "model_id": "test_model_123",
            "model_update": False,
            "params": {
                "maxiter": 100,
                "lr": 0.05,
                "stopping_threshold": 0.0001
            }
        }
        
        with patch.object(self.validator, '_validate_model_id_availability'):
            self.validator.validate(valid_request)
    
    def test_missing_required_field(self):
        """Test validation fails for missing required fields"""
        invalid_request = {
            "model_update": False,
            "params": {}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        # Check validation errors contain model_id field
        self.assertTrue(any("model_id" in str(error) for error in context.exception.validation_errors))
    
    def test_invalid_model_id_pattern(self):
        """Test validation fails for invalid model_id pattern"""
        invalid_request = {
            "model_id": "test model with spaces!",
            "params": {}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        # Check validation errors contain pattern-related error
        self.assertTrue(any("pattern" in str(error) for error in context.exception.validation_errors))
    
    def test_invalid_training_params(self):
        """Test validation fails for invalid training parameters"""
        invalid_request = {
            "model_id": "test_model",
            "params": {
                "maxiter": -5,  # Invalid: negative
                "lr": 2.0,      # Invalid: > 1.0
                "stopping_threshold": 1.0  # Invalid: > 0.1
            }
        }
        
        with patch.object(self.validator, '_validate_model_id_availability'):
            with self.assertRaises(ValidationException):
                self.validator.validate(invalid_request)
    
    def test_validate_training_files_success(self):
        """Test successful validation of training files"""
        # Create temporary CSV files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
            ue_file.write("cell_id,avg_rsrp,lon,lat,cell_el_deg\n")
            ue_file.write("cell_1,-80,139.699,35.644,5\n")
            ue_file.write("cell_2,-75,139.700,35.645,3\n")
            ue_file_path = ue_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
            topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
            topo_file.write("35.690,139.691,cell_1,0,2100\n")
            topo_file.write("35.691,139.692,cell_2,120,2100\n")
            topo_file_path = topo_file.name
        
        try:
            files = {
                "ue_training_data_file_path": ue_file_path,
                "topology_file_path": topo_file_path
            }
            
            self.validator.validate_training_files(files)
            
        finally:
            os.unlink(ue_file_path)
            os.unlink(topo_file_path)
    
    def test_validate_training_files_missing_columns(self):
        """Test validation fails for missing columns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
            ue_file.write("cell_id,avg_rsrp\n")  # Missing columns
            ue_file.write("cell_1,-80\n")
            ue_file_path = ue_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
            topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
            topo_file.write("35.690,139.691,cell_1,0,2100\n")
            topo_file_path = topo_file.name
        
        try:
            files = {
                "ue_training_data_file_path": ue_file_path,
                "topology_file_path": topo_file_path
            }
            
            with self.assertRaises(FileValidationException):
                self.validator.validate_training_files(files)
                
        finally:
            os.unlink(ue_file_path)
            os.unlink(topo_file_path)
    
    def test_validate_training_files_invalid_rsrp(self):
        """Test validation fails for invalid RSRP values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
            ue_file.write("cell_id,avg_rsrp,lon,lat,cell_el_deg\n")
            ue_file.write("cell_1,50,139.699,35.644,5\n")  # Invalid RSRP > 0
            ue_file_path = ue_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
            topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
            topo_file.write("35.690,139.691,cell_1,0,2100\n")
            topo_file_path = topo_file.name
        
        try:
            files = {
                "ue_training_data_file_path": ue_file_path,
                "topology_file_path": topo_file_path
            }
            
            with self.assertRaises(FileValidationException):
                self.validator.validate_training_files(files)
                
        finally:
            os.unlink(ue_file_path)
            os.unlink(topo_file_path)
    
    def test_validate_cell_consistency_failure(self):
        """Test validation fails when training data has cells not in topology"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
            ue_file.write("cell_id,avg_rsrp,lon,lat,cell_el_deg\n")
            ue_file.write("cell_1,-80,139.699,35.644,5\n")
            ue_file.write("cell_999,-75,139.700,35.645,3\n")  # Cell not in topology
            ue_file_path = ue_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
            topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
            topo_file.write("35.690,139.691,cell_1,0,2100\n")
            topo_file_path = topo_file.name
        
        try:
            files = {
                "ue_training_data_file_path": ue_file_path,
                "topology_file_path": topo_file_path
            }
            
            with self.assertRaises(ValidationException):
                self.validator.validate_training_files(files)
                
        finally:
            os.unlink(ue_file_path)
            os.unlink(topo_file_path)


if __name__ == '__main__':
    unittest.main()