# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from api_manager.validators.simulation_validator import SimulationRequestValidator
from api_manager.exceptions.validation_exception import ValidationException


class TestSimulationRequestValidator(unittest.TestCase):
    
    def setUp(self):
        self.validator = SimulationRequestValidator()
    
    def test_valid_simulation_request_with_generation(self):
        """Test validation of valid simulation request with UE tracks generation"""
        valid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {
                            "count": 10,
                            "velocity": 1.5,
                            "velocity_variance": 0.5
                        },
                        "car": {
                            "count": 5,
                            "velocity": 15.0,
                            "velocity_variance": 2.0
                        }
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0,
                        "max_lat": 36.0,
                        "min_lon": 139.0,
                        "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8,
                        "rng_seed": 12345
                    }
                }
            },
            "rf_prediction": {
                "model_id": "test_model"
            }
        }
        
        with patch.object(self.validator, '_validate_model_exists'):
            self.validator.validate(valid_request)
    
    def test_valid_simulation_request_with_data_id(self):
        """Test validation of valid simulation request with UE data ID"""
        valid_request = {
            "simulation_time_interval_seconds": 0.1,
            "ue_tracks": {
                "ue_data_id": "test_data_123"
            },
            "rf_prediction": {
                "model_id": "test_model"
            }
        }
        
        with patch.object(self.validator, '_validate_model_exists'):
            self.validator.validate(valid_request)
    
    def test_missing_required_fields(self):
        """Test validation fails for missing required fields"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01
            # Missing ue_tracks and rf_prediction
        }
        
        with self.assertRaises(ValidationException):
            self.validator.validate(invalid_request)
    
    def test_invalid_time_interval(self):
        """Test validation fails for invalid time intervals"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.0005,  # Too small
            "ue_tracks": {"ue_data_id": "test"},
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException):
            self.validator.validate(invalid_request)
    
    def test_ue_tracks_missing_both_options(self):
        """Test validation fails when UE tracks has neither generation nor data_id"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {},  # Empty
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException):
            self.validator.validate(invalid_request)
    
    def test_ue_tracks_has_both_options(self):
        """Test validation fails when UE tracks has both generation and data_id"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_data_id": "test",
                "ue_tracks_generation": {}
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException):
            self.validator.validate(invalid_request)
    
    def test_invalid_ue_class(self):
        """Test validation fails for invalid UE class names"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "invalid_class": {  # Invalid UE class
                            "count": 10,
                            "velocity": 1.0,
                            "velocity_variance": 0.5
                        }
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0, "max_lat": 36.0,
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        self.assertIn("Invalid UE classes", str(context.exception))
    
    def test_invalid_lat_lon_boundaries(self):
        """Test validation fails for invalid lat/lon boundaries"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {"count": 10, "velocity": 1.0, "velocity_variance": 0.5}
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 36.0,  # min > max
                        "max_lat": 35.0,
                        "min_lon": 139.0,
                        "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        self.assertIn("min_lat must be less than max_lat", str(context.exception))
    
    def test_zero_ue_count(self):
        """Test validation fails when total UE count is zero"""
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {"count": 0, "velocity": 1.0, "velocity_variance": 0.5}
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0, "max_lat": 36.0,
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        self.assertIn("Total UE count cannot be zero", str(context.exception))
    
    @patch('radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists')
    def test_excessive_ue_count(self, mock_check_model_exists):
        """Test validation fails when UE count is too high"""
        mock_check_model_exists.return_value = True  # Model exists
        
        invalid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {"count": 60000, "velocity": 1.0, "velocity_variance": 0.5}  # Too many
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0, "max_lat": 36.0,
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(invalid_request)
        
        # Check validation errors contain count limit error
        error_messages = [str(error) for error in context.exception.validation_errors]
        self.assertTrue(any("must be <= 10000" in msg for msg in error_messages), 
                       f"Expected count limit error in: {error_messages}")
    
    @patch('radp.common.helpers.file_system_helper.RADPFileSystemHelper.check_model_exists')
    def test_rf_prediction_model_not_exists(self, mock_check_model_exists):
        """Test validation fails when RF prediction model doesn't exist"""
        mock_check_model_exists.return_value = False
        
        request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {"ue_data_id": "test"},
            "rf_prediction": {"model_id": "nonexistent_model"}
        }
        
        with self.assertRaises(ValidationException) as context:
            self.validator.validate(request)
        
        self.assertIn("does not exist", str(context.exception))


if __name__ == '__main__':
    unittest.main()