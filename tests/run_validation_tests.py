#!/usr/bin/env python3
"""
Validation-focused test runner that can run without Docker.
Tests the validation system components in isolation and with mock services.
"""

import logging
import sys
import time
from datetime import datetime
import unittest
from typing import List, Tuple
import os

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../radp'))

# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(levelname)s:  %(message)s")
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)
logger.propagate = False


def run_validation_unit_tests() -> Tuple[bool, int, int]:
    """Run all validation unit tests"""
    logger.info("Running validation unit tests...")
    
    loader = unittest.TestLoader()
    
    # Discover validation tests
    validation_test_dir = os.path.join(os.path.dirname(__file__), '../services/api_manager/tests/validators')
    
    if not os.path.exists(validation_test_dir):
        logger.error(f"Validation test directory not found: {validation_test_dir}")
        return False, 0, 0
    
    try:
        test_suite = loader.discover(
            start_dir=validation_test_dir,
            pattern="test_*.py"
        )
        
        total_tests = test_suite.countTestCases()
        logger.info(f"Found {total_tests} validation unit tests")
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        success = result.wasSuccessful()
        failures = len(result.failures)
        errors = len(result.errors)
        
        return success, total_tests, failures + errors
        
    except Exception as e:
        logger.exception(f"Failed to run validation unit tests: {e}")
        return False, 0, 1


def test_validation_system_integration():
    """Test validation system components working together"""
    logger.info("Testing validation system integration...")
    
    try:
        # Import validation components
        from api_manager.validators.training_validator import TrainingRequestValidator
        from api_manager.validators.simulation_validator import SimulationRequestValidator
        from api_manager.exceptions.validation_exception import ValidationException
        
        training_validator = TrainingRequestValidator()
        simulation_validator = SimulationRequestValidator()
        
        # Test 1: Valid training request validation
        valid_training = {
            "model_id": "integration_test_model",
            "model_update": False,
            "params": {
                "maxiter": 100,
                "lr": 0.05,
                "stopping_threshold": 0.0001
            }
        }
        
        # Mock the file system helper
        import unittest.mock
        with unittest.mock.patch.object(training_validator, '_validate_model_id_availability'):
            training_validator.validate(valid_training)
        logger.info("✅ Training request validation working")
        
        # Test 2: Invalid training request validation
        invalid_training = {
            "model_id": "invalid model!",  # Invalid characters
            "params": {
                "maxiter": -5,  # Invalid range
                "lr": 2.0       # Invalid range
            }
        }
        
        try:
            with unittest.mock.patch.object(training_validator, '_validate_model_id_availability'):
                training_validator.validate(invalid_training)
            logger.error("❌ Invalid training request should have been rejected")
            return False
        except ValidationException:
            logger.info("✅ Invalid training request correctly rejected")
        
        # Test 3: Valid simulation request validation  
        valid_simulation = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {"count": 5, "velocity": 1.0, "velocity_variance": 0.5}
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0, "max_lat": 36.0,
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5, "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test_model"}
        }
        
        with unittest.mock.patch.object(simulation_validator, '_validate_model_exists'):
            simulation_validator.validate(valid_simulation)
        logger.info("✅ Simulation request validation working")
        
        # Test 4: Invalid simulation request validation
        invalid_simulation = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "invalid_class": {"count": 5, "velocity": 1.0, "velocity_variance": 0.5}
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 36.0, "max_lat": 35.0,  # Invalid: min > max
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5, "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test_model"}
        }
        
        try:
            simulation_validator.validate(invalid_simulation)
            logger.error("❌ Invalid simulation request should have been rejected")
            return False
        except ValidationException:
            logger.info("✅ Invalid simulation request correctly rejected")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import validation components: {e}")
        return False
    except Exception as e:
        logger.exception(f"❌ Validation integration test failed: {e}")
        return False


def test_error_response_format():
    """Test that validation errors return properly formatted responses"""
    logger.info("Testing error response formatting...")
    
    try:
        from api_manager.exceptions.validation_exception import ValidationException, FileValidationException
        
        # Test ValidationException formatting
        validation_error = ValidationException(
            "Test validation error",
            field="test_field",
            validation_errors=[{"field": "param1", "error": "Invalid value"}]
        )
        
        error_dict = validation_error.to_dict()
        
        required_fields = ["error", "error_type", "status_code"]
        for field in required_fields:
            if field not in error_dict:
                logger.error(f"❌ Missing required field in error response: {field}")
                return False
        
        if error_dict["error_type"] != "validation_error":
            logger.error("❌ Incorrect error type in validation error response")
            return False
        
        if error_dict["status_code"] != 400:
            logger.error("❌ Incorrect status code in validation error response")
            return False
        
        logger.info("✅ ValidationException formatting working correctly")
        
        # Test FileValidationException formatting
        file_error = FileValidationException(
            "Test file validation error",
            filename="test.csv",
            line_number=5
        )
        
        file_error_dict = file_error.to_dict()
        
        if file_error_dict["error_type"] != "file_validation_error":
            logger.error("❌ Incorrect error type in file validation error response")
            return False
        
        if "filename" not in file_error_dict:
            logger.error("❌ Missing filename in file validation error response")
            return False
        
        logger.info("✅ FileValidationException formatting working correctly")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Error response format test failed: {e}")
        return False


def main():
    """Main test runner function"""
    logger.info("🚀 RADP Validation System Test Suite")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    all_passed = True
    total_tests = 0
    failed_tests = 0
    
    # Test 1: Run validation unit tests
    logger.info("--------------------------------------------------------")
    test_start = datetime.now()
    
    unit_success, unit_count, unit_failures = run_validation_unit_tests()
    total_tests += unit_count
    failed_tests += unit_failures
    all_passed = all_passed and unit_success
    
    logger.info(f"Unit Tests Result: {'SUCCESS' if unit_success else 'FAILED'}")
    logger.info(f"Duration: {datetime.now() - test_start}")
    
    # Test 2: Integration testing
    logger.info("--------------------------------------------------------")
    test_start = datetime.now()
    
    integration_success = test_validation_system_integration()
    total_tests += 1
    if not integration_success:
        failed_tests += 1
    all_passed = all_passed and integration_success
    
    logger.info(f"Integration Test Result: {'SUCCESS' if integration_success else 'FAILED'}")
    logger.info(f"Duration: {datetime.now() - test_start}")
    
    # Test 3: Error response formatting
    logger.info("--------------------------------------------------------")
    test_start = datetime.now()
    
    format_success = test_error_response_format()
    total_tests += 1
    if not format_success:
        failed_tests += 1
    all_passed = all_passed and format_success
    
    logger.info(f"Error Format Test Result: {'SUCCESS' if format_success else 'FAILED'}")
    logger.info(f"Duration: {datetime.now() - test_start}")
    
    # Final results
    logger.info("--------------------------------------------------------")
    logger.info("--------------------------------------------------------") 
    logger.info(f"VALIDATION TEST SUMMARY: {'SUCCESS' if all_passed else 'FAILURE'}")
    logger.info(f"TOTAL TESTS RUN: {total_tests}")
    logger.info(f"TESTS PASSED: {total_tests - failed_tests}")
    logger.info(f"TESTS FAILED: {failed_tests}")
    logger.info(f"TOTAL DURATION: {datetime.now() - start_time}")
    logger.info("--------------------------------------------------------")
    
    if all_passed:
        logger.info("✅ All validation system tests passed!")
        logger.info("The comprehensive validation system is working correctly.")
    else:
        logger.info("❌ Some validation system tests failed")
        logger.info("Check the logs above for details on failures.")
    
    logger.info("--------------------------------------------------------")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())