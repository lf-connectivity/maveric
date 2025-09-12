# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from radp.client.client import RADPClient  # noqa: E402


def test_invalid_training_requests():
    """Test various invalid training request scenarios"""
    print("Testing invalid training requests...")
    
    radp_client = RADPClient()
    
    # Test 1: Invalid model_id with special characters
    try:
        invalid_request = {
            "model_id": "test model with spaces!",  # Invalid characters
            "params": {
                "maxiter": 100,
                "lr": 0.05,
                "stopping_threshold": 0.0001
            }
        }
        
        # Create minimal valid files for testing
        topology_df = pd.DataFrame({
            "cell_lat": [35.690556],
            "cell_lon": [139.691944], 
            "cell_id": ["cell_1"],
            "cell_az_deg": [0],
            "cell_carrier_freq_mhz": [2100]
        })
        
        training_df = pd.DataFrame({
            "cell_id": ["cell_1"],
            "avg_rsrp": [-80],
            "lon": [139.699058],
            "lat": [35.644327],
            "cell_el_deg": [0]
        })
        
        response = radp_client.train(
            model_id=invalid_request["model_id"],
            params=invalid_request["params"],
            ue_training_data=training_df,
            topology=topology_df
        )
        
        print("❌ Invalid model_id should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            if "validation_error" in response_data.get("error_type", ""):
                print("✅ Invalid model_id correctly rejected with validation error")
                print(f"   Error: {response_data.get('error')}")
                return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False
    
    # Test 2: Invalid training parameters
    try:
        invalid_params_request = {
            "model_id": "test_model_invalid_params",
            "params": {
                "maxiter": -5,      # Invalid: negative
                "lr": 2.0,          # Invalid: > 1.0
                "stopping_threshold": 1.0  # Invalid: > 0.1
            }
        }
        
        response = radp_client.train(
            model_id=invalid_params_request["model_id"],
            params=invalid_params_request["params"],
            ue_training_data=training_df,
            topology=topology_df
        )
        
        print("❌ Invalid training params should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            print("✅ Invalid training params correctly rejected")
            print(f"   Error: {response_data.get('error')}")
            return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False


def test_invalid_simulation_requests():
    """Test various invalid simulation request scenarios"""
    print("Testing invalid simulation requests...")
    
    radp_client = RADPClient()
    
    # Test 1: Missing required fields
    try:
        invalid_simulation = {
            "simulation_time_interval_seconds": 0.01
            # Missing ue_tracks and rf_prediction
        }
        
        response = radp_client.simulation(
            simulation_event=invalid_simulation
        )
        
        print("❌ Missing fields should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            print("✅ Missing required fields correctly rejected")
            print(f"   Error: {response_data.get('error')}")
            return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False
    
    # Test 2: Invalid UE class
    try:
        invalid_ue_class_simulation = {
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
            "rf_prediction": {"model_id": "nonexistent_model"}
        }
        
        response = radp_client.simulation(
            simulation_event=invalid_ue_class_simulation
        )
        
        print("❌ Invalid UE class should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            print("✅ Invalid UE class correctly rejected")
            print(f"   Error: {response_data.get('error')}")
            return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False


def test_invalid_file_uploads():
    """Test file validation with malformed CSV files"""
    print("Testing invalid file uploads...")
    
    radp_client = RADPClient()
    
    # Test 1: CSV with missing required columns
    try:
        # Create invalid training data (missing columns)
        invalid_training_df = pd.DataFrame({
            "cell_id": ["cell_1"],
            "avg_rsrp": [-80]
            # Missing: lon, lat, cell_el_deg
        })
        
        valid_topology_df = pd.DataFrame({
            "cell_lat": [35.690556],
            "cell_lon": [139.691944],
            "cell_id": ["cell_1"], 
            "cell_az_deg": [0],
            "cell_carrier_freq_mhz": [2100]
        })
        
        response = radp_client.train(
            model_id="test_invalid_columns",
            params={"maxiter": 100, "lr": 0.05, "stopping_threshold": 0.0001},
            ue_training_data=invalid_training_df,
            topology=valid_topology_df
        )
        
        print("❌ Invalid CSV columns should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            if "file_validation_error" in response_data.get("error_type", ""):
                print("✅ Invalid CSV columns correctly rejected")
                print(f"   Error: {response_data.get('error')}")
                return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False
    
    # Test 2: Invalid data ranges
    try:
        # Create training data with invalid RSRP values
        invalid_rsrp_df = pd.DataFrame({
            "cell_id": ["cell_1"],
            "avg_rsrp": [50],  # Invalid: positive RSRP
            "lon": [139.699058],
            "lat": [35.644327],
            "cell_el_deg": [0]
        })
        
        response = radp_client.train(
            model_id="test_invalid_rsrp",
            params={"maxiter": 100, "lr": 0.05, "stopping_threshold": 0.0001},
            ue_training_data=invalid_rsrp_df,
            topology=valid_topology_df
        )
        
        print("❌ Invalid RSRP values should have failed but didn't")
        return False
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            response_data = e.response.json()
            print("✅ Invalid RSRP values correctly rejected")
            print(f"   Error: {response_data.get('error')}")
            return True
        print(f"❌ Unexpected error response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False


def unhappy_case__invalid_requests():
    """Main function to run all invalid request tests"""
    print("🧪 Running RADP API Validation Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    try:
        test_results.append(test_invalid_training_requests())
        test_results.append(test_invalid_simulation_requests()) 
        test_results.append(test_invalid_file_uploads())
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False
    
    success_count = sum(test_results)
    total_count = len(test_results)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_count} passed")
    
    if success_count == total_count:
        print("✅ All validation integration tests passed!")
        return True
    else:
        print("❌ Some validation tests failed")
        return False


if __name__ == "__main__":
    success = unhappy_case__invalid_requests()
    sys.exit(0 if success else 1)