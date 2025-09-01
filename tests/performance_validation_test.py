#!/usr/bin/env python3
"""
Performance and stress tests for the RADP validation system.
Tests validation performance with large datasets and edge cases.
"""

import sys
import os
import time
import tempfile
import pandas as pd
from typing import Dict, List
import logging

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../radp'))

from api_manager.validators.training_validator import TrainingRequestValidator
from api_manager.validators.simulation_validator import SimulationRequestValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_large_file_validation_performance():
    """Test validation performance with large CSV files"""
    logger.info("Testing large file validation performance...")
    
    validator = TrainingRequestValidator()
    
    # Create large training dataset (10,000 rows)
    large_training_data = []
    for i in range(10000):
        large_training_data.append({
            "cell_id": f"cell_{i % 100}",  # 100 unique cells
            "avg_rsrp": -80 - (i % 50),   # Vary RSRP values
            "lon": 139.699 + (i % 1000) * 0.0001,
            "lat": 35.644 + (i % 1000) * 0.0001,
            "cell_el_deg": i % 15
        })
    
    # Create corresponding topology
    topology_data = []
    for i in range(100):
        topology_data.append({
            "cell_lat": 35.644 + i * 0.001,
            "cell_lon": 139.699 + i * 0.001,
            "cell_id": f"cell_{i}",
            "cell_az_deg": i % 360,
            "cell_carrier_freq_mhz": 2100
        })
    
    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
        df_training = pd.DataFrame(large_training_data)
        df_training.to_csv(ue_file.name, index=False)
        ue_file_path = ue_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
        df_topology = pd.DataFrame(topology_data)
        df_topology.to_csv(topo_file.name, index=False)
        topo_file_path = topo_file.name
    
    try:
        files = {
            "ue_training_data_file_path": ue_file_path,
            "topology_file_path": topo_file_path
        }
        
        # Time the validation
        start_time = time.time()
        validator.validate_training_files(files)
        end_time = time.time()
        
        validation_time = end_time - start_time
        throughput = len(large_training_data) / validation_time
        
        logger.info(f"✅ Large file validation completed")
        logger.info(f"   Dataset size: {len(large_training_data):,} training records, {len(topology_data)} cells")
        logger.info(f"   Validation time: {validation_time:.3f} seconds")
        logger.info(f"   Throughput: {throughput:.0f} records/second")
        
        # Performance threshold: should handle at least 1000 records/second
        if throughput < 1000:
            logger.warning(f"⚠️  Performance below threshold (1000 records/sec): {throughput:.0f}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Large file validation failed: {e}")
        return False
        
    finally:
        os.unlink(ue_file_path)
        os.unlink(topo_file_path)


def test_validation_with_many_errors():
    """Test validation performance with datasets containing many errors"""
    logger.info("Testing validation performance with error-heavy datasets...")
    
    validator = TrainingRequestValidator()
    
    # Create dataset with many validation errors
    error_heavy_data = []
    for i in range(1000):
        error_heavy_data.append({
            "cell_id": f"cell_{i}",
            "avg_rsrp": 50 + i,  # All invalid (positive RSRP)
            "lon": 200 + i,     # All invalid (> 180)
            "lat": 100 + i,     # All invalid (> 90)
            "cell_el_deg": 20 + i  # All invalid (> 15)
        })
    
    # Valid topology for consistency
    topology_data = [{"cell_lat": 35.644, "cell_lon": 139.699, "cell_id": f"cell_{i}", 
                     "cell_az_deg": 0, "cell_carrier_freq_mhz": 2100} for i in range(1000)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
        pd.DataFrame(error_heavy_data).to_csv(ue_file.name, index=False)
        ue_file_path = ue_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
        pd.DataFrame(topology_data).to_csv(topo_file.name, index=False)
        topo_file_path = topo_file.name
    
    try:
        files = {
            "ue_training_data_file_path": ue_file_path,
            "topology_file_path": topo_file_path
        }
        
        start_time = time.time()
        
        try:
            validator.validate_training_files(files)
            logger.error("❌ Error-heavy dataset should have failed validation")
            return False
        except FileValidationException as e:
            end_time = time.time()
            
            validation_time = end_time - start_time
            
            # Verify multiple errors were caught
            error_count = len(e.validation_errors) if hasattr(e, 'validation_errors') and e.validation_errors else 0
            
            logger.info(f"✅ Error-heavy validation completed")
            logger.info(f"   Dataset size: {len(error_heavy_data):,} records with errors")
            logger.info(f"   Validation time: {validation_time:.3f} seconds")
            logger.info(f"   Errors detected: {error_count}")
            
            # Should complete within reasonable time even with many errors
            if validation_time > 5.0:
                logger.warning(f"⚠️  Validation took too long with errors: {validation_time:.3f}s")
                return False
            
            return True
        
    except Exception as e:
        logger.error(f"❌ Error-heavy validation test failed: {e}")
        return False
        
    finally:
        os.unlink(ue_file_path)
        os.unlink(topo_file_path)


def test_concurrent_validation_requests():
    """Test validation system under concurrent load"""
    logger.info("Testing concurrent validation requests...")
    
    import threading
    import queue
    
    results = queue.Queue()
    num_threads = 10
    requests_per_thread = 5
    
    def worker():
        validator = TrainingRequestValidator()
        
        for i in range(requests_per_thread):
            try:
                # Create different request variations
                request = {
                    "model_id": f"concurrent_test_model_{threading.current_thread().ident}_{i}",
                    "model_update": False,
                    "params": {
                        "maxiter": 100 + i,
                        "lr": 0.01 + i * 0.001,
                        "stopping_threshold": 0.0001
                    }
                }
                
                # Mock the file system check
                import unittest.mock
                with unittest.mock.patch.object(validator, '_validate_model_id_availability'):
                    validator.validate(request)
                
                results.put(("success", threading.current_thread().ident, i))
                
            except Exception as e:
                results.put(("error", threading.current_thread().ident, i, str(e)))
    
    # Start concurrent validation
    start_time = time.time()
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    # Analyze results
    total_requests = num_threads * requests_per_thread
    successful = 0
    failed = 0
    
    while not results.empty():
        result = results.get()
        if result[0] == "success":
            successful += 1
        else:
            failed += 1
            logger.error(f"Concurrent validation failed: {result[3]}")
    
    concurrent_time = end_time - start_time
    throughput = total_requests / concurrent_time
    
    logger.info(f"✅ Concurrent validation completed")
    logger.info(f"   Total requests: {total_requests} ({num_threads} threads × {requests_per_thread} requests)")
    logger.info(f"   Successful: {successful}, Failed: {failed}")
    logger.info(f"   Total time: {concurrent_time:.3f} seconds")
    logger.info(f"   Throughput: {throughput:.1f} validations/second")
    
    success_rate = successful / total_requests
    if success_rate < 1.0:
        logger.error(f"❌ Success rate too low: {success_rate:.2%}")
        return False
    
    return True


def test_memory_usage_with_large_datasets():
    """Test memory usage during validation of large datasets"""
    logger.info("Testing memory usage with large datasets...")
    
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Get baseline memory usage
    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    validator = SimulationRequestValidator()
    
    # Create large simulation request with many UEs
    large_simulation = {
        "simulation_time_interval_seconds": 0.01,
        "ue_tracks": {
            "ue_tracks_generation": {
                "ue_class_distribution": {
                    "pedestrian": {"count": 5000, "velocity": 1.0, "velocity_variance": 0.5},
                    "car": {"count": 3000, "velocity": 15.0, "velocity_variance": 2.0},
                    "cyclist": {"count": 2000, "velocity": 5.0, "velocity_variance": 1.0}
                },
                "lat_lon_boundaries": {
                    "min_lat": 35.0, "max_lat": 36.0,
                    "min_lon": 139.0, "max_lon": 140.0
                },
                "gauss_markov_params": {
                    "alpha": 0.5, "variance": 0.8, "rng_seed": 42
                }
            }
        },
        "rf_prediction": {"model_id": "memory_test_model"}
    }
    
    try:
        # Mock model existence check
        import unittest.mock
        with unittest.mock.patch.object(validator, '_validate_model_exists'):
            validator.validate(large_simulation)
        
        # Check memory usage after validation
        gc.collect()
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        logger.info(f"✅ Memory usage test completed")
        logger.info(f"   UE count: {5000 + 3000 + 2000:,}")
        logger.info(f"   Baseline memory: {baseline_memory:.1f} MB")
        logger.info(f"   Peak memory: {peak_memory:.1f} MB")
        logger.info(f"   Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (< 100MB for validation)
        if memory_increase > 100:
            logger.warning(f"⚠️  High memory usage: {memory_increase:.1f} MB")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory usage test failed: {e}")
        return False


def main():
    """Run all performance tests"""
    logger.info("🏃‍♂️ RADP Validation Performance Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Large File Validation", test_large_file_validation_performance),
        ("Error-Heavy Dataset", test_validation_with_many_errors),
        ("Concurrent Validation", test_concurrent_validation_requests),
        ("Memory Usage", test_memory_usage_with_large_datasets)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        start_time = time.time()
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.exception(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        duration = time.time() - start_time
        logger.info(f"{test_name} completed in {duration:.3f}s")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All performance tests passed!")
        return 0
    else:
        logger.info("❌ Some performance tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())