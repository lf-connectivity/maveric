# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import unittest
from typing import List, Tuple

import coverage

# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.INFO)

# Create a Formatter for formatting the log messages
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:  %(message)s")
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)
logger.propagate = False

RADP_RELATIVE_PATH = "../radp"
SERVICES_RELATIVE_PATH = "../services"


def get_top_level(relative_path):
    """Helper method to get root packages from relative path"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


def get_package_roots(top_level_path):
    """Helper method to get root packages from relative path"""
    root_paths = [
        os.path.join(top_level_path, path) for path in os.listdir(top_level_path)
    ]
    return [path for path in root_paths if os.path.isdir(path)]


def run_tests(source_paths) -> Tuple[List[unittest.TestResult], int]:
    """Run all tests across source paths"""
    loader = unittest.TestLoader()

    test_results = []
    total_tests = 0

    for path in source_paths:
        test_suite = loader.discover(
            start_dir=path, top_level_dir=path, pattern="test*.py"
        )
        total_tests += test_suite.countTestCases()

        testRunner = unittest.runner.TextTestRunner()
        test_result = testRunner.run(test_suite)

        test_results.append(test_result)
    return test_results, total_tests


# get the top-level paths
radp_top_level_path = get_top_level(RADP_RELATIVE_PATH)
services_top_level_path = get_top_level(SERVICES_RELATIVE_PATH)

# insert library paths into Python path for testing
library_package_paths = get_package_roots(radp_top_level_path)
for path in library_package_paths:
    sys.path.insert(0, path)

# check if coverage enabled
run_coverage = "-c" in sys.argv

if run_coverage:
    # start coverage tracking
    cov = coverage.Coverage()
    cov.start()

    # run tests
    test_results, total_tests = run_tests(
        [radp_top_level_path, services_top_level_path]
    )

    # stop coverage tracking
    cov.stop()
else:
    # run tests
    test_results, total_tests = run_tests(
        [radp_top_level_path, services_top_level_path]
    )

success = True
failed: List[Tuple[unittest.TestResult, str]] = []
errors: List[Tuple[unittest.TestResult, str]] = []

# gather results
for test_result in test_results:
    if not test_result.wasSuccessful():
        success = False
    errors.extend(test_result.errors)
    failed.extend(test_result.failures)

# report coverage if enabled
if run_coverage:
    logger.info(cov.report())

if failed:
    logger.info("--------------------------------------------------------")
    logger.info("Tests Failed:")
    for test_result, failure_msg in failed:
        logger.exception(f"FAILURE: {test_result}")
        logger.info(failure_msg)

if errors:
    logger.info("--------------------------------------------------------")
    logger.info("Test Errors:")
    for test_result, err_msg in errors:
        logger.info("--------------------------------------------------------")
        logger.exception(test_result)
        logger.exception(err_msg)
        logger.info(err_msg)

fail_count = len(failed)
error_count = len(errors)


# report results
logger.info("--------------------------------------------------------")
logger.info(f"{'TEST RESULT:' : <20}{'SUCCESS' if success else 'FAILURE' : >36}")
logger.info(f"{'TOTAL TESTS:' : <20}{total_tests : >36}")
logger.info(f"{'PASSED:' : <20}{total_tests - fail_count - error_count : >36}")
logger.info(f"{'FAILED:' : <20}{fail_count : >36}")
logger.info(f"{'ERRORS:' : <20}{error_count : >36}")
logger.info("--------------------------------------------------------")

message = (
    "SUCCESS" if success else "TESTS FAILED: Check test reports above to see failure(s)"
)
logger.info(message)
logger.info("--------------------------------------------------------")

# set exit status for pre-commit pass/failure
sys.exit(0 if success else 1)
