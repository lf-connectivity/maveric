#!/usr/bin/env python3
"""Fast validation test runner for RADP API validation components.

This runner intentionally avoids Docker and notebooks.
"""

import logging
import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "services"))
sys.path.insert(0, os.path.join(REPO_ROOT, "radp"))
sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False


def run_validation_unit_tests() -> unittest.TestResult:
    validation_test_dir = os.path.join(
        REPO_ROOT, "services", "api_manager", "tests", "validators"
    )
    suite = unittest.TestLoader().discover(
        start_dir=validation_test_dir,
        pattern="test_*.py",
    )
    logger.info("Running %s validation unit tests", suite.countTestCases())
    return unittest.TextTestRunner(verbosity=2).run(suite)


def test_error_response_formatting() -> bool:
    from api_manager.exceptions.validation_exception import (
        FileValidationException,
        ValidationException,
    )

    validation_error = ValidationException(
        "Validation failed",
        validation_errors=[{"field": "params.lr", "error": "must be valid"}],
    ).to_dict()

    file_error = FileValidationException(
        "File validation failed",
        filename="ue_data.csv",
        validation_errors=[{"field": "lat", "error": "must be valid"}],
    ).to_dict()

    return (
        validation_error["status_code"] == 400
        and validation_error["error_type"] == "validation_error"
        and validation_error["validation_errors"][0]["field"] == "params.lr"
        and file_error["error_type"] == "file_validation_error"
        and file_error["filename"] == "ue_data.csv"
    )


def main() -> int:
    unit_result = run_validation_unit_tests()
    formatting_ok = test_error_response_formatting()
    if not formatting_ok:
        logger.error("Validation exception response formatting failed")

    success = unit_result.wasSuccessful() and formatting_ok
    logger.info("Validation test result: %s", "SUCCESS" if success else "FAILURE")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
