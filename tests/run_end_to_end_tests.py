# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import time
from datetime import datetime

from happy_case_tests.happy_rf_prediction import happy_case__rf_prediction
from happy_case_tests.happy_ue_track_gen_rf_prediction import (
    happy_case__ue_track_gen_rf_prediction,
)

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


# add new integration tests to this list
tests = [happy_case__rf_prediction, happy_case__ue_track_gen_rf_prediction]

delay_for_service_startup = "--delayed-start" in sys.argv

if delay_for_service_startup:
    # wait 60 seconds to allow for service to bootup
    # TODO: replace this workaround with a solution that checks startup health of RADP service
    # startup health can likely be checked by calling kafka and listing groups
    time.sleep(60)

# test metrics setup
all_passed = True
run = 0
passed = 0
failed = 0
start_time = datetime.now()

# run each test, log results
for test in tests:
    logger.info("--------------------------------------------------------")
    test_passed = True
    test_start_time = datetime.now()
    test_name = test.__name__

    try:
        logger.info(f"Running test: {test_name}")
        test()
    except Exception as e:
        test_passed = False
        all_passed = False
        failed += 1
        logger.exception(f"Encountered exception during execution of test: {test_name}")
        logger.exception(e)
    else:
        passed += 1
    run += 1

    logger.info("")
    logger.info(f"{'Result:' : <20}{'SUCCESS' if test_passed else 'FAILED' : >36}")
    logger.info(
        f"{'Test duration:' : <20}{str(datetime.now() - test_start_time) : >36}"
    )

# report results
logger.info("--------------------------------------------------------")
logger.info("--------------------------------------------------------")
logger.info(
    f"{'TEST RESULT SUMMARY:' : <24}{'SUCCESS' if all_passed else 'FAILURE' : >32}"
)
logger.info(f"{'TESTS RUN:' : <20}{run : >36}")
logger.info(f"{'TESTS PASSED:' : <20}{passed : >36}")
logger.info(f"{'TESTS FAILED:' : <20}{failed : >36}")
logger.info(f"{'TESTS DURATION:' : <20}{str(datetime.now() - start_time) : >36}")
logger.info("--------------------------------------------------------")

message = (
    "SUCCESS" if all_passed else "TESTS FAILED: Check logs above to see failure(s)"
)
logger.info(message)
logger.info("--------------------------------------------------------")

# set exit status for pre-commit pass/failure
sys.exit(0 if all_passed else 1)
