# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import signal
import sys

from rf_prediction.rf_prediction_consumer import RFPredictionConsumer


# define a sigterm handler to allow docker to gracefully exit
def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)

    inference_service = RFPredictionConsumer()
    inference_service.consume_from_jobs()
