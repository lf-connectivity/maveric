# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import signal
import sys

from ue_tracks_generation.ue_tracks_generation_consumer import UETracksGenerationConsumer


# define a sigterm handler to allow docker to gracefully exit
def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)

    ue_tracks_generation_service = UETracksGenerationConsumer()
    ue_tracks_generation_service.consume_from_jobs()
