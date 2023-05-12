# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import socket

kafka_producer_config = {
    "bootstrap.servers": "kafka:9092",
    "client.id": socket.gethostname(),
    "retries": 5,
    "retry.backoff.ms": 200,
}
