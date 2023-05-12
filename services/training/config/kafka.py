# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Document these values
kafka_consumer_config = {
    "bootstrap.servers": "kafka:9092",
    "group.id": "training",  # ID to help us distinguish between different consumers on a topic
    "auto.offset.reset": "smallest",
    "max.poll.interval.ms": 43200000,  # 12 hour timeout on training polling
}
