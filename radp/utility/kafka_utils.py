# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import time
import uuid
from typing import Dict, List

import confluent_kafka as kafka

from radp.common.constants import JOB_ID

logger = logging.getLogger(__name__)


def produce_object_to_kafka_topic(
    producer: kafka.Producer,
    topic: str,
    value: Dict,
    kafka_poll_interval=10,
):
    """Produce a message to a kafka topic"""

    job_id = str(uuid.uuid4())
    value[JOB_ID] = job_id

    # dump object to json string
    job_json_string = json.dumps(value)

    try:
        producer.produce(
            topic,
            value=job_json_string,
        )

        # poll event to flush produce
        producer.poll(kafka_poll_interval)

        return job_id
    except Exception as e:
        logger.exception(f"Unexpected exception occurred calling kafka produce: {e}")
        raise e


def safe_subscribe(consumer: kafka.Consumer, topics: List[str]):
    """Safely subscribe to a list of topics once all exist"""
    MAX_ATTEMPTS = 100
    SLEEP_INTERVAL = 3
    topics_found = [topic_metadata for topic_metadata in consumer.list_topics().topics]
    current_attempt = 0

    # sleep until all topics exist
    while not all([topic_name in topics_found for topic_name in topics]):
        current_attempt += 1
        if current_attempt >= MAX_ATTEMPTS:
            logger.exception(f"Timed out while attempting to subscribe consumer '{consumer}' to topics: {topics}")
            raise Exception(f"Timed out while attempting to subscribe consumer '{consumer}' to topics: {topics}")

        time.sleep(SLEEP_INTERVAL)
        topics_found = [topic_metadata for topic_metadata in consumer.list_topics().topics]

    # all topics exist, subscribe to them
    try:
        consumer.subscribe(topics)
    except Exception as e:
        logger.exception(f"Exception occurred while attempting to subscribe to topics: {e}")
        raise e
