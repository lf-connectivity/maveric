# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

from confluent_kafka import Consumer

from orchestration.config.kafka import kafka_consumer_config
from orchestration.orchestrator import Orchestrator
from radp.common import constants
from radp.utility.kafka_utils import safe_subscribe

logger = logging.getLogger(__name__)
ORCHESTRATION_CONSUMER_TOPICS = [
    constants.KAFKA_JOBS_TOPIC_NAME,
    constants.KAFKA_OUTPUTS_TOPIC_NAME,
]


class OrchestrationConsumer:
    """Service which consumes orchestration jobs and output events"""

    def __init__(self):
        """Initialize consumer and driver resources"""
        self.consumer = Consumer(kafka_consumer_config)
        self.orchestrator = Orchestrator()

        # subscribe to topics
        safe_subscribe(consumer=self.consumer, topics=ORCHESTRATION_CONSUMER_TOPICS)
        logger.info(f"Subscribed to topics: {ORCHESTRATION_CONSUMER_TOPICS}")

    def consume(self):
        """Method to consume orchestration jobs and output events"""

        logger.info("Starting orchestration consume loop")
        try:
            # Poll for new messages from Kafka
            while True:
                message = self.consumer.poll(constants.KAFKA_CONSUMER_POLL_INTERVAL)

                if message is None:
                    # Initial message consumption may take up to `session.timeout.ms` for
                    # the consumer group to rebalance and start consuming
                    logger.debug("Waiting...")
                    continue
                if message.error():
                    logger.exception(f"Error consuming from {constants.KAFKA_JOBS_TOPIC_NAME} topic: {message.error()}")
                    continue

                # Extract the (optional) key and value, and print.
                logger.debug(f"Consumed message value = {message.value().decode('utf-8')}")

                # pull event object from message
                event = json.loads(message.value().decode("utf-8"))

                # wrap event handling to stay running if one event fails
                try:
                    # check which topic message is from
                    if message.topic() == constants.KAFKA_JOBS_TOPIC_NAME:
                        if event[constants.KAFKA_JOB_TYPE] != constants.JOB_TYPE_ORCHESTRATION:
                            # skip non-orchestration jobs
                            logger.debug(f"Consumed non-orchestration job: {event}... skipping")
                        else:
                            # handle orchestration job
                            logger.info(f"Consumed orchestration job: {event}... handling")
                            self.orchestrator.handle_orchestration_job(event)
                        continue
                    else:
                        # handle output event
                        logger.info(f"Consumed output event: {event}... handling")
                        self.orchestrator.handle_output_event(event)
                except Exception:
                    logger.exception(f"Error occurred while handling job: {event}")
        except KeyboardInterrupt:
            pass
        finally:
            # Leave group and commit final offsets
            self.consumer.close()
            logger.info("Closing consume loop")
