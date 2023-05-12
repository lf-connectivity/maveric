# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging

from confluent_kafka import Consumer

from radp.common import constants
from radp.utility.kafka_utils import safe_subscribe
from training.config.kafka import kafka_consumer_config
from training.training_driver import TrainingDriver

logger = logging.getLogger(__name__)

TRAINING_CONSUMER_TOPICS = [constants.KAFKA_JOBS_TOPIC_NAME]


class TrainingConsumer:
    """Service which consumes training jobs to run ML training"""

    def __init__(self):
        """Initialize consumer and driver resources"""
        self.consumer = Consumer(kafka_consumer_config)
        self.training_driver = TrainingDriver()

        # subscribe to topics
        safe_subscribe(consumer=self.consumer, topics=TRAINING_CONSUMER_TOPICS)
        logger.info(f"Subscribed to topics: {TRAINING_CONSUMER_TOPICS}")

    def consume_from_jobs(self):
        """Method to consume and handle training jobs topic"""

        logger.info("Starting consume jobs loop")
        try:
            # Poll for new messages from Kafka and print them.
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
                job_data = json.loads(message.value().decode("utf-8"))

                # ignore non-training related jobs
                if job_data[constants.KAFKA_JOB_TYPE] != constants.JOB_TYPE_TRAINING:
                    continue

                # execute training
                try:
                    self.training_driver.handle_training_job(job_data)
                    logger.info(f"Successfully trained model {job_data[constants.MODEL_ID]}")
                except Exception as e:
                    logger.exception(f"Exception occurred while handling training job: {job_data}\n{e}")
        except KeyboardInterrupt:
            pass
        finally:
            # Leave group and commit final offsets
            self.consumer.close()
            logger.info("Closing consume loop")
