#!/bin/bash

# subshell
(
    # blocks until server up
    kafka-topics \
        --bootstrap-server 0.0.0.0:9092 \
        --list

    # precreate topics
    IFS=', ' read -r -a topics <<< "$KAFKA_TOPICS";
    for topic in "${topics[@]}"
    do
        kafka-topics \
            --bootstrap-server 0.0.0.0:9092 \
            --create \
            --if-not-exists \
            --replication-factor 1 \
            --partitions 2 \
            --topic $topic
    done
# run in background
) &

# passthru original cmd
exec "$@"
