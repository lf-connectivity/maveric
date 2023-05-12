FROM confluentinc/cp-kafka:7.3.1

COPY ./services/kafka/entrypoint.sh /entrypoint.sh

CMD ["/etc/confluent/docker/run"]

ENTRYPOINT [ "/entrypoint.sh" ]
