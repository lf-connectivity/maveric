FROM python:3.8-slim

# pip cache dir is kind of useless in Docker containers
#   since the services are isolated from each other
# it also takes up a lot of disk space
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# confluent-kafka-python does not yet have prebuilt
#   wheel packages for arm64 for pip to install
#   https://github.com/confluentinc/confluent-kafka-python/issues/1439
# confluent also does not yet have prebuilt
#   librdkafka packages for Debian for arm64 either
# but we can build librdkafka from source
#   and let the pip package build itself against it
SHELL [ "bash", "-c" ]
RUN if [[ $(dpkg --print-architecture) != amd64 ]]; then \
    apt update && \
    apt install -y --no-install-recommends git libssl-dev gcc g++ make && \
    git clone https://github.com/edenhill/librdkafka.git librdkafka && \
    cd librdkafka && \
    git checkout tags/v2.0.2 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf librdkafka && \
    touch /etc/ld.so.conf.d/librdkafka.conf && \
    echo /usr/local/lib >> /etc/ld.so.conf.d/librdkafka.conf && \
    ldconfig \
    ; fi

RUN pip3 install --upgrade pip

COPY client/requirements.txt requirements.client.txt
RUN pip3 install -r requirements.client.txt

COPY common/requirements.txt requirements.common.txt
RUN pip3 install -r requirements.common.txt

COPY digital_twin/requirements.txt requirements.digital_twin.txt
RUN pip3 install -r requirements.digital_twin.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY utility/requirements.txt requirements.utility.txt
RUN pip3 install -r requirements.utility.txt

RUN rm requirements*.txt
