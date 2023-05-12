FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

RUN apt update && \
    apt install -y python3 python3-pip

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH="$PYTHONPATH:/app/"

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN rm requirements*.txt

COPY main.py main.py
