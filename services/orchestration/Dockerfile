FROM radp AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH="$PYTHONPATH:/app/"

RUN pip3 install --upgrade pip

COPY services/orchestration/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN rm requirements*.txt

FROM base AS dev

COPY services/requirements-dev.txt requirements-dev.txt
RUN pip3 install -r requirements-dev.txt

RUN rm requirements*.txt

FROM base AS prod

COPY radp radp
COPY services/orchestration orchestration
