FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=configs/default.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY configs ./configs
COPY data/raw ./data/raw
COPY data/processed/.gitkeep ./data/processed/.gitkeep
COPY data/eval/.gitkeep ./data/eval/.gitkeep

RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main --extras tracking --no-interaction --no-ansi \
    && chmod +x scripts/docker_entrypoint.sh

EXPOSE 8501

CMD ["scripts/docker_entrypoint.sh"]
