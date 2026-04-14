## Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

WORKDIR /app
COPY pyproject.toml README.md /app/

# Install all dependencies including deep-learning extras into a virtualenv.
# --no-dev: exclude development dependencies
# --no-install-project: skip building and installing the project package itself
# --compile: generate *.pyc files to lower memory load at startup
RUN uv sync --no-dev --no-install-project --compile --extra deep-learning


## Model download stage
# Downloads only the inference files needed at runtime (training artifacts are excluded).
# AWS credentials are only used during build and are not present in the final image.
FROM amazon/aws-cli AS model-downloader

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION=eu-central-1
ARG BERT_MODEL_S3_BUCKET
ARG BERT_MODEL_S3_KEY_BACKBONE=backbone
ARG BERT_MODEL_S3_KEY_LITHOLOGY_HEAD=lithology_head
ARG BERT_MODEL_S3_KEY_EN_MAIN_HEAD=en_main_head

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Download the shared backbone once, then the small task-specific heads.
# Head directories follow the standard HuggingFace layout (config.json + model.safetensors + tokenizer files).
# If BERT_MODEL_S3_BUCKET is not provided, model directories are left empty and
# models must be supplied at runtime via BERT_MODEL_PATH_* or BERT_MODEL_S3_KEY_* env vars.
RUN if [ -n "${BERT_MODEL_S3_BUCKET}" ]; then \
    aws s3 cp s3://${BERT_MODEL_S3_BUCKET}/${BERT_MODEL_S3_KEY_BACKBONE}/backbone.safetensors /models/backbone/backbone.safetensors \
 && aws s3 cp s3://${BERT_MODEL_S3_BUCKET}/${BERT_MODEL_S3_KEY_LITHOLOGY_HEAD}/ /models/lithology_head/ --recursive --exclude "*" --include "config.json" --include "model.safetensors" --include "tokenizer.json" --include "tokenizer_config.json" --include "special_tokens_map.json" --include "vocab.txt" \
 && aws s3 cp s3://${BERT_MODEL_S3_BUCKET}/${BERT_MODEL_S3_KEY_EN_MAIN_HEAD}/ /models/en_main_head/ --recursive --exclude "*" --include "config.json" --include "model.safetensors" --include "tokenizer.json" --include "tokenizer_config.json" --include "special_tokens_map.json" --include "vocab.txt"; \
  else \
    echo "BERT_MODEL_S3_BUCKET not set — skipping model download." \
 && mkdir -p /models/backbone /models/lithology_head /models/en_main_head; \
  fi


## Runtime stage
FROM python:3.12-slim

ARG VERSION

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Curl installation step for health check
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application source code into the container
COPY ./src /app/src

# Copy baked-in models from the download stage
COPY --from=model-downloader /models /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV APP_VERSION=$VERSION
ENV PATH="/app/.venv/bin:$PATH"

# Point the API at the baked-in backbone and heads. These take priority over S3 download.
ENV BERT_MODEL_PATH_BACKBONE=/app/models/backbone/backbone.safetensors
ENV BERT_MODEL_PATH_LITHOLOGY_HEAD=/app/models/lithology_head
ENV BERT_MODEL_PATH_EN_MAIN_HEAD=/app/models/en_main_head

# Expose port 8000 for the FastAPI Borehole app
EXPOSE 8000

# Run a health check to ensure the container is healthy
HEALTHCHECK CMD curl --silent --fail http://localhost:8000/health || exit 1

# Command to run the FastAPI Borehole app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
