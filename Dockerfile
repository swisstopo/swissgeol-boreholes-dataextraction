## ------ Build stage
# Use the specified Python-slim version as the base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Add temporary SCM version for hatchling.build
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

# Setup working directory and copy pyproject files
WORKDIR /app
COPY pyproject.toml README.md /app/

# Install all dependencies including deep-learning extras into a virtualenv.
# --no-dev: exclude development dependencies
# --no-install-project: skip building and installing the project package itself
# --compile: generate *.pyc files to lower memory load at startup
RUN uv sync --no-dev --no-install-project --compile --extra deep-learning


## ------ Runtime stage
FROM python:3.12-slim

ARG VERSION

# Main working directory
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Curl installation step for health check
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Source files and the three inference models (committed to the repository)
COPY ./src /app/src
COPY ./models/backbone /app/models/backbone
COPY ./models/en_main_head /app/models/en_main_head
COPY ./models/lithology_head /app/models/lithology_head

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV APP_VERSION=$VERSION
ENV PATH="/app/.venv/bin:$PATH"

# Expose port 8000 for the FastAPI Borehole app
EXPOSE 8000

# Run a health check to ensure the container is healthy
HEALTHCHECK CMD curl --silent --fail http://localhost:8000/health || exit 1

# Command to run the FastAPI Borehole app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
