## ------ Build stage
# Use the specifidied Python-slim version as the base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Add temporary SCM version for hatchling.build
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

# Setup working directory and copy pyproject files
WORKDIR /app
COPY pyproject.toml README.md /app/

# --frozen: Do not update uv.lock
# --no-install-project: Skip building and installing the project package itself
RUN uv sync --no-dev --no-install-project
# Export requirements for pip
RUN uv export --no-hashes --no-emit-project --format requirements-txt > requirements.txt


## ------ Runtime stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

# Set arguments to be passed from build-args
ARG VERSION

# Main working directory
WORKDIR /app
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install python packages
RUN pip install -r requirements.txt

# Curl installation step for health check
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Source files
COPY ./src /app/src

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV APP_VERSION=$VERSION

# Expose port 8000 for the FastAPI Borehole app
EXPOSE 8000

# Run a health check to ensure the container is healthy
HEALTHCHECK CMD curl --silent --fail http://localhost:8000/health || exit 1

# Command to run the FastAPI Borehole app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
