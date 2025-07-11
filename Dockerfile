# Use an official Python runtime as a parent image - Use the latest slim version as the base image
FROM python:3.12-slim

# Set arguments to be passed from build-args
ARG VERSION

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml and any other configuration (optional)
COPY pyproject.toml /app/

# Install pip-tools and use it to resolve dependencies
RUN pip install --no-cache-dir pip-tools \
    && pip-compile --generate-hashes \
    && pip-sync

# Curl installation step for health check
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application source code into the container
COPY ./src /app/src
COPY ./config /app/config

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
