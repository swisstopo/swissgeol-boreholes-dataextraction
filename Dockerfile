# Use an official Python runtime as a parent image - Use the latest slim version as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml and any other configuration (optional)
COPY pyproject.toml /app/

# Install pip-tools and use it to resolve dependencies
RUN pip install --no-cache-dir pip-tools \
    && pip-compile --generate-hashes \
    && pip-sync

# Copy the rest of the application source code into the container
COPY ./src /app
COPY ./config /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV COMPUTING_ENVIRONMENT="API"

# Expose port 8000 for the FastAPI Borehole app
EXPOSE 8000

# Command to run the FastAPI Borehole app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
