"""Pytest configuration file."""

import boto3
import pytest
from app.common.aws import get_s3_client
from app.common.config import config
from app.main import app
from fastapi.testclient import TestClient
from moto import mock_aws

TEST_BUCKET_NAME = "test-bucket"


@pytest.fixture(autouse=True, scope="session")
def override_bucket_name():
    """Automatically override the bucket name before the test session."""
    config.bucket_name = TEST_BUCKET_NAME


@pytest.fixture(scope="session")
def test_bucket_name():
    """Returns the name of the test bucket."""
    return TEST_BUCKET_NAME


@pytest.fixture(scope="function")
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="function")
def s3_client(monkeypatch):
    """Mocked S3 client."""
    with mock_aws():
        # Create the mocked S3 client
        conn = boto3.client("s3", region_name="eu-central-1")
        # We need to create the bucket since this is all in Moto's 'virtual' AWS account
        conn.create_bucket(Bucket=config.bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        # Monkeypatch boto3.client to return the mocked S3 client
        monkeypatch.setattr("boto3.client", lambda *args, **kwargs: conn)

        # Patch the s3_client in the aws module to use the mock
        monkeypatch.setattr("app.common.aws._s3_client", conn)

        yield conn


def test_s3_functionality(s3_client):
    """Test the S3 functionality."""
    # This now runs in the mock_s3 context
    s3_client_instance = get_s3_client()
    response = s3_client_instance.list_buckets()
    assert "Buckets" in response
