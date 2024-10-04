"""Pytest configuration file."""

import boto3
import pytest
from app.common.aws import get_s3_client
from app.common.config import config
from app.main import app
from fastapi.testclient import TestClient
from moto import mock_aws


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
        conn.create_bucket(
            Bucket=config.test_bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
        )
        # Monkeypatch boto3.client to return the mocked S3 client
        monkeypatch.setattr("boto3.client", lambda *args, **kwargs: conn)

        # Mock s3_client.get_object to use config.test_bucket_name
        original_get_object = conn.get_object

        def mock_get_object(Bucket, Key, *args, **kwargs):
            if Bucket == config.bucket_name:
                Bucket = config.test_bucket_name  # Replace with test bucket name
            return original_get_object(*args, Bucket=Bucket, Key=Key, **kwargs)

        # Mock s3_client.upload_file to use config.test_bucket_name
        original_upload_file = conn.upload_file

        def mock_upload_file(Filename, Bucket, Key, *args, **kwargs):
            # Replace the Bucket with the test bucket name if necessary
            if Bucket == config.bucket_name:
                Bucket = config.test_bucket_name  # Replace with test bucket name

            # Call the original upload_file with the modified or original arguments
            return original_upload_file(*args, Filename=Filename, Bucket=Bucket, Key=Key, **kwargs)

        # Mock the list_buckets method with the test bucket name
        original_list_buckets = conn.list_buckets

        def mock_list_buckets(*args, **kwargs):
            response = original_list_buckets(*args, **kwargs)
            response["Buckets"] = [{"Name": config.test_bucket_name}]
            return response

        monkeypatch.setattr(conn, "get_object", mock_get_object)
        monkeypatch.setattr(conn, "upload_file", mock_upload_file)
        monkeypatch.setattr(conn, "list_buckets", mock_list_buckets)

        # Patch the s3_client in the aws module to use the mock
        monkeypatch.setattr("app.common.aws._s3_client", conn)

        yield conn


def test_s3_functionality(s3_client):
    """Test the S3 functionality."""
    # This now runs in the mock_s3 context
    s3_client_instance = get_s3_client()
    response = s3_client_instance.list_buckets()
    assert "Buckets" in response
