"""Pytest configuration file."""

import boto3
import pytest
from app.common.config import config
from app.main import app
from botocore.errorfactory import ClientError
from fastapi.testclient import TestClient
from moto import mock_aws


@pytest.fixture(scope="session")
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@mock_aws
@pytest.fixture(scope="session")
def s3_client():
    """Mocked S3 client."""
    conn = boto3.client("s3", region_name="eu-west-1")
    try:
        # We need to create the bucket since this is all in Moto's 'virtual' AWS account
        conn.create_bucket(Bucket=config.bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
    except ClientError:
        # Bucket already exists, that's fine
        print("Bucket already exists")
    yield conn
