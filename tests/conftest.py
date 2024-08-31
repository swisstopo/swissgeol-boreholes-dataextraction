# tests/conftest.py
import boto3
import pytest
from app.common.config import config
from app.main import app
from fastapi.testclient import TestClient
from moto import mock_aws


@pytest.fixture(scope="session")
def test_client():
    return TestClient(app)


@mock_aws
@pytest.fixture(scope="session")
def s3_client():
    conn = boto3.resource("s3", region_name="eu-west-1")
    # We need to create the bucket since this is all in Moto's 'virtual' AWS account
    conn.create_bucket(Bucket=config.bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
    yield conn
