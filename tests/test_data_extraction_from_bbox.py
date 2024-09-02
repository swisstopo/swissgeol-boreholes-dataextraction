"""Tests for the extract_data endpoint.

To see the objects that are being created before the tests are run, you can look at tests/conftest.py. The
test_client fixture is created by the TestClient(app) call, which creates a test client for the FastAPI app.
The s3_client fixture is created by the mock_aws decorator, which mocks the AWS S3 client using Moto.

"""

from pathlib import Path

import pytest
from app.common.config import config
from app.common.schemas import ExtractDataRequest
from fastapi.testclient import TestClient

TEST_PDF_KEY = "/pdfs/geoquat/train/sample.pdf"
TEST_PDF_PATH = Path(__file__).parent.parent / "example" / "example_borehole_profile.pdf"
TEST_PNG_KEY = "/pngs/geoquat/train/sample-1.png"
TEST_PNG_PATH = Path(__file__).parent.parent / "example" / "sample-1.png"


@pytest.fixture(scope="module")
def upload_test_pdf(s3_client):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=config.bucket_name, Key=TEST_PDF_KEY)


@pytest.fixture(scope="module")
def upload_test_png(s3_client, upload_test_pdf):
    """Upload a test PNG file to S3."""
    s3_client.upload_file(Filename=str(TEST_PNG_PATH), Bucket=config.bucket_name, Key=TEST_PNG_KEY)


def test_extract_coordinate_fail(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = ExtractDataRequest(
        filename=TEST_PDF_KEY,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        format="coordinates",
    )
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "detail" in json_response


def test_extract_text_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    target_text = """BLS AlpTransit AG Lötschberg-Basislinie Sondierbohrung : SST KB 5
    Bauherrschaft: BLS AlpTransit
    Bohrfirma : jet injectobohr AG
    Bohrmeister : Dragnic
    Ausführungsdatum 2.-3. 9. 1995
    Koordinaten : 615 790 / 157 500
    Kote Bezugspunkt : ~788,6 m ü. M. """

    request = ExtractDataRequest(
        filename=TEST_PDF_KEY,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 1000, "y1": 1000},
        format="text",
    )
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text


def test_extract_coordinate_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = ExtractDataRequest(
        filename=TEST_PDF_KEY,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 3000, "y1": 3000},
        format="coordinates",
    )
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "coordinates" in json_response
    assert json_response["coordinates"]["east"] == 615790
    assert json_response["coordinates"]["north"] == 157500
    assert json_response["coordinates"]["page"] == 1
    assert json_response["coordinates"]["spacial_reference_system"] == "LV95"
