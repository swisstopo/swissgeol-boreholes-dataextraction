"""Tests for the extract_data endpoint.

To see the objects that are being created before the tests are run, you can look at tests/conftest.py. The
test_client fixture is created by the TestClient(app) call, which creates a test client for the FastAPI app.
The s3_client fixture is created by the mock_aws decorator, which mocks the AWS S3 client using Moto.

"""

from pathlib import Path

import fitz
import pytest
from app.common.aws import load_pdf_from_aws
from app.common.config import config
from app.common.schemas import ExtractDataRequest, FormatTypes
from fastapi.testclient import TestClient

TEST_PDF_KEY = "pdfs/sample.pdf"
TEST_PDF_PATH = Path(__file__).parent.parent / "example" / "example_borehole_profile.pdf"
TEST_PNG_KEY = "pngs/sample-1.png"
TEST_PNG_PATH = Path(__file__).parent.parent / "example" / "sample-1.png"


def get_default_small_coordinate_request():
    """Return a default ExtractDataRequest for coordinates."""
    return ExtractDataRequest(
        filename=TEST_PDF_KEY.split("/")[-1],
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        format=FormatTypes.COORDINATES,
    )


def get_default_coordinate_request():
    """Return a default ExtractDataRequest for coordinates."""
    return ExtractDataRequest(
        filename=TEST_PDF_KEY.split("/")[-1],
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 3000, "y1": 3000},
        format=FormatTypes.COORDINATES,
    )


@pytest.fixture(scope="function")
def upload_test_pdf(s3_client):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=config.test_bucket_name, Key=TEST_PDF_KEY)


@pytest.fixture(scope="function")
def upload_test_png(s3_client, upload_test_pdf):
    """Upload a test PNG file to S3."""
    s3_client.upload_file(Filename=str(TEST_PNG_PATH), Bucket=config.test_bucket_name, Key=TEST_PNG_KEY)


def test_load_pdf_from_aws(upload_test_pdf):
    """Test loading a PDF from mocked AWS S3."""
    pdf_document = load_pdf_from_aws(TEST_PDF_KEY.split("/")[-1])
    assert pdf_document is not None
    assert isinstance(pdf_document, fitz.Document)


def test_extract_coordinate_fail(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = get_default_small_coordinate_request()
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "detail" in json_response


def test_extract_text_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    target_text = (
        "BLS AlpTransit AG Lötschberg-Basislinie Sondierbohrung : SST KB 5 "
        "Bauherrschaft: BLS AlpTransit "
        "Bohrfirma : jet injectobohr AG "
        "Bohrmeister : Dragnic "
        "Ausführungsdatum 2.-3. 9. 1995 "
        "Koordinaten : 615 790 / 157 500 "
        "Kote Bezugspunkt : ~788,6 m ü. M. "
    )

    request = ExtractDataRequest(
        filename=TEST_PDF_KEY.split("/")[-1],
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 1000, "y1": 1000},
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text


def test_extract_text_empty(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    target_text = ""

    request = ExtractDataRequest(
        filename=TEST_PDF_KEY.split("/")[-1],
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text


def test_extract_coordinate_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = get_default_coordinate_request()
    response = test_client.post("/api/V1/extract_data", json=request.model_dump())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "coordinates" in json_response
    assert json_response["coordinates"]["east"] == 615790
    assert json_response["coordinates"]["north"] == 157500
    assert json_response["coordinates"]["projection"] == "LV03"


def test_incomplete_request(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an incomplete request."""
    request = get_default_coordinate_request()
    request_json = request.model_dump()
    del request_json["bbox"]
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "bbox field - Field required"}


def test_page_number_out_of_range(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an out-of-range page number."""
    request = get_default_coordinate_request()
    request_json = request.model_dump()
    request_json["page_number"] = 2
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "page not in document"}

    request_json["page_number"] = 0
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "Page number must be a positive integer"}


def test_invalid_format(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid format."""
    request = get_default_coordinate_request()
    request_json = request.model_dump()
    request_json["format"] = "invalid"
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {
        "detail": "format field - Input should be 'text', 'number', 'coordinates' or 'elevation'"
    }


def test_invalid_bbox(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid bounding box."""
    request = get_default_coordinate_request()
    request_json = request.model_dump()
    request_json["bbox"] = {"x0": 0, "y0": 0, "x1": 100, "y1": -100.0}
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "Bounding box coordinate must be a positive integer"}


def test_invalid_pdf(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid PDF."""
    request = get_default_coordinate_request()
    request_json = request.model_dump()
    request_json["filename"] = "invalid.pdf"
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 404
    assert response.json() == {"detail": "Failed to load PDF document. The filename is not found in the bucket."}
