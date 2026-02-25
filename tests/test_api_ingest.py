import os
import tempfile

import pytest
import requests


BASE_URL = "http://127.0.0.1:8000"


def _ensure_server_available() -> None:
    try:
        response = requests.post(f"{BASE_URL}/health", timeout=2)
    except requests.RequestException:
        pytest.skip("API server not running on http://127.0.0.1:8000")
    if response.status_code >= 400:
        pytest.skip("API server not ready for ingest tests")


def test_api_ingest():
    _ensure_server_available()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
        temp_file.write(b"hello")
        file_path = temp_file.name
    try:
        with open(file_path, "rb") as f:
            files = [("files", (os.path.basename(file_path), f, "text/plain"))]
            response = requests.post(f"{BASE_URL}/ingest", files=files, timeout=10)
        assert response.status_code < 500
    finally:
        os.remove(file_path)
