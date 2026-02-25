import json
import os

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


def _ensure_server_available() -> None:
    try:
        response = requests.post(f"{BASE_URL}/health", timeout=2)
    except requests.RequestException:
        pytest.skip("API server not running on http://127.0.0.1:8000")
    if response.status_code >= 400:
        pytest.skip("API server not ready for session flow test")

def test_session_flow():
    _ensure_server_available()
    print("Testing Session Flow...")
    
    # 1. Create Session
    print("1. Creating Session...")
    resp = requests.post(f"{BASE_URL}/sessions", json={"title": "Test Session"})
    assert resp.status_code == 200
    
    session_data = resp.json()["session"]
    session_id = session_data["id"]
    print(f"Session Created: {session_id}")
    
    # 2. List Sessions
    print("2. Listing Sessions...")
    resp = requests.get(f"{BASE_URL}/sessions")
    sessions = resp.json()["sessions"]
    found = any(s["id"] == session_id for s in sessions)
    print(f"Session found in list: {found}")
    
    # 3. Upload File (Mocking a file upload)
    # Create a dummy file
    try:
        with open("test_doc.txt", "w") as f:
            f.write("Apple produces iPhone. Tim Cook is the CEO of Apple.")

        print("3. Uploading File to Session...")
        with open("test_doc.txt", "rb") as f:
            files = {"files": ("test_doc.txt", f)}
            resp = requests.post(
                f"{BASE_URL}/ingest", files=files, params={"session_id": session_id}
            )

        print(f"Upload Status: {resp.status_code}")
        print(resp.json())
    
        # 4. Query Session
        print("4. Querying Session...")
        query_payload = {
            "question": "Who is the CEO of Apple?",
            "session_id": session_id,
            "memory_mode": "session",
        }
        resp = requests.post(f"{BASE_URL}/query", json=query_payload)
        print(f"Query Result: {resp.json()}")
        
        # 5. Check History
        print("5. Checking History...")
        resp = requests.get(f"{BASE_URL}/sessions/{session_id}/history")
        history = resp.json()["history"]
        print(f"History length: {len(history)}")
        print(history)
    finally:
        os.remove("test_doc.txt")

if __name__ == "__main__":
    test_session_flow()
