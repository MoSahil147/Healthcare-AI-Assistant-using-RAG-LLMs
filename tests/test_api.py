import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# patch ingest so we don't hit the HF API during test import
with patch("app.embeddings.ingest_documents", return_value=10):
    from app.main import app

client = TestClient(app, raise_server_exceptions=True)

MOCK_RAG = {
    "answer": "Yes, patients can request medication refills through telehealth.",
    "sources": [{"document": "telehealth_guidelines.txt", "chunk": "Medication refill requests..."}],
    "confidence": "high",
}

MOCK_FALLBACK = {
    "answer": "I could not find this information in the provided documents.",
    "sources": [],
    "confidence": "none",
}

MOCK_APPOINTMENT = {
    "answer": "Available slots for Cardiology around Monday, June 02: Monday 9:00 AM.",
    "sources": [],
    "confidence": "high",
    "tool_used": "check_available_slots",
    "tool_output": {
        "department": "Cardiology",
        "requested_date": "Monday, June 02",
        "available_slots": ["Monday 9:00 AM"],
        "note": "These are simulated slots. Please call 1800-123-4567 to confirm.",
    },
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest():
    with patch("app.main.ingest_documents", return_value=42):
        response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["chunks_stored"] == 42


def test_ask_rag():
    with patch("app.main.route", return_value=MOCK_RAG):
        response = client.post("/ask", json={"question": "Can a patient refill meds via telehealth?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body and "sources" in body and "confidence" in body


def test_ask_fallback():
    with patch("app.main.route", return_value=MOCK_FALLBACK):
        response = client.post("/ask", json={"question": "What is the boiling point of water?"})
    assert response.status_code == 200
    assert response.json()["confidence"] == "none"


def test_ask_appointment():
    with patch("app.main.route", return_value=MOCK_APPOINTMENT):
        response = client.post("/ask", json={"question": "Book a cardiology slot for Monday"})
    assert response.status_code == 200
    assert response.json()["tool_used"] == "check_available_slots"


def test_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
