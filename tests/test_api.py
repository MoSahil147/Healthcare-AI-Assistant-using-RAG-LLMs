import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

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


# --- API endpoint tests ---

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


def test_ask_auto_assigns_session_id():
    with patch("app.main.route", return_value=MOCK_RAG):
        response = client.post("/ask", json={"question": "What is HIPAA?"})
    assert response.status_code == 200


def test_ask_accepts_explicit_session_id():
    with patch("app.main.route", return_value=MOCK_RAG) as mock_route:
        client.post("/ask", json={"question": "What is HIPAA?", "session_id": "test-session-abc"})
        _, kwargs = mock_route.call_args
        # route is called positionally: route(question, session_id)
        args = mock_route.call_args[0]
        assert args[1] == "test-session-abc"


# --- Agent unit tests ---

from app.agent import _get_department, _get_date, check_available_slots, _CANONICAL_SLOTS, _ALIASES


def test_get_department_canonical():
    assert _get_department("I need a cardiology appointment") == "cardiology"
    assert _get_department("book a slot in dermatology") == "dermatology"
    assert _get_department("neurology please") == "neurology"


def test_get_department_alias():
    assert _get_department("orthopedics appointment") == "orthopaedics"
    assert _get_department("psychiatry slot") == "mental health"
    assert _get_department("pediatrics visit") == "paediatrics"
    assert _get_department("gynecology checkup") == "gynaecology"


def test_get_department_default():
    assert _get_department("I want to see a doctor") == "default"
    assert _get_department("") == "default"


def test_get_date_returns_next_occurrence():
    result = _get_date("I want Monday")
    assert "Monday" in result


def test_get_date_no_day_mentioned():
    assert _get_date("I want an appointment soon") == "next available date"


def test_check_available_slots_known_dept():
    result = check_available_slots("cardiology", "Monday, June 02")
    assert result["department"] == "Cardiology"
    assert result["available_slots"] == _CANONICAL_SLOTS["cardiology"]
    assert "1800-123-4567" in result["note"]


def test_check_available_slots_default():
    result = check_available_slots("default", "next available date")
    assert result["department"] == "General"
    assert result["available_slots"] == _CANONICAL_SLOTS["default"]


def test_aliases_resolve_to_canonical_slots():
    # _get_department resolves aliases before check_available_slots is called,
    # so the canonical key's slots should always be returned for alias input
    for alias, canon in _ALIASES.items():
        resolved = _get_department(f"I need {alias} appointment")
        assert resolved == canon
        result = check_available_slots(resolved, "any")
        assert result["available_slots"] == _CANONICAL_SLOTS[canon]


# --- RAG unit tests ---

from app.rag import _confidence, _get_history, _session_histories


def test_confidence_high():
    assert _confidence(0.1) == "high"
    assert _confidence(0.29) == "high"


def test_confidence_medium():
    assert _confidence(0.3) == "medium"
    assert _confidence(0.59) == "medium"


def test_confidence_low():
    assert _confidence(0.6) == "low"
    assert _confidence(0.99) == "low"


def test_session_history_isolated():
    _session_histories.clear()
    h1 = _get_history("session-1")
    h1.append({"role": "user", "content": "hello"})

    h2 = _get_history("session-2")
    assert h2 == []  # session-2 should not see session-1's history


def test_session_history_persists_within_session():
    _session_histories.clear()
    h = _get_history("session-x")
    h.append({"role": "user", "content": "question"})
    assert _get_history("session-x")[0]["content"] == "question"
