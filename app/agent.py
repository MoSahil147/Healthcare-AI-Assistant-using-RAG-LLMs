import logging
import re
from datetime import date, timedelta

from app.rag import answer_question

logger = logging.getLogger(__name__)

# if any of these words appear in the question, skip RAG and go to the appointment tool
GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "howdy"}

# if any of these words appear in the question, skip RAG and go to the appointment tool
APPOINTMENT_KEYWORDS = {
    "appointment", "book", "slot", "schedule", "available", "availability",
    "cardiology", "orthopaedics", "orthopedics", "dermatology", "neurology",
    "general medicine", "psychiatry", "mental health", "paediatrics", "pediatrics",
    "gynaecology", "gynecology", "ob/gyn", "oncology", "endocrinology",
    "gastroenterology", "physiotherapy", "physical therapy", "nephrology", "urology",
}

# one slot list per department, no duplicated entries
# in a real system this would call a scheduling API
_CANONICAL_SLOTS: dict[str, list[str]] = {
    "cardiology":       ["Monday 9:00 AM", "Monday 2:30 PM", "Wednesday 11:00 AM"],
    "orthopaedics":     ["Tuesday 10:00 AM", "Thursday 3:00 PM", "Friday 9:30 AM"],
    "dermatology":      ["Wednesday 9:00 AM", "Friday 1:00 PM"],
    "neurology":        ["Monday 11:00 AM", "Thursday 9:00 AM"],
    "general medicine": ["Monday 8:00 AM", "Tuesday 2:00 PM", "Wednesday 4:00 PM", "Friday 10:00 AM"],
    "mental health":    ["Monday 10:00 AM", "Tuesday 3:00 PM", "Thursday 11:00 AM"],
    "paediatrics":      ["Monday 9:30 AM", "Wednesday 2:00 PM"],
    "gynaecology":      ["Tuesday 11:00 AM", "Thursday 2:00 PM"],
    "oncology":         ["Wednesday 10:00 AM", "Friday 9:00 AM"],
    "default":          ["Monday 10:00 AM", "Wednesday 2:00 PM", "Friday 11:00 AM"],
}

# alternate spellings that resolve to the canonical department name above
_ALIASES: dict[str, str] = {
    "orthopedics": "orthopaedics",
    "psychiatry":  "mental health",
    "pediatrics":  "paediatrics",
    "gynecology":  "gynaecology",
}


def check_available_slots(department: str, requested_date: str) -> dict:
    slots = _CANONICAL_SLOTS.get(department, _CANONICAL_SLOTS["default"])
    dept_label = department.title() if department != "default" else "General"
    return {
        "department":      dept_label,
        "requested_date":  requested_date,
        "available_slots": slots,
        "note": "These are simulated slots. Please call 1800-123-4567 to confirm.",
    }


def _get_department(question: str) -> str:
    # walk through canonical departments first, then check aliases
    q = question.lower()
    for dept in _CANONICAL_SLOTS:
        if dept != "default" and dept in q:
            return dept
    for alias, canon in _ALIASES.items(): # alias is worng spelling, canon is right, see _ALIASES
        if alias in q:
            return canon
    return "default"


def _get_date(question: str) -> str:
    # pull out the day name if mentioned and convert it to an actual calendar date
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    q = question.lower()
    for name, num in weekdays.items():
        if name in q:
            today = date.today()
            days_ahead = (num - today.weekday()) % 7 or 7 # or 7 is to handle the case where teh result is 0, will force to next week
            return (today + timedelta(days=days_ahead)).strftime("%A, %B %d") # Monday, June 27
    return "next available date"


def route(question: str, session_id: str = "") -> dict: # turns to dict, as key value pairs
    logger.info("Routing: %s", question)

    # greeting
    if any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS):
        logger.info("Greeting detected")
        return {
            "answer":     "Hello! How can I help you with your healthcare questions today?",
            "sources":    [],
            "confidence": "high",
            "tool_used":  None,
            "tool_output": None,
        }

    # appointment check uses pure keyword matching, no LLM call needed here
    if any(kw in question.lower() for kw in APPOINTMENT_KEYWORDS):
        dept = _get_department(question)
        requested_date = _get_date(question)
        slot_info = check_available_slots(dept, requested_date)

        answer = (
            f"I can check mock appointment availability for {slot_info['department']}. "
            f"Available slots around {requested_date}: {', '.join(slot_info['available_slots'])}. "
            f"{slot_info['note']}"
        )
        logger.info("Appointment tool used -- dept: %s", dept)
        return {
            "answer":      answer,
            "sources":     [],
            "confidence":  "high",
            "tool_used":   "check_available_slots",
            "tool_output": slot_info,
        }

    # not an appointment question or greeting, hand off to the RAG pipeline
    return answer_question(question, session_id)
