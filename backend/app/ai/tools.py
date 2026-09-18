"""Tool (function-calling) schemas exposed to the LLM.

Four tools only — every extra tool costs tokens and latency on a phone call:

* `search_knowledge_base` — the model can ask for another retrieval pass with a
  better query when the pre-fetched context missed (e.g. the caller named a
  specialisation we did not detect).
* `lookup_course` — exact structured lookup by course code/slug; returns fees,
  eligibility, duration as JSON so the model never has to guess a number.
* `escalate_to_human` — the only correct action when the KB has no answer.
* `offer_details` — send a long list/fee table by SMS, WhatsApp or email instead
  of reading it out loud.
"""

from __future__ import annotations

from typing import Any

from .llm.base import ToolSpec

SEARCH_KB = ToolSpec(
    name="search_knowledge_base",
    description=(
        "Search the NMIMS Global University, Dhule knowledge base for records that answer the "
        "caller's question. Use it when the provided context is missing or only "
        "partly relevant. Returns short text snippets with titles, categories and "
        "whether each record is verified."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query, in English, using the exact course/exam names.",
            },
            "category": {
                "type": "string",
                "enum": [
                    "course", "specialisation", "eligibility", "fees",
                    "admission_process", "important_dates", "entrance_exam",
                    "scholarships", "hostel", "placements", "facilities",
                    "documents", "contact", "transport", "loan_payment",
                    "department", "policy", "faq", "university",
                ],
                "description": "Optional category filter.",
            },
            "academic_year": {
                "type": "string",
                "description": "Optional academic year filter, e.g. 2025-26.",
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        },
        "required": ["query"],
    },
)

LOOKUP_COURSE = ToolSpec(
    name="lookup_course",
    description=(
        "Fetch the exact structured record for one programme: annual and total "
        "fees, eligibility, entrance exam, duration, seats, school/department. "
        "Prefer this over search_knowledge_base when the caller named a specific "
        "programme."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "course": {
                "type": "string",
                "description": "Programme name or code, e.g. 'B.Tech CSE', 'B.Pharm', 'MBA Finance'.",
            },
            "specialisation": {
                "type": "string",
                "description": "Optional specialisation, e.g. 'Artificial Intelligence'.",
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["fees", "eligibility", "duration", "seats", "entrance_exam",
                             "placements", "hostel", "all"],
                },
                "description": "Which facts to return. Defaults to all.",
            },
        },
        "required": ["course"],
    },
)

ESCALATE = ToolSpec(
    name="escalate_to_human",
    description=(
        "Transfer the call to a human admissions counsellor. REQUIRED when the "
        "knowledge base does not contain the answer, when the caller asks for a "
        "person, when the matter is a complaint / payment dispute / grievance, or "
        "when you are not confident. Provide a concise summary: the caller's "
        "language, the programme they asked about, and what could not be answered."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": [
                    "kb_no_answer", "low_confidence", "caller_requested_human",
                    "complaint", "payment_dispute", "sensitive_or_legal",
                    "unsupported_language", "technical_failure", "out_of_scope",
                ],
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentences for the human agent: who is calling and what they need.",
            },
            "priority": {"type": "string", "enum": ["normal", "high"], "default": "normal"},
        },
        "required": ["reason", "summary"],
    },
)

OFFER_DETAILS = ToolSpec(
    name="offer_details",
    description=(
        "Offer to send detailed information by SMS, WhatsApp or email instead of "
        "reading a long list out loud. Use for full fee breakdowns, document "
        "lists, the complete course catalogue, or a prospectus link."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "enum": ["sms", "whatsapp", "email"]},
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The facts/lines to send. Keep each under 140 characters.",
            },
            "title": {"type": "string", "description": "Short label for the message, e.g. 'B.Tech CSE fees'."},
            "ask_for_destination": {
                "type": "boolean",
                "default": True,
                "description": "True = ask the caller for their number/email; False = use the calling number.",
            },
        },
        "required": ["channel", "items"],
    },
)

ALL_TOOLS: list[ToolSpec] = [SEARCH_KB, LOOKUP_COURSE, ESCALATE, OFFER_DETAILS]


def tools_for(supports_tools: bool) -> list[ToolSpec]:
    return list(ALL_TOOLS) if supports_tools else []


def render_tool_result(name: str, payload: dict[str, Any]) -> str:
    """Compact JSON is cheaper than prose and the models parse it reliably."""
    import json

    return json.dumps(
        {"tool": name, "result": payload}, ensure_ascii=False, default=str
    )[:6000]
