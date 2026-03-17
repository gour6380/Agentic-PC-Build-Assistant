from __future__ import annotations

from typing import Any

from .config import CSV_FILES
from .llm import LLMHelper
from .models import SessionState
from .utils import safe_float

def merge_state(state: SessionState, extracted: dict[str, Any]) -> SessionState:
    """Merge normalized extraction output into the current session state."""
    if extracted.get("intent"):
        state.intent = extracted["intent"]
    if extracted.get("budget_target") is not None:
        state.budget_target = float(extracted["budget_target"])
    if extracted.get("budget_min") is not None:
        state.budget_min = float(extracted["budget_min"])
    if extracted.get("budget_max") is not None:
        state.budget_max = float(extracted["budget_max"])
    if extracted.get("use_case"):
        state.use_case = extracted["use_case"]
    if extracted.get("budget_currency"):
        state.budget_currency = str(extracted["budget_currency"]).upper()
    if extracted.get("display_currency"):
        state.display_currency = str(extracted["display_currency"]).upper()
    if extracted.get("conversion_mode"):
        state.conversion_mode = str(extracted["conversion_mode"])

    existing_categories = set(state.requested_categories)
    state.requested_categories = sorted(existing_categories | set(extracted.get("requested_categories", [])))
    state.unsupported_categories = sorted(set(state.unsupported_categories) | set(extracted.get("unsupported_categories", [])))

    for key, value in extracted.get("preferences", {}).items():
        if value in (None, [], ""):
            continue
        if isinstance(value, list):
            current = state.preferences.get(key, [])
            state.preferences[key] = sorted(set(current) | set(value))
        else:
            state.preferences[key] = value
    return state

def _normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the raw extraction payload into the session merge shape."""
    requested_categories = [
        str(category)
        for category in payload.get("requested_categories", [])
        if category not in (None, "")
    ]
    unsupported_categories = [
        str(category)
        for category in payload.get("unsupported_categories", [])
        if category not in (None, "")
    ]

    valid_categories = sorted({category for category in requested_categories if category in CSV_FILES})
    invalid_categories = sorted({category for category in requested_categories if category not in CSV_FILES})
    unsupported_categories = sorted(set(unsupported_categories) | set(invalid_categories))

    return {
        "intent": payload.get("intent"),
        "budget_target": safe_float(payload.get("budget_target")),
        "budget_min": safe_float(payload.get("budget_min")),
        "budget_max": safe_float(payload.get("budget_max")),
        "budget_currency": payload.get("budget_currency"),
        "display_currency": payload.get("display_currency"),
        "conversion_mode": payload.get("conversion_mode"),
        "use_case": payload.get("use_case"),
        "requested_categories": valid_categories,
        "unsupported_categories": unsupported_categories,
        "preferences": {
            "preferred_brands": sorted(set(str(item) for item in payload.get("preferred_brands", []) if item)),
            "excluded_brands": sorted(set(str(item) for item in payload.get("excluded_brands", []) if item)),
            "include_peripherals": payload.get("include_peripherals"),
            "form_factor": payload.get("form_factor"),
            "memory_target_gb": payload.get("memory_target_gb"),
            "storage_target_gb": payload.get("storage_target_gb"),
            "needs_wifi": payload.get("needs_wifi"),
        },
    }

def extract_requirements(message: str, state: SessionState, llm_helper: LLMHelper) -> SessionState:
    """Extract, normalize, and merge planning requirements for a user turn."""
    llm_result = llm_helper.extract(message, state)
    normalized = _normalize_extraction_payload(llm_result)
    return merge_state(state, normalized)
