from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class PartCandidate:
    """A normalized catalog candidate for one hardware category."""
    category: str
    name: str
    price: float
    attributes: dict[str, Any]

@dataclass
class BuildProposal:
    """A proposed recommendation assembled from selected catalog parts."""
    selected_parts: dict[str, PartCandidate] = field(default_factory=dict)
    total_price: float = 0.0
    unmet_constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempt: int = 1

@dataclass
class ValidationReport:
    """The validation result for a proposed build or single-part recommendation."""
    passed: bool
    issues: list[str]
    budget_ok: bool
    coverage_ok: bool
    compatibility_warnings: list[str]

@dataclass
class CurrencyContext:
    """Currency-normalization details derived for the current turn."""
    needs_conversion: bool
    source_currency: str
    target_currency: str
    rate: float | None = None
    rate_date: str | None = None
    normalized_budget_target_usd: float | None = None
    normalized_budget_min_usd: float | None = None
    normalized_budget_max_usd: float | None = None
    warnings: list[str] = field(default_factory=list)
    conversion_mode: str = "budget_and_display"
    reuse_existing_build: bool = False
    budget_to_usd_rate: float | None = None
    usd_to_display_rate: float | None = None

@dataclass
class SessionState:
    """Mutable session memory carried across turns of the assistant."""
    intent: str | None = None
    budget_target: float | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    budget_currency: str = "USD"
    display_currency: str = "USD"
    conversion_mode: str = "budget_and_display"
    normalized_budget_target_usd: float | None = None
    normalized_budget_min_usd: float | None = None
    normalized_budget_max_usd: float | None = None
    exchange_rate: float | None = None
    exchange_rate_base: str | None = None
    exchange_rate_date: str | None = None
    exchange_rate_source: str | None = None
    currency_warnings: list[str] = field(default_factory=list)
    use_case: str | None = None
    preferences: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    current_build: BuildProposal | None = None
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    requested_categories: list[str] = field(default_factory=list)
    conversation_history: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    unsupported_categories: list[str] = field(default_factory=list)
