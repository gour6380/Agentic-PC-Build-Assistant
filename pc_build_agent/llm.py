from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from typing import Any

from .config import CSV_FILES, DEFAULT_MODEL
from .models import BuildProposal, CurrencyContext, SessionState, ValidationReport
from .openai_support import build_openai_client, request_json_payload, request_text_payload

class LLMHelper:
    """Planner-side OpenAI helper for extraction and final explanations."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.client = build_openai_client("this agentic workflow")

    def extract(self, user_message: str, state: SessionState) -> dict[str, Any]:
        """Extract structured planning requirements from the current user turn."""
        prompt = textwrap.dedent(
            f"""
            Extract PC build planning details from the user request and current session state.
            Return JSON only with keys:
            intent, budget_target, budget_min, budget_max, use_case, requested_categories,
            unsupported_categories, preferred_brands, excluded_brands,
            include_peripherals, form_factor, memory_target_gb,
            storage_target_gb, needs_wifi, budget_currency, display_currency,
            conversion_mode.

            Rules:
            - Allowed requested_categories are: {sorted(CSV_FILES)}.
            - Put anything not covered by that list into unsupported_categories.
            - For a full custom PC request, set intent to "full_build".
            - For a single component request, set intent to "single_part".
            - Use ISO currency codes like USD, INR, EUR, GBP, JPY for budget_currency and display_currency.
            - Default plain "$" or plain "dollars" to USD unless the user clearly names a different currency.
            - If the user asks to show an existing build in another currency, set conversion_mode to "display_only".
            - Otherwise use "budget_and_display" for a non-USD planning request and "usd_only" for normal USD requests.
            - If the user gives one exact budget, set budget_target to that number and
              budget_min/budget_max to a +/- 5% band.
            - If the user gives an upper bound like "under $400", set budget_max and
              budget_target to that value and leave budget_min null.
            - If the user gives a budget range, preserve the range and set budget_target
              to the midpoint.
            - If a field is missing, use null for scalars and [] for arrays.

            Current state:
            {json.dumps(self._state_snapshot(state), indent=2)}

            User message:
            {user_message}
            """
        ).strip()

        return request_json_payload(
            client=self.client,
            model=self.model,
            prompt=prompt,
            progress_message="Planner agent: extracting requirements",
            failure_context="requirement extraction failed",
        )

    def explain(
        self,
        state: SessionState,
        build: BuildProposal,
        report: ValidationReport,
        currency_context: CurrencyContext,
        display_payload: dict[str, Any],
    ) -> str | None:
        """Generate the final user-facing recommendation summary."""
        payload = {
            "state": self._state_snapshot(state),
            "build": {
                "selected_parts": {category: asdict(candidate) for category, candidate in build.selected_parts.items()},
                "total_price": build.total_price,
                "warnings": build.warnings,
            },
            "validation": asdict(report),
            "currency_context": asdict(currency_context),
            "display_payload": display_payload,
        }
        prompt = textwrap.dedent(
            f"""
            You are a transparent PC build assistant. Summarize the recommendation in 5-8 short sentences.
            Mention any warnings or limitations plainly.
            Then include the exact budget/price lines from display_payload in your response.
            Make it clear that the dataset prices are USD and the second currency is a live reference conversion.
            Do not invent parts or prices that are not present in the provided data.
            End by asking if the user wants changes.

            Data:
            {json.dumps(payload, indent=2)}
            """
        ).strip()

        return request_text_payload(
            client=self.client,
            model=self.model,
            prompt=prompt,
            progress_message="Planner agent: generating final response",
            failure_context="response generation failed",
        )

    def _state_snapshot(self, state: SessionState) -> dict[str, Any]:
        """Build the session snapshot used in planner prompts."""
        return {
            "intent": state.intent,
            "budget_target": state.budget_target,
            "budget_min": state.budget_min,
            "budget_max": state.budget_max,
            "budget_currency": state.budget_currency,
            "display_currency": state.display_currency,
            "normalized_budget_target_usd": state.normalized_budget_target_usd,
            "normalized_budget_min_usd": state.normalized_budget_min_usd,
            "normalized_budget_max_usd": state.normalized_budget_max_usd,
            "exchange_rate": state.exchange_rate,
            "exchange_rate_base": state.exchange_rate_base,
            "exchange_rate_date": state.exchange_rate_date,
            "exchange_rate_source": state.exchange_rate_source,
            "use_case": state.use_case,
            "preferences": state.preferences,
            "requested_categories": state.requested_categories,
            "unsupported_categories": state.unsupported_categories,
        }
