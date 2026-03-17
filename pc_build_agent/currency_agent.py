from __future__ import annotations

import json
import textwrap
from typing import Any

from .config import DEFAULT_MODEL
from .exchange_rates import ExchangeRateError, ExchangeRateQuote, ExchangeRateTool, UnsupportedCurrencyError
from .models import BuildProposal, CurrencyContext, SessionState, ValidationReport
from .openai_support import build_openai_client, request_json_payload
from .utils import unique_strings

def _clean_currency_code(value: Any) -> str | None:
    """Normalize currency names and aliases into ISO-style codes."""
    if value in (None, ""):
        return None
    cleaned = str(value).strip().upper()
    alias_map = {
        "RUPEE": "INR",
        "RUPEES": "INR",
        "INDIAN RUPEE": "INR",
        "INDIAN RUPEES": "INR",
        "EURO": "EUR",
        "EUROS": "EUR",
        "POUND": "GBP",
        "POUNDS": "GBP",
        "POUND STERLING": "GBP",
        "YEN": "JPY",
        "YUAN": "CNY",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
    }
    return alias_map.get(cleaned, cleaned)

def format_money(amount: float, currency: str) -> str:
    """Format a numeric amount for a user-facing currency display line."""
    code = currency.upper()
    symbols = {
        "USD": "$",
        "EUR": "EUR ",
        "GBP": "GBP ",
        "INR": "INR ",
        "JPY": "JPY ",
        "AUD": "AUD ",
        "CAD": "CAD ",
    }
    prefix = symbols.get(code, f"{code} ")
    return f"{prefix}{amount:,.2f}"

VALID_CONVERSION_MODES = {"budget_and_display", "display_only", "usd_only"}

class CurrencyAgent:
    """Resolve budget/display currency needs for the planning workflow."""

    def __init__(self, model: str = DEFAULT_MODEL, rate_tool: ExchangeRateTool | None = None) -> None:
        self.model = model
        self.rate_tool = rate_tool or ExchangeRateTool()
        self.client = build_openai_client("this currency agent")

    def analyze(self, user_message: str, state: SessionState) -> dict[str, Any]:
        """Use the model to extract currency intent for the current turn."""
        prompt = textwrap.dedent(
            f"""
            You are the currency agent in a multi-agent PC build workflow.
            Return JSON only with keys:
            budget_currency, display_currency, conversion_mode, reuse_existing_build, warnings.

            Allowed conversion_mode values:
            - "budget_and_display": use the requested currency budget, normalize to USD internally, and display final prices in the requested currency.
            - "display_only": keep the existing USD build and only convert the displayed prices.
            - "usd_only": treat everything as USD.

            Rules:
            - Default plain "$" and plain "dollars" to USD unless the user explicitly names another country/currency.
            - "rupee" or "rupees" means INR.
            - "show this in rupees" or similar, when a build already exists, should set conversion_mode to "display_only" and reuse_existing_build to true.
            - If the user is clearly budgeting in a non-USD currency, set budget_currency and display_currency to that currency unless they ask otherwise.
            - Use ISO currency codes like USD, INR, EUR, GBP, JPY.
            - If there is no non-USD currency request, use USD for both fields.
            - warnings should be a list of short strings. Include a warning if you are assuming USD from an ambiguous "$" reference.

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
            progress_message="Currency agent: resolving budget and display currency",
            failure_context="currency analysis failed",
        )

    def resolve(self, user_message: str, state: SessionState) -> CurrencyContext:
        """Resolve live currency context and normalize budgets into USD."""
        payload = self.analyze(user_message, state)

        source_currency = _clean_currency_code(payload.get("budget_currency")) or state.budget_currency or "USD"
        target_currency = _clean_currency_code(payload.get("display_currency")) or source_currency
        conversion_mode = str(payload.get("conversion_mode") or "budget_and_display")
        if conversion_mode not in VALID_CONVERSION_MODES:
            conversion_mode = "budget_and_display"

        warnings = unique_strings(str(item) for item in payload.get("warnings", []) if item)
        reuse_existing_build = bool(payload.get("reuse_existing_build"))

        if source_currency == "USD" and target_currency == "USD":
            context = self._build_context(
                state=state,
                source_currency="USD",
                target_currency="USD",
                warnings=warnings,
                conversion_mode=conversion_mode,
                reuse_existing_build=reuse_existing_build,
                budget_quote=None,
                display_quote=None,
            )
            self._apply_context_to_state(state, context, None)
            return context

        supported = self._supported_currencies()
        self._ensure_supported_currency(source_currency, supported)
        self._ensure_supported_currency(target_currency, supported)

        budget_quote: ExchangeRateQuote | None = None
        if source_currency != "USD":
            budget_quote = self._fetch_quote(
                base_currency=source_currency,
                target_currency="USD",
                unsupported_message=(
                    f"currency conversion could not be completed because {source_currency} is not supported by the live rate source."
                ),
                retry_message="Retry later or switch to a USD budget.",
            )

        display_quote: ExchangeRateQuote | None = None
        if target_currency != "USD":
            display_quote = self._fetch_quote(
                base_currency="USD",
                target_currency=target_currency,
                unsupported_message=(
                    f"currency conversion could not be completed because {target_currency} is not supported by the live rate source."
                ),
                retry_message="Retry later or switch to USD output.",
            )

        context = self._build_context(
            state=state,
            source_currency=source_currency,
            target_currency=target_currency,
            warnings=warnings,
            conversion_mode=conversion_mode,
            reuse_existing_build=reuse_existing_build,
            budget_quote=budget_quote,
            display_quote=display_quote,
        )
        self._apply_context_to_state(state, context, display_quote or budget_quote)
        return context

    def build_display_payload(
        self,
        state: SessionState,
        build: BuildProposal,
        report: ValidationReport,
        context: CurrencyContext,
    ) -> dict[str, Any]:
        """Build the display payload used in the final explanation prompt."""
        display_parts = [
            self._display_part(category, candidate.name, candidate.price, state, context)
            for category, candidate in build.selected_parts.items()
        ]

        total_usd = round(build.total_price, 2)
        total_display = self._display_amount(total_usd, state.display_currency, context.usd_to_display_rate)
        budget_original = state.budget_target
        budget_usd = state.normalized_budget_target_usd or state.budget_target

        price_lines = [self._format_price_line(part, state.display_currency) for part in display_parts]

        total_line = f"Total: {format_money(total_usd, 'USD')}"
        if state.display_currency != "USD":
            total_line += f" / {format_money(total_display, state.display_currency)}"

        budget_line = None
        if budget_original is not None:
            budget_line = f"Original budget: {format_money(budget_original, state.budget_currency)}"
            if state.budget_currency != "USD" and budget_usd is not None:
                budget_line += f" | Internal USD planning budget: {format_money(budget_usd, 'USD')}"

        exchange_line = None
        if state.display_currency != "USD" and state.exchange_rate is not None:
            exchange_line = (
                f"Reference exchange rate: 1 {state.exchange_rate_base or 'USD'} = "
                f"{state.exchange_rate:.4f} {state.display_currency} "
                f"(date: {state.exchange_rate_date}, source: {state.exchange_rate_source})"
            )

        return {
            "currency_context": {
                "budget_currency": state.budget_currency,
                "display_currency": state.display_currency,
                "conversion_mode": state.conversion_mode,
                "exchange_rate": state.exchange_rate,
                "exchange_rate_base": state.exchange_rate_base,
                "exchange_rate_date": state.exchange_rate_date,
                "exchange_rate_source": state.exchange_rate_source,
                "warnings": unique_strings([*state.currency_warnings, *context.warnings]),
            },
            "budget_line": budget_line,
            "price_lines": price_lines,
            "total_line": total_line,
            "exchange_line": exchange_line,
            "disclaimer": (
                "Currency conversion uses a live reference exchange rate. "
                "Bank, card, and payment-provider transaction rates may differ."
            ),
            "build": {
                "selected_parts": display_parts,
                "total_price_usd": total_usd,
                "total_price_display": total_display,
                "validation_issues": report.issues,
                "compatibility_warnings": report.compatibility_warnings,
            },
        }

    def _supported_currencies(self) -> dict[str, str]:
        """Fetch or return the cached supported-currency list."""
        try:
            return self.rate_tool.get_supported_currencies()
        except ExchangeRateError as exc:
            raise RuntimeError(f"currency conversion could not be completed: {exc}") from exc

    def _ensure_supported_currency(self, code: str, supported: dict[str, str]) -> None:
        """Raise a user-facing error if a requested currency is unavailable."""
        if code.upper() != "USD" and code.upper() not in supported:
            raise RuntimeError(
                f"currency conversion could not be completed because {code} is not supported by the live rate source. "
                "Please retry with USD or another supported currency code."
            )

    def _fetch_quote(
        self,
        base_currency: str,
        target_currency: str,
        unsupported_message: str,
        retry_message: str,
    ) -> ExchangeRateQuote:
        """Fetch a quote and normalize provider errors into workflow errors."""
        try:
            return self.rate_tool.get_latest_rate(base_currency, target_currency)
        except UnsupportedCurrencyError as exc:
            raise RuntimeError(unsupported_message) from exc
        except ExchangeRateError as exc:
            raise RuntimeError(
                f"currency conversion could not be completed because the live exchange-rate service failed: {exc}. "
                f"{retry_message}"
            ) from exc

    def _normalize_budget_values(
        self,
        state: SessionState,
        budget_quote: ExchangeRateQuote | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Convert the stored budget range into USD using the given quote."""
        if budget_quote is None:
            return state.budget_target, state.budget_min, state.budget_max
        return (
            round(state.budget_target * budget_quote.rate, 2) if state.budget_target is not None else None,
            round(state.budget_min * budget_quote.rate, 2) if state.budget_min is not None else None,
            round(state.budget_max * budget_quote.rate, 2) if state.budget_max is not None else None,
        )

    def _build_context(
        self,
        state: SessionState,
        source_currency: str,
        target_currency: str,
        warnings: list[str],
        conversion_mode: str,
        reuse_existing_build: bool,
        budget_quote: ExchangeRateQuote | None,
        display_quote: ExchangeRateQuote | None,
    ) -> CurrencyContext:
        """Assemble the final `CurrencyContext` object for the turn."""
        normalized_target, normalized_min, normalized_max = self._normalize_budget_values(state, budget_quote)
        rate_quote = display_quote or budget_quote
        return CurrencyContext(
            needs_conversion=source_currency != "USD" or target_currency != "USD",
            source_currency=source_currency,
            target_currency=target_currency,
            rate=rate_quote.rate if rate_quote else 1.0,
            rate_date=rate_quote.date if rate_quote else None,
            normalized_budget_target_usd=normalized_target,
            normalized_budget_min_usd=normalized_min,
            normalized_budget_max_usd=normalized_max,
            warnings=warnings,
            conversion_mode=conversion_mode,
            reuse_existing_build=reuse_existing_build and state.current_build is not None,
            budget_to_usd_rate=budget_quote.rate if budget_quote else 1.0,
            usd_to_display_rate=display_quote.rate if display_quote else 1.0,
        )

    def _display_amount(
        self,
        amount_usd: float,
        display_currency: str,
        usd_to_display_rate: float | None,
    ) -> float:
        """Convert a USD amount into the requested display currency."""
        if display_currency == "USD":
            return round(amount_usd, 2)
        return round(amount_usd * (usd_to_display_rate or 1.0), 2)

    def _display_part(
        self,
        category: str,
        name: str,
        price: float,
        state: SessionState,
        context: CurrencyContext,
    ) -> dict[str, Any]:
        """Build the display payload fragment for one selected part."""
        price_usd = round(price, 2)
        return {
            "category": category.replace("-", " ").title(),
            "name": name,
            "price_usd": price_usd,
            "price_display": self._display_amount(price_usd, state.display_currency, context.usd_to_display_rate),
        }

    def _format_price_line(self, part: dict[str, Any], display_currency: str) -> str:
        """Format a single per-part price line for the final response."""
        line = (
            f"- {part['category']}: {part['name']} "
            f"({format_money(part['price_usd'], 'USD')}"
        )
        if display_currency != "USD":
            line += f" / {format_money(part['price_display'], display_currency)})"
        else:
            line += ")"
        return line

    def _apply_context_to_state(
        self,
        state: SessionState,
        context: CurrencyContext,
        quote: ExchangeRateQuote | None,
    ) -> None:
        """Persist resolved currency context back onto the session state."""
        state.budget_currency = context.source_currency
        state.display_currency = context.target_currency
        state.conversion_mode = context.conversion_mode
        state.normalized_budget_target_usd = context.normalized_budget_target_usd
        state.normalized_budget_min_usd = context.normalized_budget_min_usd
        state.normalized_budget_max_usd = context.normalized_budget_max_usd
        state.exchange_rate = context.usd_to_display_rate if state.display_currency != "USD" else context.budget_to_usd_rate
        state.exchange_rate_base = "USD" if state.display_currency != "USD" else context.source_currency
        state.exchange_rate_date = context.rate_date
        state.exchange_rate_source = quote.source if quote else None
        state.currency_warnings = unique_strings(context.warnings)

    def _state_snapshot(self, state: SessionState) -> dict[str, Any]:
        """Build the compact currency-state snapshot used in the prompt."""
        return {
            "budget_currency": state.budget_currency,
            "display_currency": state.display_currency,
            "conversion_mode": state.conversion_mode,
            "budget_target": state.budget_target,
            "budget_min": state.budget_min,
            "budget_max": state.budget_max,
            "current_build_exists": state.current_build is not None,
        }
