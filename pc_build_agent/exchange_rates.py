from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

FRANKFURTER_API_BASE = "https://api.frankfurter.dev/v1"
FRANKFURTER_SOURCE = "Frankfurter (ECB reference rates)"

class ExchangeRateError(RuntimeError):
    """Raised when the live exchange-rate service cannot satisfy a request."""
    pass

class UnsupportedCurrencyError(ExchangeRateError):
    """Raised when a currency is not supported by the live provider."""
    pass

@dataclass
class ExchangeRateQuote:
    """A normalized live exchange-rate quote."""
    base: str
    target: str
    rate: float
    date: str
    source: str = FRANKFURTER_SOURCE

class ExchangeRateTool:
    """Fetch and cache live reference exchange rates for budget conversion."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._currencies_cache: dict[str, str] | None = None
        self._rate_cache: dict[tuple[str, str], ExchangeRateQuote] = {}

    def _request_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Request JSON data from the configured exchange-rate endpoint."""
        url = f"{FRANKFURTER_API_BASE}/{path.lstrip('/')}"
        try:
            response = httpx.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise ExchangeRateError(f"failed to fetch exchange-rate data from {url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ExchangeRateError(f"unexpected exchange-rate response from {url}")
        return payload

    def get_supported_currencies(self) -> dict[str, str]:
        """Return the cached list of provider-supported currency codes."""
        if self._currencies_cache is None:
            payload = self._request_json("currencies")
            self._currencies_cache = {str(code).upper(): str(name) for code, name in payload.items()}
            self._currencies_cache["USD"] = self._currencies_cache.get("USD", "United States Dollar")
        return self._currencies_cache

    def get_latest_rate(self, base_currency: str, target_currency: str) -> ExchangeRateQuote:
        """Return the latest available quote from one currency into another."""
        base = base_currency.upper()
        target = target_currency.upper()
        if base == target:
            return ExchangeRateQuote(
                base=base,
                target=target,
                rate=1.0,
                date=datetime.now(timezone.utc).date().isoformat(),
                source=FRANKFURTER_SOURCE,
            )

        supported = self.get_supported_currencies()
        if base not in supported:
            raise UnsupportedCurrencyError(f"unsupported base currency: {base}")
        if target not in supported:
            raise UnsupportedCurrencyError(f"unsupported target currency: {target}")

        cache_key = (base, target)
        if cache_key in self._rate_cache:
            return self._rate_cache[cache_key]

        payload = self._request_json("latest", {"base": base, "symbols": target})
        rates = payload.get("rates")
        if not isinstance(rates, dict) or target not in rates:
            raise ExchangeRateError(f"rate for {base}->{target} was not present in the provider response")

        try:
            rate = float(rates[target])
        except (TypeError, ValueError) as exc:
            raise ExchangeRateError(f"invalid rate value for {base}->{target}") from exc

        quote = ExchangeRateQuote(
            base=base,
            target=target,
            rate=rate,
            date=str(payload.get("date") or datetime.now(timezone.utc).date().isoformat()),
            source=FRANKFURTER_SOURCE,
        )
        self._rate_cache[cache_key] = quote
        return quote

    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert an amount using the latest cached or fetched quote."""
        quote = self.get_latest_rate(from_currency, to_currency)
        return round(amount * quote.rate, 2)
