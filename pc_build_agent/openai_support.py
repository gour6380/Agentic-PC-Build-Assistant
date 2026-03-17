from __future__ import annotations

from typing import Any

from .config import get_config_value
from .utils import parse_json_response, progress_step
from openai import OpenAI

def build_openai_client(purpose: str) -> Any:
    """Build an OpenAI client from the configured project credentials."""
    api_key = get_config_value("OPENAI_API_KEY", "")
    base_url = get_config_value("OPENAI_BASE_URL", "")
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed in the active environment.")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is required for {purpose}. Set it in the project .env file or your environment.")
    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)

def _request_output_text(
    client: Any,
    model: str,
    prompt: str,
    progress_message: str,
    failure_context: str,
) -> str:
    """Request a model response and return its output text."""
    try:
        with progress_step(progress_message):
            response = client.responses.create(model=model, input=prompt)
    except Exception as exc:
        raise RuntimeError(f"{failure_context}: {exc}") from exc
    return response.output_text or ""

def request_json_payload(
    client: Any,
    model: str,
    prompt: str,
    progress_message: str,
    failure_context: str,
) -> dict[str, Any]:
    """Request a JSON object payload from the model."""
    payload = parse_json_response(
        _request_output_text(
            client=client,
            model=model,
            prompt=prompt,
            progress_message=progress_message,
            failure_context=failure_context,
        )
        or "{}"
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{failure_context}: returned a non-object response.")
    return payload

def request_text_payload(
    client: Any,
    model: str,
    prompt: str,
    progress_message: str,
    failure_context: str,
) -> str:
    """Request a non-empty text response from the model."""
    output = _request_output_text(
        client=client,
        model=model,
        prompt=prompt,
        progress_message=progress_message,
        failure_context=failure_context,
    ).strip()
    if not output:
        raise RuntimeError(f"{failure_context}: returned an empty message.")
    return output
