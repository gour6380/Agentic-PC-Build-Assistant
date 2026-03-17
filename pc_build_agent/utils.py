from __future__ import annotations

import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from pprint import pformat
from typing import Any, Iterable

ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "planner": "\033[96m",
    "currency": "\033[92m",
    "assistant": "\033[95m",
    "user": "\033[94m",
    "system": "\033[93m",
    "trace": "\033[90m",
    "success": "\033[92m",
    "error": "\033[91m",
}

def safe_float(value: Any) -> float | None:
    """Convert a loose numeric value into a float when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None

def extract_numbers(value: Any) -> list[float]:
    """Extract all numeric substrings from a value as floats."""
    text = str(value or "")
    return [float(match.replace(",", "")) for match in re.findall(r"\d[\d,]*\.?\d*", text)]

def parse_comma_fields(value: Any) -> list[float]:
    """Parse comma-separated numeric fields into a list of floats."""
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    parsed: list[float] = []
    for part in parts:
        number = safe_float(part)
        if number is not None:
            parsed.append(number)
    return parsed

def normalize_text(value: Any) -> str:
    """Normalize text for case-insensitive matching."""
    return str(value or "").strip().lower()

def words_in_text(words: list[str], text: str) -> bool:
    """Return whether any candidate word appears in the target text."""
    text_lower = normalize_text(text)
    return any(word in text_lower for word in words)

def parse_json_response(value: Any) -> Any:
    """Parse JSON output, including fenced code-block responses."""
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text, count=1)
        text = re.sub(r"\s*```$", "", text, count=1)
        text = text.strip()
    return json.loads(text)

def pretty_format(value: Any) -> str:
    """Return a stable pretty-printed representation for trace logs."""
    return pformat(value, sort_dicts=False, width=100)

def unique_strings(values: Iterable[str]) -> list[str]:
    """Return unique non-empty strings while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result

def terminal_supports_color(stream: Any | None = None) -> bool:
    """Return whether ANSI color output is safe for the target stream."""
    target = stream or sys.stdout
    return bool(
        hasattr(target, "isatty")
        and target.isatty()
        and os.getenv("NO_COLOR") is None
        and os.getenv("TERM", "").lower() != "dumb"
    )

def colorize(text: str, style: str, stream: Any | None = None) -> str:
    """Apply ANSI color styling when the terminal supports it."""
    if not terminal_supports_color(stream):
        return text
    code = ANSI_COLORS.get(style)
    if not code:
        return text
    return f"{code}{text}{ANSI_RESET}"

def infer_message_style(message: str) -> str:
    """Infer a log style label from a trace message."""
    text = normalize_text(message)
    if "planner agent" in text or "planner state" in text:
        return "planner"
    if "currency agent" in text or "currency conversion" in text:
        return "currency"
    if text.startswith("user input"):
        return "user"
    if "response generation failed" in text or "failed" in text:
        return "error"
    if "attempt " in text or "recommendation assembly" in text or "missing fields" in text or "unsupported categories" in text:
        return "system"
    return "trace"

class ConsoleProgress:
    """Render a lightweight spinner for model and tool progress."""

    def __init__(self, message: str, interval_seconds: float = 0.2) -> None:
        self.message = message
        self.interval_seconds = interval_seconds
        self.style = infer_message_style(message)
        self._frames = ("|", "/", "-", "\\")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_width = 0

    def start(self) -> None:
        """Start the background spinner thread if it is not already running."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, status: str) -> None:
        """Stop the spinner and print a final status label."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        plain_line = f"\r{self.message}... {status}"
        status_style = "success" if status == "done" else "error"
        display_line = (
            f"\r{colorize(self.message, self.style)}... "
            f"{colorize(status, status_style)}"
        )
        padding = max(0, self._last_width - len(plain_line))
        sys.stdout.write(display_line + (" " * padding) + "\n")
        sys.stdout.flush()
        self._thread = None

    def _run(self) -> None:
        """Continuously render spinner frames until stopped."""
        frame_index = 0
        while not self._stop_event.wait(self.interval_seconds):
            frame = self._frames[frame_index % len(self._frames)]
            plain_line = f"\r{self.message}... {frame}"
            display_line = (
                f"\r{colorize(self.message, self.style)}... "
                f"{colorize(frame, self.style)}"
            )
            padding = max(0, self._last_width - len(plain_line))
            sys.stdout.write(display_line + (" " * padding))
            sys.stdout.flush()
            self._last_width = len(plain_line)
            frame_index += 1

@contextmanager
def progress_step(message: str) -> Any:
    """Wrap a block in a console progress spinner."""
    progress = ConsoleProgress(message)
    progress.start()
    try:
        yield
    except Exception:
        progress.stop("failed")
        raise
    else:
        progress.stop("done")
