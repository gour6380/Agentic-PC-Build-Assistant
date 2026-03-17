from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "pc_part_dataset"
DOTENV_ALLOWED_KEYS = {
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
}

def _read_dotenv_file(dotenv_path: Path) -> dict[str, str]:
    """Read allowed OpenAI settings from the project `.env` file."""
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key not in DOTENV_ALLOWED_KEYS:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values

DOTENV_PATH = ROOT_DIR / ".env"
DOTENV_VALUES = _read_dotenv_file(DOTENV_PATH)

def get_config_value(name: str, default: str = "") -> str:
    """Return a config value from `.env` first, then the process environment."""
    if name in DOTENV_VALUES:
        value = DOTENV_VALUES[name]
        return value if value.strip() else default
    value = os.getenv(name, default)
    return value if str(value).strip() else default

DEFAULT_MODEL = get_config_value("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_FULL_BUILD_CATEGORIES = [
    "cpu",
    "motherboard",
    "memory",
    "internal-hard-drive",
    "video-card",
    "power-supply",
    "case",
    "cpu-cooler",
]
DEFAULT_OPTIONAL_CATEGORIES = [
    "monitor",
    "keyboard",
    "mouse",
    "headphones",
    "speakers",
    "webcam",
]

CSV_FILES = {
    "cpu": "cpu.csv",
    "motherboard": "motherboard.csv",
    "memory": "memory.csv",
    "internal-hard-drive": "internal-hard-drive.csv",
    "video-card": "video-card.csv",
    "power-supply": "power-supply.csv",
    "case": "case.csv",
    "cpu-cooler": "cpu-cooler.csv",
    "monitor": "monitor.csv",
    "keyboard": "keyboard.csv",
    "mouse": "mouse.csv",
    "headphones": "headphones.csv",
    "speakers": "speakers.csv",
    "webcam": "webcam.csv",
    "wireless-network-card": "wireless-network-card.csv",
    "wired-network-card": "wired-network-card.csv",
    "sound-card": "sound-card.csv",
    "ups": "ups.csv",
    "case-fan": "case-fan.csv",
    "case-accessory": "case-accessory.csv",
    "fan-controller": "fan-controller.csv",
    "external-hard-drive": "external-hard-drive.csv",
    "optical-drive": "optical-drive.csv",
    "os": "os.csv",
    "thermal-paste": "thermal-paste.csv",
}

SOCKET_RULES = {
    "AM5": {
        "cpu_patterns": [r"ryzen\s+[3579]\s+([789]\d{3}|8\d{3})", r"ryzen\s+7\s+9800x3d"],
        "microarchitectures": {"Zen 4", "Zen 5"},
    },
    "AM4": {
        "cpu_patterns": [r"ryzen\s+[3579]\s+[1-5]\d{3}", r"ryzen\s+\d+\s+5\d{3}"],
        "microarchitectures": {"Zen", "Zen+", "Zen 2", "Zen 3"},
    },
    "LGA1700": {
        "cpu_patterns": [r"i[3579]-1[234]\d{3}", r"core\s+i[3579]-1[234]\d{3}"],
        "microarchitectures": {"Alder Lake", "Raptor Lake", "Raptor Lake Refresh"},
    },
    "LGA1851": {
        "cpu_patterns": [r"core\s+ultra", r"ultra\s+[579]"],
        "microarchitectures": {"Arrow Lake"},
    },
}

CASE_SUPPORT = {
    "ATX": {"ATX", "Micro ATX", "Mini ITX"},
    "Micro ATX": {"Micro ATX", "Mini ITX"},
    "Mini ITX": {"Mini ITX"},
}

BUDGET_PROFILES = {
    "gaming": {
        "cpu": 0.20,
        "motherboard": 0.11,
        "memory": 0.08,
        "internal-hard-drive": 0.09,
        "video-card": 0.30,
        "power-supply": 0.08,
        "case": 0.07,
        "cpu-cooler": 0.07,
    },
    "workstation": {
        "cpu": 0.25,
        "motherboard": 0.12,
        "memory": 0.11,
        "internal-hard-drive": 0.10,
        "video-card": 0.20,
        "power-supply": 0.08,
        "case": 0.07,
        "cpu-cooler": 0.07,
    },
    "ai": {
        "cpu": 0.18,
        "motherboard": 0.10,
        "memory": 0.10,
        "internal-hard-drive": 0.09,
        "video-card": 0.34,
        "power-supply": 0.08,
        "case": 0.06,
        "cpu-cooler": 0.05,
    },
    "office": {
        "cpu": 0.22,
        "motherboard": 0.14,
        "memory": 0.10,
        "internal-hard-drive": 0.12,
        "video-card": 0.18,
        "power-supply": 0.08,
        "case": 0.09,
        "cpu-cooler": 0.07,
    },
    "balanced": {
        "cpu": 0.22,
        "motherboard": 0.12,
        "memory": 0.09,
        "internal-hard-drive": 0.10,
        "video-card": 0.24,
        "power-supply": 0.08,
        "case": 0.08,
        "cpu-cooler": 0.07,
    },
}

MINIMUM_VIABLE_PRICES = {
    "cpu": 80.0,
    "motherboard": 80.0,
    "memory": 40.0,
    "internal-hard-drive": 40.0,
    "video-card": 140.0,
    "power-supply": 50.0,
    "case": 40.0,
    "cpu-cooler": 20.0,
}
