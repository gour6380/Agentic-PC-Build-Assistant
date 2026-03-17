from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .config import CSV_FILES, DEFAULT_FULL_BUILD_CATEGORIES, MINIMUM_VIABLE_PRICES
from .models import PartCandidate
from .utils import extract_numbers, normalize_text, parse_comma_fields, safe_float

class CatalogTool:
    """Load and query the local CSV-backed PC parts catalog."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.tables: dict[str, pl.DataFrame] = {}
        self._minimum_prices: dict[str, float] = {}
        self._supported_categories: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load supported CSV tables and cache per-category minimum prices."""
        missing_files: list[str] = []
        for category, filename in CSV_FILES.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                if category in DEFAULT_FULL_BUILD_CATEGORIES:
                    missing_files.append(filename)
                continue
            frame = pl.read_csv(file_path, ignore_errors=True)
            frame = frame.with_columns(
                pl.col("name").cast(pl.Utf8).alias("name"),
                pl.col("name").cast(pl.Utf8).str.to_lowercase().alias("name_lower"),
                pl.col("price").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("price_num"),
            )
            cleaned = frame.filter(pl.col("price_num").is_not_null())
            minimum_price = cleaned.select(pl.col("price_num").min()).item() if cleaned.height else None
            if minimum_price is None:
                if category in DEFAULT_FULL_BUILD_CATEGORIES:
                    missing_files.append(filename)
                continue
            self.tables[category] = cleaned
            self._minimum_prices[category] = float(minimum_price)

        if missing_files:
            joined = ", ".join(missing_files)
            raise FileNotFoundError(f"Missing required dataset files: {joined}")
        self._supported_categories = set(self.tables)

    @property
    def supported_categories(self) -> set[str]:
        """Return the categories available in the loaded dataset."""
        return set(self._supported_categories)

    def minimum_price(self, category: str) -> float:
        """Return the cached minimum price for a category."""
        return self._minimum_prices[category]

    def minimum_full_build_cost(self, categories: list[str] | None = None) -> float:
        """Estimate the minimum viable cost for the requested build categories."""
        active = categories or DEFAULT_FULL_BUILD_CATEGORIES
        total = 0.0
        for category in active:
            if category not in self.tables:
                continue
            dataset_floor = self.minimum_price(category)
            viability_floor = MINIMUM_VIABLE_PRICES.get(category, dataset_floor)
            total += max(dataset_floor, viability_floor)
        return total

    def search_catalog(self, category: str, filters: dict[str, Any] | None = None, top_k: int = 5) -> list[PartCandidate]:
        """Filter and rank catalog candidates for one category."""
        if category not in self.tables:
            return []

        filters = filters or {}
        frame = self.tables[category]

        min_price = safe_float(filters.get("min_price"))
        max_price = safe_float(filters.get("max_price"))
        if min_price is not None:
            frame = frame.filter(pl.col("price_num") >= min_price)
        if max_price is not None:
            frame = frame.filter(pl.col("price_num") <= max_price)

        excluded_brands = [normalize_text(item) for item in filters.get("excluded_brands", [])]
        for brand in excluded_brands:
            frame = frame.filter(~pl.col("name_lower").str.contains(brand, literal=True))

        contains = [normalize_text(item) for item in filters.get("contains", [])]
        for token in contains:
            frame = frame.filter(pl.col("name_lower").str.contains(token, literal=True))

        if category == "motherboard" and filters.get("socket"):
            frame = frame.filter(pl.col("socket").cast(pl.Utf8) == filters["socket"])
        if category == "motherboard" and filters.get("form_factor"):
            frame = frame.filter(pl.col("form_factor").cast(pl.Utf8) == filters["form_factor"])
        records = frame.to_dicts()
        if not records:
            return []
        target_price = safe_float(filters.get("target_price"))
        preferred_brands = [normalize_text(item) for item in filters.get("preferred_brands", [])]
        use_case = normalize_text(filters.get("use_case"))
        scored = sorted(
            (
                (
                    self._score_candidate(
                        category=category,
                        row=row,
                        target_price=target_price,
                        preferred_brands=preferred_brands,
                        use_case=use_case,
                    ),
                    row,
                )
                for row in records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [self._row_to_candidate(category, row) for _, row in scored[:top_k]]

    def _score_candidate(
        self,
        category: str,
        row: dict[str, Any],
        target_price: float | None,
        preferred_brands: list[str],
        use_case: str,
    ) -> float:
        """Score a raw catalog row against budget and use-case preferences."""
        score = 0.0
        name = normalize_text(row.get("name"))
        price = safe_float(row.get("price_num")) or safe_float(row.get("price")) or 0.0

        if target_price:
            distance = abs(price - target_price) / max(target_price, 1.0)
            score += max(0.0, 35.0 - distance * 35.0)

        for brand in preferred_brands:
            if brand and brand in name:
                score += 18.0

        if category == "cpu":
            boost_clock = max(extract_numbers(row.get("boost_clock")) or [0.0])
            core_count = safe_float(row.get("core_count")) or 0.0
            score += core_count * 2.2 + boost_clock * 3.2
            if use_case == "gaming" and "x3d" in name:
                score += 25.0
            if use_case in {"workstation", "ai"}:
                score += core_count * 2.5
        elif category == "video-card":
            memory_gb = safe_float(row.get("memory")) or 0.0
            score += memory_gb * 3.0
            if use_case in {"gaming", "ai"}:
                score += memory_gb * 1.5
            if "5070" in name or "5080" in name or "5090" in name:
                score += 15.0
        elif category == "motherboard":
            max_memory = safe_float(row.get("max_memory")) or 0.0
            memory_slots = safe_float(row.get("memory_slots")) or 0.0
            score += max_memory * 0.1 + memory_slots * 3.0
        elif category == "memory":
            speed_values = parse_comma_fields(row.get("speed"))
            speed = max(speed_values) if speed_values else 0.0
            modules = parse_comma_fields(row.get("modules"))
            total_gb = (
                int(modules[0]) * int(modules[1])
                if len(modules) >= 2
                else (safe_float(row.get("price_per_gb")) or 0.0)
            )
            score += speed / 500.0 + total_gb * 1.5
        elif category in {"internal-hard-drive", "external-hard-drive"}:
            price_per_gb = safe_float(row.get("price_per_gb")) or 10.0
            capacity = safe_float(row.get("capacity")) or 0.0
            drive_type = normalize_text(row.get("type"))
            score += capacity / 250.0 - price_per_gb * 20.0
            if drive_type == "ssd":
                score += 12.0
        elif category == "power-supply":
            wattage = safe_float(row.get("wattage")) or 0.0
            efficiency = normalize_text(row.get("efficiency"))
            score += wattage / 100.0
            if efficiency == "gold":
                score += 8.0
            elif efficiency == "platinum":
                score += 10.0
        elif category == "case":
            if "tempered glass" in normalize_text(row.get("side_panel")):
                score += 3.0
        elif category == "cpu-cooler":
            size = max(extract_numbers(row.get("size")) or [0.0])
            score += size / 40.0
        elif category == "monitor":
            refresh = safe_float(row.get("refresh_rate")) or 0.0
            screen_size = safe_float(row.get("screen_size")) or 0.0
            score += refresh / 20.0 + screen_size / 4.0

        score -= price / 400.0
        return score

    def _row_to_candidate(self, category: str, row: dict[str, Any]) -> PartCandidate:
        """Convert a raw row dictionary into a normalized `PartCandidate`."""
        attributes = {key: value for key, value in row.items() if key not in {"name", "name_lower", "price_num"}}
        price = safe_float(row.get("price_num")) or safe_float(row.get("price")) or 0.0
        return PartCandidate(category=category, name=str(row["name"]), price=price, attributes=attributes)
