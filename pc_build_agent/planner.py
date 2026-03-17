from __future__ import annotations

import re
from collections.abc import Callable

from .catalog import CatalogTool
from .config import (
    BUDGET_PROFILES,
    CASE_SUPPORT,
    CSV_FILES,
    DEFAULT_FULL_BUILD_CATEGORIES,
    DEFAULT_OPTIONAL_CATEGORIES,
    SOCKET_RULES,
)
from .models import BuildProposal, PartCandidate, SessionState, ValidationReport
from .utils import extract_numbers, normalize_text, parse_comma_fields, safe_float

def infer_cpu_socket(cpu: PartCandidate) -> str | None:
    """Infer a CPU socket from the dataset fields and model name patterns."""
    name = normalize_text(cpu.name)
    microarchitecture = str(cpu.attributes.get("microarchitecture") or "")
    for socket, rule in SOCKET_RULES.items():
        if microarchitecture in rule["microarchitectures"]:
            return socket
        for pattern in rule["cpu_patterns"]:
            if re.search(pattern, name):
                return socket
    return None

def estimate_gpu_tier(gpu: PartCandidate | None) -> str:
    """Classify a GPU into a simple entry, mid, or high tier bucket."""
    if gpu is None:
        return "entry"
    memory_gb = safe_float(gpu.attributes.get("memory")) or 0.0
    chipset = normalize_text(gpu.attributes.get("chipset") or gpu.name)
    if any(token in chipset for token in ["5090", "5080", "5070 ti", "4080", "4090"]) or memory_gb >= 16:
        return "high"
    if any(token in chipset for token in ["5070", "4070", "4060 ti", "7800", "7900"]) or memory_gb >= 12:
        return "mid"
    return "entry"

def recommended_psu_wattage(cpu: PartCandidate | None, gpu: PartCandidate | None) -> int:
    """Estimate a conservative PSU target from the selected CPU and GPU."""
    cpu_tdp = safe_float(cpu.attributes.get("tdp")) if cpu else 0.0
    gpu_tier = estimate_gpu_tier(gpu)
    if gpu_tier == "high":
        return 850
    if gpu_tier == "mid":
        return 750
    if (cpu_tdp or 0.0) >= 120:
        return 650
    return 550

def parse_memory_modules(candidate: PartCandidate) -> tuple[int, int]:
    """Parse a memory kit into module count and total capacity."""
    numbers = parse_comma_fields(candidate.attributes.get("modules"))
    if len(numbers) >= 2:
        module_count = int(numbers[0])
        per_module_gb = int(numbers[1])
        total_gb = module_count * per_module_gb
        return module_count, total_gb
    return 1, 0

def normalize_board_form_factor(value: str | None) -> str | None:
    """Normalize motherboard form-factor labels into common names."""
    text = normalize_text(value)
    if not text:
        return None
    if "mini" in text and "itx" in text:
        return "Mini ITX"
    if "micro" in text and "atx" in text:
        return "Micro ATX"
    if text == "atx":
        return "ATX"
    return str(value)

def case_supports_board(case_candidate: PartCandidate, board_form_factor: str | None) -> bool | None:
    """Return whether the case metadata appears to support the board size."""
    if not board_form_factor:
        return None
    case_type = normalize_text(case_candidate.attributes.get("type"))
    if not case_type:
        return None
    if "mini itx" in case_type or "itx" in case_type:
        supported = CASE_SUPPORT["Mini ITX"]
    elif "microatx" in case_type or "micro atx" in case_type:
        supported = CASE_SUPPORT["Micro ATX"]
    elif "atx" in case_type or "mid tower" in case_type or "full tower" in case_type:
        supported = CASE_SUPPORT["ATX"]
    else:
        return None
    return board_form_factor in supported

def choose_use_case(state: SessionState) -> str:
    """Return the active use case or the balanced default profile."""
    return state.use_case or "balanced"

def planning_budget_target(state: SessionState) -> float:
    """Return the active planning target budget in USD."""
    return state.normalized_budget_target_usd or state.budget_target or 0.0

def planning_budget_min(state: SessionState) -> float | None:
    """Return the active planning minimum budget in USD."""
    return state.normalized_budget_min_usd if state.normalized_budget_min_usd is not None else state.budget_min

def planning_budget_max(state: SessionState) -> float | None:
    """Return the active planning maximum budget in USD."""
    return state.normalized_budget_max_usd if state.normalized_budget_max_usd is not None else state.budget_max

def budget_slice(total_budget: float, category: str, use_case: str, attempt: int) -> float:
    """Allocate a per-category budget slice for the current attempt."""
    profile = BUDGET_PROFILES.get(use_case, BUDGET_PROFILES["balanced"])
    attempt_scale = {1: 1.0, 2: 0.92, 3: 0.84}.get(attempt, 0.80)
    return round(total_budget * profile[category] * attempt_scale, 2)

def _base_search_filters(state: SessionState, budget: float, minimum_price: float) -> dict[str, object]:
    """Build the common search-filter payload shared across pickers."""
    return {
        "max_price": max(budget, minimum_price),
        "target_price": budget,
        "preferred_brands": state.preferences.get("preferred_brands", []),
        "excluded_brands": state.preferences.get("excluded_brands", []),
        "use_case": choose_use_case(state),
    }

def _search_with_budget(
    catalog: CatalogTool,
    state: SessionState,
    category: str,
    budget: float,
    minimum_price: float,
    top_k: int,
    **extra_filters: object,
) -> list[PartCandidate]:
    """Run a catalog search with shared budget and preference filters."""
    filters = _base_search_filters(state, budget, minimum_price)
    filters.update(extra_filters)
    return catalog.search_catalog(category, filters, top_k=top_k)

def _top_candidate(candidates: list[PartCandidate]) -> PartCandidate | None:
    """Return the highest-ranked candidate if one exists."""
    return candidates[0] if candidates else None

def _pick_ranked_candidate(
    raw_candidates: list[PartCandidate],
    scorer: Callable[[PartCandidate], float | None],
) -> PartCandidate | None:
    """Rank candidates with a scorer and return the best surviving option."""
    ranked: list[tuple[float, PartCandidate]] = []
    for candidate in raw_candidates:
        score = scorer(candidate)
        if score is None:
            continue
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else _top_candidate(raw_candidates)

def pick_cpu(catalog: CatalogTool, state: SessionState, attempt: int) -> PartCandidate | None:
    """Pick a CPU candidate for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "cpu", choose_use_case(state), attempt)
    return _top_candidate(_search_with_budget(catalog, state, "cpu", budget, 85, top_k=8))

def pick_gpu(catalog: CatalogTool, state: SessionState, attempt: int) -> PartCandidate | None:
    """Pick a GPU candidate for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "video-card", choose_use_case(state), attempt)
    return _top_candidate(_search_with_budget(catalog, state, "video-card", budget, 140, top_k=10))

def pick_motherboard(catalog: CatalogTool, state: SessionState, cpu: PartCandidate | None, attempt: int) -> PartCandidate | None:
    """Pick a motherboard candidate that best fits the CPU and preferences."""
    budget = budget_slice(planning_budget_target(state), "motherboard", choose_use_case(state), attempt)
    socket = infer_cpu_socket(cpu) if cpu else None
    form_factor = state.preferences.get("form_factor")
    contains = ["wifi"] if state.preferences.get("needs_wifi") else []
    candidates = _search_with_budget(
        catalog,
        state,
        "motherboard",
        budget,
        90,
        top_k=10,
        socket=socket,
        form_factor=form_factor,
        contains=contains,
    )
    if candidates:
        return candidates[0]
    return _top_candidate(
        _search_with_budget(
            catalog,
            state,
            "motherboard",
            budget,
            100,
            top_k=10,
            max_price=max(budget * 1.2, 100),
            socket=socket,
        )
    )

def _score_memory_candidate(
    candidate: PartCandidate,
    desired_gb: int,
    board_slots: int,
    board_max: int,
) -> float | None:
    """Score a memory candidate against the desired capacity and board limits."""
    module_count, total_gb = parse_memory_modules(candidate)
    if module_count > board_slots or total_gb > board_max:
        return None
    score = 100 - abs(total_gb - desired_gb)
    if total_gb >= desired_gb:
        score += 25
    else:
        score -= 120
    speed = max(parse_comma_fields(candidate.attributes.get("speed")) or [0.0])
    return score + speed / 300.0

def pick_memory(catalog: CatalogTool, state: SessionState, board: PartCandidate | None, attempt: int) -> PartCandidate | None:
    """Pick a memory kit that fits the board and target capacity."""
    budget = budget_slice(planning_budget_target(state), "memory", choose_use_case(state), attempt)
    desired_gb = state.preferences.get("memory_target_gb")
    if desired_gb is None:
        desired_gb = 64 if choose_use_case(state) in {"workstation", "ai"} and planning_budget_target(state) >= 1800 else 32

    raw_candidates = _search_with_budget(catalog, state, "memory", budget, 45, top_k=30)
    board_slots = int(safe_float(board.attributes.get("memory_slots")) or 4) if board else 4
    board_max = int(safe_float(board.attributes.get("max_memory")) or 128) if board else 128
    return _pick_ranked_candidate(
        raw_candidates,
        lambda candidate: _score_memory_candidate(candidate, desired_gb, board_slots, board_max),
    )

def _score_storage_candidate(candidate: PartCandidate, desired_gb: int) -> float:
    """Score a storage candidate against the desired capacity target."""
    capacity = int(safe_float(candidate.attributes.get("capacity")) or 0)
    score = 90 - abs(capacity - desired_gb) / 50.0
    if normalize_text(candidate.attributes.get("type")) == "ssd":
        score += 10
    return score

def pick_storage(catalog: CatalogTool, state: SessionState, attempt: int) -> PartCandidate | None:
    """Pick an internal storage device for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "internal-hard-drive", choose_use_case(state), attempt)
    desired_gb = state.preferences.get("storage_target_gb")
    if desired_gb is None:
        desired_gb = 2000 if choose_use_case(state) in {"workstation", "ai"} else 1000

    raw_candidates = _search_with_budget(catalog, state, "internal-hard-drive", budget, 50, top_k=20)
    return _pick_ranked_candidate(
        raw_candidates,
        lambda candidate: _score_storage_candidate(candidate, desired_gb),
    )

def _score_psu_candidate(candidate: PartCandidate, wattage: int) -> float | None:
    """Score a power supply candidate against the estimated wattage need."""
    candidate_wattage = safe_float(candidate.attributes.get("wattage")) or 0.0
    if candidate_wattage < wattage:
        return None
    efficiency = normalize_text(candidate.attributes.get("efficiency"))
    score = 100 - abs(candidate_wattage - wattage) / 10.0
    if efficiency == "gold":
        score += 10
    elif efficiency == "platinum":
        score += 12
    return score

def pick_psu(catalog: CatalogTool, state: SessionState, cpu: PartCandidate | None, gpu: PartCandidate | None, attempt: int) -> PartCandidate | None:
    """Pick a power supply for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "power-supply", choose_use_case(state), attempt)
    wattage = recommended_psu_wattage(cpu, gpu)
    raw_candidates = _search_with_budget(catalog, state, "power-supply", budget, 55, top_k=25)
    return _pick_ranked_candidate(
        raw_candidates,
        lambda candidate: _score_psu_candidate(candidate, wattage),
    )

def _score_case_candidate(candidate: PartCandidate, board_form_factor: str | None) -> float | None:
    """Score a case candidate against board support and simple quality hints."""
    supported = case_supports_board(candidate, board_form_factor)
    if supported is False:
        return None
    score = 100.0
    if "tempered glass" in normalize_text(candidate.attributes.get("side_panel")):
        score += 5.0
    return score

def pick_case(catalog: CatalogTool, state: SessionState, board: PartCandidate | None, attempt: int) -> PartCandidate | None:
    """Pick a case for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "case", choose_use_case(state), attempt)
    board_form_factor = normalize_board_form_factor(board.attributes.get("form_factor")) if board else state.preferences.get("form_factor")
    raw_candidates = _search_with_budget(catalog, state, "case", budget, 45, top_k=25)
    return _pick_ranked_candidate(
        raw_candidates,
        lambda candidate: _score_case_candidate(candidate, board_form_factor),
    )

def _score_cooler_candidate(candidate: PartCandidate, cpu_tdp: float | None) -> float:
    """Score a CPU cooler against radiator size and thermal demand."""
    size = max(extract_numbers(candidate.attributes.get("size")) or [120.0])
    score = 70 + size / 10.0
    if (cpu_tdp or 0.0) >= 120 and size >= 240:
        score += 10.0
    return score

def pick_cooler(catalog: CatalogTool, state: SessionState, cpu: PartCandidate | None, attempt: int) -> PartCandidate | None:
    """Pick a CPU cooler for the current build attempt."""
    budget = budget_slice(planning_budget_target(state), "cpu-cooler", choose_use_case(state), attempt)
    cpu_tdp = safe_float(cpu.attributes.get("tdp")) if cpu else 65.0
    raw_candidates = _search_with_budget(catalog, state, "cpu-cooler", budget, 20, top_k=20)
    return _pick_ranked_candidate(
        raw_candidates,
        lambda candidate: _score_cooler_candidate(candidate, cpu_tdp),
    )

def pick_generic_part(catalog: CatalogTool, state: SessionState, category: str) -> PartCandidate | None:
    """Pick a generic single-part or optional-category recommendation."""
    target = planning_budget_target(state) or planning_budget_max(state) or 0.0
    per_part_budget = target if state.intent == "single_part" else max((target * 0.10), 35.0)
    return _top_candidate(_search_with_budget(catalog, state, category, per_part_budget, 25, top_k=10))

def assemble_build(catalog: CatalogTool, state: SessionState, attempt: int = 1) -> BuildProposal:
    """Assemble a full-build or single-part proposal from the catalog."""
    proposal = BuildProposal(attempt=attempt)

    if state.intent == "single_part":
        if not state.requested_categories:
            proposal.unmet_constraints.append("No part category was identified for the single-part request.")
            return proposal
        category = state.requested_categories[0]
        candidate = pick_generic_part(catalog, state, category)
        if not candidate:
            proposal.unmet_constraints.append(f"No suitable {category} candidate was found in the dataset.")
            return proposal
        proposal.selected_parts[category] = candidate
        proposal.total_price = candidate.price
        return proposal

    minimum_cost = catalog.minimum_full_build_cost()
    if planning_budget_max(state) is not None and planning_budget_max(state) < minimum_cost:
        proposal.unmet_constraints.append(
            f"Budget is too low for the minimum dataset-backed full build. Estimated floor is ${minimum_cost:.2f}."
        )
        return proposal

    cpu = pick_cpu(catalog, state, attempt)
    gpu = pick_gpu(catalog, state, attempt)
    board = pick_motherboard(catalog, state, cpu, attempt)
    memory = pick_memory(catalog, state, board, attempt)
    storage = pick_storage(catalog, state, attempt)
    psu = pick_psu(catalog, state, cpu, gpu, attempt)
    case = pick_case(catalog, state, board, attempt)
    cooler = pick_cooler(catalog, state, cpu, attempt)

    for category, candidate in [
        ("cpu", cpu),
        ("motherboard", board),
        ("memory", memory),
        ("internal-hard-drive", storage),
        ("video-card", gpu),
        ("power-supply", psu),
        ("case", case),
        ("cpu-cooler", cooler),
    ]:
        if candidate is None:
            proposal.unmet_constraints.append(f"No {category} candidate was found for attempt {attempt}.")
        else:
            proposal.selected_parts[category] = candidate

    extra_categories = [
        category
        for category in state.requested_categories
        if category not in DEFAULT_FULL_BUILD_CATEGORIES and category in catalog.supported_categories
    ]
    if state.preferences.get("include_peripherals"):
        for category in DEFAULT_OPTIONAL_CATEGORIES:
            if category in state.requested_categories and category not in extra_categories:
                extra_categories.append(category)
    for category in extra_categories:
        candidate = pick_generic_part(catalog, state, category)
        if candidate:
            proposal.selected_parts[category] = candidate
        else:
            proposal.warnings.append(f"Optional category {category} could not be matched from the dataset.")

    proposal.total_price = round(sum(candidate.price for candidate in proposal.selected_parts.values()), 2)
    if planning_budget_max(state) is not None and proposal.total_price > planning_budget_max(state):
        proposal.warnings.append(
            f"Attempt {attempt} is over the current maximum budget by ${proposal.total_price - float(planning_budget_max(state)):.2f}."
        )
    if planning_budget_min(state) is not None and proposal.total_price < planning_budget_min(state):
        proposal.warnings.append(
            f"Attempt {attempt} spends ${float(planning_budget_min(state)) - proposal.total_price:.2f} below the minimum budget band."
        )
    return proposal

def validate_build(build: BuildProposal, state: SessionState) -> ValidationReport:
    """Validate budget, coverage, and soft compatibility checks for a proposal."""
    issues = list(build.unmet_constraints)
    compatibility_warnings = list(build.warnings)

    required_categories = state.requested_categories[:] if state.intent == "single_part" else DEFAULT_FULL_BUILD_CATEGORIES[:]
    if state.intent == "full_build":
        for category in state.requested_categories:
            if category not in required_categories and category not in state.unsupported_categories:
                required_categories.append(category)

    coverage_ok = all(category in build.selected_parts for category in required_categories if category in CSV_FILES)
    if not coverage_ok:
        missing = [category for category in required_categories if category in CSV_FILES and category not in build.selected_parts]
        issues.append(f"Missing required categories: {', '.join(missing)}.")

    budget_ok = True
    if planning_budget_max(state) is not None and build.total_price > planning_budget_max(state):
        budget_ok = False
        issues.append(f"Total price ${build.total_price:.2f} exceeds the maximum budget ${float(planning_budget_max(state)):.2f}.")

    if state.unsupported_categories:
        issues.append(f"Unsupported request items: {', '.join(state.unsupported_categories)}.")

    board = build.selected_parts.get("motherboard")
    memory = build.selected_parts.get("memory")
    cpu = build.selected_parts.get("cpu")
    case_candidate = build.selected_parts.get("case")

    if board and memory:
        board_slots = int(safe_float(board.attributes.get("memory_slots")) or 0)
        board_max_memory = int(safe_float(board.attributes.get("max_memory")) or 0)
        module_count, total_memory = parse_memory_modules(memory)
        if board_slots and module_count > board_slots:
            issues.append(
                f"Memory kit uses {module_count} modules but the motherboard only has {board_slots} slots."
            )
        if board_max_memory and total_memory > board_max_memory:
            issues.append(
                f"Memory capacity {total_memory} GB exceeds the motherboard maximum of {board_max_memory} GB."
            )

    if cpu and board:
        inferred_socket = infer_cpu_socket(cpu)
        board_socket = str(board.attributes.get("socket") or "")
        if inferred_socket and board_socket and inferred_socket != board_socket:
            compatibility_warnings.append(
                f"CPU socket is inferred as {inferred_socket} but the selected motherboard uses {board_socket}. "
                "CPU socket inference is a soft check because the CPU dataset does not include socket values."
            )
        if inferred_socket is None:
            compatibility_warnings.append(
                "CPU socket could not be inferred confidently from the dataset, so CPU/motherboard compatibility is not fully verified."
            )

    if case_candidate and board:
        board_form_factor = normalize_board_form_factor(board.attributes.get("form_factor"))
        supported = case_supports_board(case_candidate, board_form_factor)
        if supported is False:
            compatibility_warnings.append(
                "Case and motherboard form factor look mismatched based on the dataset labels. "
                "This remains a soft check."
            )
        elif supported is None:
            compatibility_warnings.append(
                "Case form-factor compatibility could not be fully checked from the available case metadata."
            )

    if state.intent == "full_build":
        compatibility_warnings.extend(
            [
                "GPU length versus case clearance is not verified because the dataset does not expose case GPU clearance.",
                "CPU cooler clearance is not verified because the dataset does not expose case or RAM height limits.",
                "Storage lane availability and full PSU sizing are estimated rather than fully validated from the dataset.",
            ]
        )

    passed = budget_ok and coverage_ok and not issues
    return ValidationReport(
        passed=passed,
        issues=issues,
        budget_ok=budget_ok,
        coverage_ok=coverage_ok,
        compatibility_warnings=compatibility_warnings,
    )
