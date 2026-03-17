from __future__ import annotations

from dataclasses import asdict

from .catalog import CatalogTool
from .currency_agent import CurrencyAgent, format_money
from .llm import LLMHelper
from .models import BuildProposal, CurrencyContext, SessionState, ValidationReport
from .parsing import extract_requirements
from .planner import assemble_build, planning_budget_max, validate_build
from .utils import colorize, infer_message_style, normalize_text, pretty_format

def log_step(state: SessionState, message: str, payload: object | None = None) -> None:
    """Append a formatted trace entry to the session log."""
    if payload is None:
        state.logs.append(message)
        return
    state.logs.append(f"{message}\n{pretty_format(payload)}")

def next_question(state: SessionState) -> str | None:
    """Return the next required follow-up question for the current state."""
    state.missing_fields.clear()
    if state.intent is None:
        state.missing_fields.append("intent")
        return "Do you want a full custom PC build or a recommendation for one specific part?"
    if state.intent == "full_build" and state.budget_target is None and state.budget_max is None:
        state.missing_fields.append("budget")
        return "What budget should I use for the full build? If you have one exact number, I will treat it as a +/- 5% range."
    if state.intent == "full_build" and state.use_case is None:
        state.missing_fields.append("use_case")
        return "What is the main use case for this build, such as gaming, workstation work, AI, streaming, or office?"
    if state.intent == "single_part" and not state.requested_categories:
        state.missing_fields.append("requested_categories")
        return "Which part do you want recommended, such as CPU, GPU, monitor, or storage?"
    return None

def _low_budget_message(state: SessionState, minimum_cost_usd: float) -> str:
    """Build the low-budget response for infeasible full-build requests."""
    message = (
        f"That budget is too low for a dataset-backed full build. The current minimum across the required core "
        f"categories is about {format_money(minimum_cost_usd, 'USD')}."
    )
    if state.display_currency != "USD" and state.exchange_rate:
        converted_floor = round(minimum_cost_usd * state.exchange_rate, 2)
        message += f" That is approximately {format_money(converted_floor, state.display_currency)} at the current reference rate."
    message += " Increase the budget or ask for a reduced-scope part recommendation."
    return message

def _final_response(
    state: SessionState,
    build: BuildProposal,
    report: ValidationReport,
    currency_agent: CurrencyAgent,
    currency_context: CurrencyContext,
    llm_helper: LLMHelper,
    failure_prefix: str,
) -> str:
    """Generate the final response or return a model-failure message."""
    display_payload = currency_agent.build_display_payload(state, build, report, currency_context)
    try:
        return llm_helper.explain(state, build, report, currency_context, display_payload).strip()
    except RuntimeError as exc:
        log_step(state, f"Response generation failed: {exc}")
        return f"{failure_prefix}: {exc}"

def run_turn(
    user_message: str,
    state: SessionState,
    catalog: CatalogTool,
    llm_helper: LLMHelper,
    currency_agent: CurrencyAgent,
) -> str:
    """Process one user turn through extraction, planning, validation, and response generation."""
    state.conversation_history.append(user_message)
    log_step(state, f"User input received: {user_message}")
    try:
        state = extract_requirements(user_message, state, llm_helper)
    except RuntimeError as exc:
        log_step(state, f"Requirement extraction failed: {exc}")
        return f"I could not continue because the model call failed during requirement extraction: {exc}"
    log_step(
        state,
        "Planner state after requirement extraction:",
        {
            "intent": state.intent,
            "use_case": state.use_case,
            "budget_target": state.budget_target,
            "budget_min": state.budget_min,
            "budget_max": state.budget_max,
            "budget_currency": state.budget_currency,
            "display_currency": state.display_currency,
            "requested_categories": state.requested_categories,
            "unsupported_categories": state.unsupported_categories,
            "preferences": state.preferences,
        },
    )

    if state.unsupported_categories and not state.requested_categories and state.intent != "full_build":
        unsupported = ", ".join(state.unsupported_categories)
        log_step(state, f"Unsupported categories requested: {unsupported}")
        return (
            f"I cannot recommend {unsupported} because those items are not present in the local dataset. "
            "Please ask for a supported PC component such as a CPU, GPU, storage drive, monitor, or a full build."
        )

    display_only_reuse = state.current_build is not None and state.conversion_mode == "display_only"
    if not display_only_reuse:
        question = next_question(state)
        if question:
            log_step(state, f"Missing fields detected: {', '.join(state.missing_fields)}")
            return question

    try:
        currency_context = currency_agent.resolve(user_message, state)
    except RuntimeError as exc:
        log_step(state, f"Currency conversion failed: {exc}")
        return f"I could not continue because currency conversion failed: {exc}"
    log_step(
        state,
        "Currency agent result:",
        asdict(currency_context),
    )

    if currency_context.reuse_existing_build and state.current_build is not None:
        log_step(state, "Reusing existing build for display-only currency conversion.")
        report = validate_build(state.current_build, state)
        return _final_response(
            state=state,
            build=state.current_build,
            report=report,
            currency_agent=currency_agent,
            currency_context=currency_context,
            llm_helper=llm_helper,
            failure_prefix=(
                "I converted the existing build context, but the model call failed while generating the final response"
            ),
        )

    if state.intent == "full_build" and planning_budget_max(state) is not None:
        minimum_cost = catalog.minimum_full_build_cost()
        if float(planning_budget_max(state)) < minimum_cost:
            log_step(state, f"Budget below minimum dataset-backed build floor: ${minimum_cost:.2f}")
            return _low_budget_message(state, minimum_cost)

    log_step(state, "Starting recommendation assembly.")
    best_build: BuildProposal | None = None
    best_report: ValidationReport | None = None
    for attempt in range(1, 4):
        build = assemble_build(catalog, state, attempt=attempt)
        report = validate_build(build, state)
        state.revision_history.append(
            {
                "attempt": attempt,
                "build_total": build.total_price,
                "issues": report.issues,
                "warnings": report.compatibility_warnings,
            }
        )
        log_step(
            state,
            f"Attempt {attempt} review:",
            {
                "build_total_usd": round(build.total_price, 2),
                "passed": report.passed,
                "issue_count": len(report.issues),
                "issues": report.issues,
                "compatibility_warnings": report.compatibility_warnings,
            },
        )
        best_build = build
        best_report = report
        if report.passed:
            break

    if best_build is None or best_report is None:
        return "I could not assemble a recommendation from the dataset."

    state.current_build = best_build
    return _final_response(
        state=state,
        build=best_build,
        report=best_report,
        currency_agent=currency_agent,
        currency_context=currency_context,
        llm_helper=llm_helper,
        failure_prefix="I assembled a recommendation, but the model call failed while generating the final response",
    )

def print_logs(state: SessionState) -> None:
    """Print the collected agent trace for the current session state."""
    print(colorize("Agent trace:", "trace"))
    for entry in state.logs:
        lines = entry.splitlines()
        if not lines:
            continue
        style = infer_message_style(lines[0])
        print(f"- {colorize(lines[0], style)}")
        for line in lines[1:]:
            print(f"  {colorize(line, 'trace')}")

def demo_messages() -> list[str]:
    """Return the representative prompts used by demo mode."""
    return [
        "Build me a gaming PC around $1500 with WiFi and 32 GB RAM.",
        "I need a custom PC for video editing but I forgot the budget.",
        "Build me an office PC with a budget of 900 dollars.",
        "Recommend a GPU under $400 for 1440p gaming.",
        "Can you build me a full PC for $250?",
        "I need a gaming build around $2800 and also include a monitor.",
        "Recommend a printer for my setup.",
    ]

def interactive_loop(catalog: CatalogTool, llm_helper: LLMHelper, currency_agent: CurrencyAgent) -> None:
    """Run the interactive terminal loop for the assistant."""
    state = SessionState()
    print(colorize("PC Build Assistant", "assistant"))
    print("Type 'quit' to exit.\n")
    while True:
        user_message = input(colorize("You: ", "user")).strip()
        if not user_message:
            continue
        if normalize_text(user_message) in {"quit", "exit"}:
            break
        response = run_turn(user_message, state, catalog, llm_helper, currency_agent)
        print(f"\n{colorize('Assistant:', 'assistant')}\n{response}\n")
        print_logs(state)
        print()

def run_demo(catalog: CatalogTool, llm_helper: LLMHelper, currency_agent: CurrencyAgent) -> None:
    """Run the built-in demo scenarios and print their traces."""
    for index, message in enumerate(demo_messages(), start=1):
        state = SessionState()
        print("=" * 80)
        print(colorize(f"Demo {index}", "system"))
        print(f"{colorize('User:', 'user')} {message}")
        response = run_turn(message, state, catalog, llm_helper, currency_agent)
        print(f"\n{colorize('Assistant:', 'assistant')}")
        print(response)
        print()
        print_logs(state)
        print()
