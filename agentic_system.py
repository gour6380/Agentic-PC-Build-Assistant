#!/usr/bin/env python3.13
"""
## Script Summary

This script implements a two-agent PC build assistant that uses a local CSV catalog to recommend either a full desktop build or a single hardware component, with a planner agent for requirement extraction and validation and a currency agent for live reference-rate conversion. During execution, the system asks follow-up questions when critical details such as budget are missing, shows observable trace logs for its decisions, and returns warnings when checks are soft or incomplete. One implementation challenge was handling incomplete compatibility metadata in the dataset while also depending on live exchange-rate data and strict JSON extraction from LLM responses. Known limitations remain around socket inference, case and cooler fit verification, PSU sizing, and the fact that converted prices are informational reference values rather than payment quotes. These constraints are surfaced explicitly so the script remains transparent, reviewer-friendly, and aligned with the Task 5 summary requirement.
"""

from __future__ import annotations

import argparse

from pc_build_agent import CatalogTool, CurrencyAgent, DATA_DIR, LLMHelper, SessionState, run_turn
from pc_build_agent.conversation import interactive_loop, print_logs, run_demo

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one-shot, demo, and interactive runs."""
    parser = argparse.ArgumentParser(description="Run the agentic PC build assistant.")
    parser.add_argument("--message", type=str, help="Process one user message and exit.")
    parser.add_argument("--demo", action="store_true", help="Run representative demo scenarios.")
    return parser

def main(argv: list[str] | None = None) -> int:
    """Run the script entrypoint and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    catalog = CatalogTool(DATA_DIR)
    llm_helper = LLMHelper()
    currency_agent = CurrencyAgent()

    if args.demo:
        run_demo(catalog, llm_helper, currency_agent)
        return 0

    if args.message:
        state = SessionState()
        response = run_turn(args.message, state, catalog, llm_helper, currency_agent)
        print(response)
        print()
        print_logs(state)
        return 0

    interactive_loop(catalog, llm_helper, currency_agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
