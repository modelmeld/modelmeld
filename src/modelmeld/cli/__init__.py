# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""ModelMeld customer-facing CLI.

Console entry: `modelmeld <subcommand> [options]`. Subcommands:
  - setup: one-command onboarding for a coding tool (Claude Code et al)
  - doctor: diagnose an existing setup (post-launch)
  - status: show recent routing + cost summary (post-launch)

The setup CLI exists because every byte of customer onboarding friction
is a customer we lost. Today's reverse-engineered footguns (CRLF in env
vars, terminal-wrap inserting whitespace, cache file format wrappers,
mode bits, Claude Code's discovery gate, BYOK header prefix) all live
here as hard-fought corner cases that the CLI handles automatically.
"""
from __future__ import annotations

import sys

from modelmeld.cli.doctor import run_doctor
from modelmeld.cli.setup import run_setup


def main(argv: list[str] | None = None) -> int:
    """Console entry point. Returns process exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="modelmeld",
        description="ModelMeld customer CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_p = subparsers.add_parser(
        "doctor",
        help="Diagnose an existing ModelMeld + coding-tool setup",
    )
    doctor_p.add_argument(
        "--tool",
        choices=["claude-code"],
        default="claude-code",
        help="Which coding tool's setup to diagnose (default: claude-code)",
    )

    setup_p = subparsers.add_parser(
        "setup",
        help="Configure a coding tool to route through ModelMeld",
    )
    setup_p.add_argument(
        "--tool",
        choices=["claude-code"],  # cursor, aider, cline, continue post-launch
        default="claude-code",
        help="Which coding tool to configure (default: claude-code)",
    )
    setup_p.add_argument(
        "--api-key",
        help="Your ModelMeld API key (gws_…). If omitted, prompted interactively.",
    )
    setup_p.add_argument(
        "--byok-anthropic",
        help="Your Anthropic API key for BYOK frontier routing. Optional.",
    )
    setup_p.add_argument(
        "--byok-openai",
        help="Your OpenAI API key for BYOK frontier routing. Optional.",
    )
    setup_p.add_argument(
        "--base-url",
        default="https://api.modelmeld.ai",
        help="ModelMeld gateway URL (default: hosted endpoint)",
    )
    setup_p.add_argument(
        "--allow-custom-host",
        action="store_true",
        help=(
            "Allow --base-url to point at hosts outside the default "
            "allowlist (api.modelmeld.ai, *.modelmeld.ai, localhost, "
            "RFC1918). REQUIRED for self-hosted gateways on custom hosts. "
            "HTTPS still required for non-loopback hosts."
        ),
    )
    setup_p.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Don't validate the setup with a live API call",
    )
    setup_p.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive — fail rather than prompt for missing values",
    )

    args = parser.parse_args(argv)

    if args.command == "setup":
        return run_setup(args)
    if args.command == "doctor":
        return run_doctor(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
