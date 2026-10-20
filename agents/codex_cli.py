"""Codex CLI harness."""

from __future__ import annotations

from dynacomsec.agents.templated_cli import TemplatedCliHarness


class CodexCliHarness(TemplatedCliHarness):
    """Run Codex CLI as a DynaComSec full-agent target."""

    NAME = "codex-cli"
    COMMAND_ENV_VAR = "DYNACOMSEC_CODEX_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = (
        'response_file="/tmp/dynacomsec-codex-last-message-$$.txt"; '
        'event_file="/tmp/dynacomsec-codex-events-$$.jsonl"; '
        'mkdir -p "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        'cd "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        "codex exec --json --skip-git-repo-check --sandbox danger-full-access "
        '--model {model} --output-last-message "$response_file" '
        '< {instruction_file} > "$event_file" 2>&1; '
        "status=$?; "
        'cat "$response_file" 2>/dev/null || true; '
        'rm -f "$response_file" "$event_file"; '
        "exit $status"
    )
