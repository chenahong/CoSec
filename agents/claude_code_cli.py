"""Claude Code CLI harness."""

from __future__ import annotations

from dynacomsec.agents.templated_cli import TemplatedCliHarness


class ClaudeCodeCliHarness(TemplatedCliHarness):
    """Run Claude Code as a DynaComSec full-agent target."""

    NAME = "claude-code-cli"
    COMMAND_ENV_VAR = "DYNACOMSEC_CLAUDE_CODE_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = "claude --model {model} --print < {instruction_file}"
