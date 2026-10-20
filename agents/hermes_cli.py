"""Hermes CLI harness."""

from __future__ import annotations

from dynacomsec.agents.templated_cli import TemplatedCliHarness


class HermesCliHarness(TemplatedCliHarness):
    """Run Hermes as a DynaComSec full-agent target."""

    NAME = "hermes-cli"
    COMMAND_ENV_VAR = "DYNACOMSEC_HERMES_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = "hermes --model {model} --prompt-file {instruction_file}"
