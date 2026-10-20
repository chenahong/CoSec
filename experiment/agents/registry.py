"""Harbor agent registry for DynaComSec experiment runs."""

from __future__ import annotations


IN_TREE_AGENT_IMPORT_PATHS: dict[str, str] = {
    "generic-cli": "dynacomsec.experiment.agents.templated_cli_harness:DynaComSecGenericCliHarness",
    "claude-code": "dynacomsec.experiment.agents.templated_cli_harness:DynaComSecClaudeCodeCliHarness",
    "codex": "dynacomsec.experiment.agents.templated_cli_harness:DynaComSecCodexCliHarness",
    "hermes": "dynacomsec.experiment.agents.templated_cli_harness:DynaComSecHermesCliHarness",
    "openclaw": "dynacomsec.experiment.agents.templated_cli_harness:DynaComSecOpenClawCliHarness",
}


def available_agents() -> list[str]:
    return sorted(IN_TREE_AGENT_IMPORT_PATHS)
