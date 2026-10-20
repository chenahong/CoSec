"""Agent harness registry.

Inspired by chi-bench's in-tree harness registry, but deliberately smaller:
DynaComSec can resolve a named harness without depending on Harbor.
"""

from __future__ import annotations

import importlib

from dynacomsec.agents.base import AgentHarness


AGENT_IMPORT_PATHS: dict[str, str] = {
    "claude-code-cli": "dynacomsec.agents.claude_code_cli:ClaudeCodeCliHarness",
    "codex-cli": "dynacomsec.agents.codex_cli:CodexCliHarness",
    "generic-cli": "dynacomsec.agents.generic_cli:GenericCliHarness",
    "hermes-cli": "dynacomsec.agents.hermes_cli:HermesCliHarness",
    "openclaw-cli": "dynacomsec.agents.openclaw_cli:OpenClawCliHarness",
}


def available_agents() -> list[str]:
    return sorted(AGENT_IMPORT_PATHS)


def load_agent_class(name: str) -> type[AgentHarness]:
    import_path = AGENT_IMPORT_PATHS.get(name)
    if not import_path:
        options = ", ".join(available_agents())
        raise ValueError(f"unknown agent harness: {name!r}. Available agent harnesses: {options}")
    module_name, _, class_name = import_path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"invalid agent import path for {name!r}: {import_path!r}")
    module = importlib.import_module(module_name)
    harness_cls = getattr(module, class_name)
    if not issubclass(harness_cls, AgentHarness):
        raise TypeError(f"{import_path} is not an AgentHarness subclass")
    return harness_cls


def build_agent(name: str, **kwargs: object) -> AgentHarness:
    return load_agent_class(name)(**kwargs)
