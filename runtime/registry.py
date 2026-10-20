"""Runtime registry for DynaComSec target-agent backends.

This mirrors the small-but-useful part of chi-bench's agent registry: names are
kept in one place, then resolved dynamically by the runner.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from dynacomsec.runtime.base import AgentRuntime


RUNTIME_IMPORT_PATHS: dict[str, str] = {
    "dryrun": "dynacomsec.runtime.dryrun:DryRunRuntime",
    "openclaw": "dynacomsec.runtime.openclaw:OpenClawRuntime",
    "openclaw-gateway": "dynacomsec.runtime.openclaw:OpenClawRuntime",
    "openai-chat": "dynacomsec.runtime.openai_chat:OpenAIChatRuntime",
}


def available_runtimes() -> list[str]:
    return sorted(RUNTIME_IMPORT_PATHS)


def load_runtime_class(name: str) -> type[AgentRuntime]:
    import_path = RUNTIME_IMPORT_PATHS.get(name)
    if not import_path:
        options = ", ".join(available_runtimes())
        raise ValueError(f"unknown runtime: {name!r}. Available runtimes: {options}")
    module_name, _, class_name = import_path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"invalid runtime import path for {name!r}: {import_path!r}")
    module = importlib.import_module(module_name)
    runtime_cls = getattr(module, class_name)
    if not issubclass(runtime_cls, AgentRuntime):
        raise TypeError(f"{import_path} is not an AgentRuntime subclass")
    return runtime_cls


def build_runtime(name: str, factory: Callable[[type[AgentRuntime]], AgentRuntime]) -> AgentRuntime:
    """Load a runtime class and instantiate it with caller-owned arguments."""

    return factory(load_runtime_class(name))
