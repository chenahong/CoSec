"""Lightweight full-agent harness interface.

Runtimes are best for chat-completions style backends. Harnesses are for agents
that need setup, local files, CLIs, or SDK loops. The interface intentionally
also implements AgentRuntime so the existing DynaComSec runner can use either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dynacomsec.runtime.base import AgentRequest, AgentResponse, AgentRuntime


class AgentHarness(AgentRuntime, ABC):
    """Base class for CLI/SDK agents that run a DynaComSec scenario turn."""

    def __init__(
        self,
        *,
        model: str | None = None,
        workdir: str | Path | None = None,
        timeout: int = 3600,
        **_: object,
    ) -> None:
        self.model = model
        self.workdir = Path(workdir or ".").resolve()
        self.timeout = timeout

    def setup(self, scenario: dict) -> None:
        """Prepare harness state before the first scenario turn."""

    def teardown(self, scenario: dict) -> None:
        """Clean up harness state after the scenario finishes."""

    def collect_logs(self) -> dict:
        """Return harness-specific diagnostic metadata."""

        return {}

    @staticmethod
    @abstractmethod
    def harness_name() -> str:
        """Stable harness name used by CLI flags and result files."""

    @property
    def name(self) -> str:
        return self.harness_name()

    @abstractmethod
    def send(self, request: AgentRequest) -> AgentResponse:
        """Send one turn to the full agent and return the observed response."""

