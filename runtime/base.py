"""Runtime interface for target personal agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRequest:
    """A single message delivered to a target agent in one community session."""

    messages: list[dict]
    session_id: str
    community_id: str
    agent_id: str
    metadata: dict | None = None


@dataclass(frozen=True)
class AgentResponse:
    """A runtime response plus lightweight execution metadata."""

    content: str
    runtime: str
    model: str
    session_id: str
    raw: dict | None = None


class AgentRuntime(ABC):
    """Abstract interface implemented by real-agent backends."""

    @abstractmethod
    def send(self, request: AgentRequest) -> AgentResponse:
        """Send one request to the target agent and return its response."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable runtime name used in result files."""
