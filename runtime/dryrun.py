"""Dry-run runtime for smoke tests without a live agent."""

from __future__ import annotations

from dynacomsec.runtime.base import AgentRequest, AgentResponse, AgentRuntime


class DryRunRuntime(AgentRuntime):
    """Deterministic placeholder runtime used to validate scenario plumbing."""

    def send(self, request: AgentRequest) -> AgentResponse:
        last = request.messages[-1]["content"] if request.messages else ""
        content = (
            "[DRY-RUN] No real agent was contacted. "
            f"community={request.community_id}; session={request.session_id}; message={last[:160]}"
        )
        return AgentResponse(
            content=content,
            runtime=self.name,
            model="dryrun",
            session_id=request.session_id,
            raw=None,
        )

    @property
    def name(self) -> str:
        return "dryrun"
