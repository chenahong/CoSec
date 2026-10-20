"""OpenClaw runtime adapter.

This follows the LITMUS OpenClaw integration style: OpenClaw is reached through
its OpenAI-compatible Chat Completions endpoint and the OpenAI ``user`` field is
used as the stable session key.
"""

from __future__ import annotations

import os

from dynacomsec.runtime.base import AgentRequest, AgentResponse, AgentRuntime


class OpenClawRuntime(AgentRuntime):
    """Runtime adapter for a live OpenClaw Gateway."""

    def __init__(
        self,
        gateway_url: str | None = None,
        gateway_token: str | None = None,
        agent_id: str = "default",
        max_tokens: int = 2000,
    ) -> None:
        resolved_url = gateway_url or os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
        resolved_token = gateway_token or os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        if agent_id in {"default", "openclaw", "openclaw/default"}:
            self._model = "openclaw/default"
        else:
            self._model = f"openclaw/{agent_id}"
        self._gateway_url = resolved_url.rstrip("/")
        self._max_tokens = max_tokens
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenClawRuntime requires the openai package. Install it before using --runtime openclaw."
            ) from exc
        self._client = OpenAI(
            api_key=resolved_token or "no-key",
            base_url=f"{self._gateway_url}/v1",
        )

    def send(self, request: AgentRequest) -> AgentResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=request.messages,
            user=request.session_id,
        )
        content = response.choices[0].message.content or ""
        return AgentResponse(
            content=content,
            runtime=self.name,
            model=self._model,
            session_id=request.session_id,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    @property
    def name(self) -> str:
        return "openclaw"

    @property
    def description(self) -> str:
        return f"OpenClawRuntime(gateway={self._gateway_url}, model={self._model})"
