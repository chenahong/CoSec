"""OpenAI-compatible chat runtime for simple API-backed agents."""

from __future__ import annotations

import os
from typing import Any

from dynacomsec.runtime.base import AgentRequest, AgentResponse, AgentRuntime


class OpenAIChatRuntime(AgentRuntime):
    """Send DynaComSec turns directly to an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2000,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenAIChatRuntime requires the openai package. Install it before using --runtime openai-chat."
            ) from exc
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._max_tokens = max_tokens
        client_kwargs: dict[str, Any] = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "")}
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self._client = OpenAI(**client_kwargs)

    def send(self, request: AgentRequest) -> AgentResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=request.messages,
            max_tokens=self._max_tokens,
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
        return "openai-chat"
