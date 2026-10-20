"""Community-scoped conversation sessions for DynaComSec trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SessionMode = Literal["replay", "native"]


@dataclass
class SessionState:
    session_id: str
    community_id: str
    agent_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    backend_session_id: str | None = None
    backend_session_key: str | None = None
    turns: int = 0


class SessionManager:
    """Keep community conversations isolated while the Agent workspace stays shared.

    ``replay`` sends the stored same-session transcript on every turn and works
    with stateless CLIs. ``native`` sends only the current turn and expects the
    adapter command to bind ``session_id`` to the Agent's own create/resume API.
    """

    def __init__(self, mode: SessionMode = "replay") -> None:
        if mode not in {"replay", "native"}:
            raise ValueError(f"unsupported session mode: {mode}")
        self.mode: SessionMode = mode
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str, community_id: str, agent_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(
                session_id=session_id,
                community_id=community_id,
                agent_id=agent_id,
            )
            self._sessions[session_id] = state
            return state
        if state.community_id != community_id or state.agent_id != agent_id:
            raise ValueError(
                f"session {session_id!r} is already bound to "
                f"community={state.community_id!r}, agent={state.agent_id!r}"
            )
        return state

    def messages_for_turn(
        self,
        *,
        session_id: str,
        community_id: str,
        agent_id: str,
        system_context: str,
        user_message: str,
        include_history: bool | None = None,
    ) -> list[dict[str, str]]:
        state = self.get_or_create(session_id, community_id, agent_id)
        replay = self.mode == "replay" if include_history is None else include_history
        messages: list[dict[str, str]] = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        if replay:
            messages.extend(dict(message) for message in state.transcript)
        messages.append({"role": "user", "content": user_message})
        return messages

    def record_turn(
        self,
        *,
        session_id: str,
        community_id: str,
        agent_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        state = self.get_or_create(session_id, community_id, agent_id)
        state.transcript.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        )
        state.turns += 1

    def bind_backend_session(self, session_id: str, backend_session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown session: {session_id}")
        state.backend_session_id = backend_session_id

    def bind_backend_session_key(self, session_id: str, backend_session_key: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown session: {session_id}")
        state.backend_session_key = backend_session_key

    def find(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self._sessions),
            "sessions": [
                {
                    "session_id": state.session_id,
                    "community_id": state.community_id,
                    "agent_id": state.agent_id,
                    "turns": state.turns,
                    "message_count": len(state.transcript),
                    "backend_session_id": state.backend_session_id,
                    "backend_session_key": state.backend_session_key,
                }
                for state in sorted(self._sessions.values(), key=lambda item: item.session_id)
            ],
        }
