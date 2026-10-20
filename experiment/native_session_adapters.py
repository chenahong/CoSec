"""Agent-specific native session command adapters."""

from __future__ import annotations

import hashlib
import json
import shlex
import uuid
from dataclasses import dataclass
from typing import Any

from dynacomsec.experiment.session_manager import SessionManager


_RESPONSE_START = "<<<DYNACOMSEC_NATIVE_RESPONSE>>>"


@dataclass(frozen=True)
class NativeInvocation:
    command: str
    backend_session_id: str | None = None


class NativeSessionAdapter:
    """Translate a DynaComSec session into one Agent's native session API."""

    name = "native"

    def build_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        community_id: str,
        agent_id: str,
        instruction_file: str,
        model: str,
        usage_file: str | None = None,
    ) -> NativeInvocation:
        raise NotImplementedError

    def complete_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        stdout: str,
    ) -> str:
        return stdout.strip()


class OpenClawSessionKeyNativeAdapter(NativeSessionAdapter):
    """Use a deterministic OpenClaw local CLI session key per community."""

    name = "openclaw-session-key"

    def build_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        community_id: str,
        agent_id: str,
        instruction_file: str,
        model: str,
        usage_file: str | None = None,
    ) -> NativeInvocation:
        state = manager.get_or_create(session_id, community_id, agent_id)
        session_key = openclaw_session_key(session_id)
        if state.backend_session_key is None:
            manager.bind_backend_session_key(session_id, session_key)
        args = [
            "openclaw",
            "agent",
            "--local",
            "--agent",
            "main",
            "--session-key",
            session_key,
            "--message-file",
            instruction_file,
            "--json",
        ]
        if model and model != "openclaw/default":
            args.extend(["--model", model])
        command = " ".join(shlex.quote(item) for item in args)
        return NativeInvocation(command=command, backend_session_id=state.backend_session_id)

    def complete_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        stdout: str,
    ) -> str:
        payload = _last_openclaw_response_object(stdout)
        if not payload:
            raise RuntimeError("OpenClaw local native adapter returned invalid JSON")
        backend_id = extract_openclaw_session_id(payload)
        state = manager_state(manager, session_id)
        manager.bind_backend_session(
            session_id,
            backend_id or state.backend_session_key or session_id,
        )
        response = extract_openclaw_response(payload)
        if response is None:
            raise RuntimeError("OpenClaw local native adapter could not extract the response")
        return response.strip()


class ClaudeCodeNativeAdapter(NativeSessionAdapter):
    """Capture Claude Code's session ID and resume it on later turns."""

    name = "claude-session-uuid"

    def build_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        community_id: str,
        agent_id: str,
        instruction_file: str,
        model: str,
        usage_file: str | None = None,
    ) -> NativeInvocation:
        state = manager.get_or_create(session_id, community_id, agent_id)
        backend_id = state.backend_session_id
        session_args: list[str] = []
        if backend_id is not None:
            session_args = ["--resume", backend_id]
        args = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--model",
            model,
            *session_args,
        ]
        command = " ".join(shlex.quote(item) for item in args)
        command += f" < {shlex.quote(instruction_file)}"
        return NativeInvocation(command=command, backend_session_id=backend_id)

    def complete_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        stdout: str,
    ) -> str:
        payload = _last_json_object(stdout)
        if payload:
            backend_id = extract_claude_session_id(payload)
            if backend_id:
                manager.bind_backend_session(session_id, backend_id)
            response = extract_claude_response(payload)
            if response is not None:
                return response.strip()
        return stdout.strip()


class CodexConversationNativeAdapter(NativeSessionAdapter):
    """Capture Codex's conversation UUID and use ``exec resume`` thereafter."""

    name = "codex-conversation-uuid"

    def build_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        community_id: str,
        agent_id: str,
        instruction_file: str,
        model: str,
        usage_file: str | None = None,
    ) -> NativeInvocation:
        state = manager.get_or_create(session_id, community_id, agent_id)
        response_file = f"/tmp/dynacomsec-codex-{uuid.uuid4().hex}.txt"
        base_args = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "--model",
            model,
            "--output-last-message",
            response_file,
        ]
        if state.backend_session_id:
            args = [
                *base_args,
                "resume",
                state.backend_session_id,
                "-",
            ]
        else:
            args = [*base_args, "-"]
        codex = " ".join(shlex.quote(item) for item in args)
        command = (
            'mkdir -p "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
            'cd "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
            f"{codex} < {shlex.quote(instruction_file)}; "
            "status=$?; "
            f"printf '\\n{_RESPONSE_START}\\n'; "
            f"cat {shlex.quote(response_file)} 2>/dev/null || true; "
            f"rm -f {shlex.quote(response_file)}; "
            "exit $status"
        )
        return NativeInvocation(command=command, backend_session_id=state.backend_session_id)

    def complete_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        stdout: str,
    ) -> str:
        event_text, marker, response = stdout.partition(_RESPONSE_START)
        if not marker:
            raise RuntimeError("Codex native adapter did not receive the final-response marker")
        state = manager_state(manager, session_id)
        if state.backend_session_id is None:
            backend_id = extract_codex_session_id(event_text)
            if not backend_id:
                raise RuntimeError("Codex native adapter could not extract a conversation UUID")
            manager.bind_backend_session(session_id, backend_id)
        return response.strip()


class HermesSessionNativeAdapter(NativeSessionAdapter):
    """Capture Hermes's session ID and use ``--resume`` on later turns."""

    name = "hermes-session-id"

    def build_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        community_id: str,
        agent_id: str,
        instruction_file: str,
        model: str,
        usage_file: str | None = None,
    ) -> NativeInvocation:
        state = manager.get_or_create(session_id, community_id, agent_id)
        args = ["hermes"]
        if state.backend_session_id:
            args.extend(["--resume", state.backend_session_id])
        args.extend(["-z", "__DYNACOMSEC_HERMES_PROMPT__", "--provider", "custom"])
        if model:
            args.extend(["--model", model])
        if usage_file:
            args.extend(["--usage-file", usage_file])
        args.extend(
            [
                "--in",
                "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}",
                "--ignore-rules",
            ]
        )
        hermes = " ".join(shlex.quote(item) for item in args)
        prompt = f"$(cat {shlex.quote(instruction_file)})"
        hermes = hermes.replace("__DYNACOMSEC_HERMES_PROMPT__", f'"{prompt}"')
        hermes = hermes.replace(
            "'${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}'",
            '"${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"',
        )
        command = (
            'mkdir -p "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
            'cd "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
            f"{hermes}; "
            "status=$?; "
            f"printf '\\n{_RESPONSE_START}\\n'; "
            "hermes sessions list 2>/dev/null || true; "
            "exit $status"
        )
        return NativeInvocation(command=command, backend_session_id=state.backend_session_id)

    def complete_invocation(
        self,
        *,
        manager: SessionManager,
        session_id: str,
        stdout: str,
    ) -> str:
        response, marker, sessions = stdout.partition(_RESPONSE_START)
        if not marker:
            raise RuntimeError("Hermes native adapter did not receive the sessions marker")
        state = manager_state(manager, session_id)
        backend_id = extract_hermes_session_id(sessions)
        if backend_id:
            manager.bind_backend_session(session_id, backend_id)
        elif state.backend_session_id is None:
            raise RuntimeError("Hermes native adapter could not extract a session ID")
        return response.strip()


def manager_state(manager: SessionManager, session_id: str):
    state = manager.find(session_id)
    if state is None:
        raise KeyError(f"unknown session: {session_id}")
    return state


def extract_codex_session_id(text: str) -> str | None:
    """Extract a Codex thread/conversation ID from JSONL events."""

    keys = ("thread_id", "conversation_id", "session_id", "threadId", "conversationId")
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = _find_string(event, keys)
        if found:
            return found
    return None


def extract_claude_session_id(payload: dict[str, Any]) -> str | None:
    return _find_string(payload, ("session_id", "sessionId", "conversation_id", "conversationId"))


def extract_claude_response(payload: dict[str, Any]) -> str | None:
    for key in ("result", "response", "content", "message", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = payload.get("messages")
    if isinstance(messages, list):
        texts: list[str] = []
        for item in messages:
            if isinstance(item, dict):
                value = item.get("content") or item.get("text")
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        if texts:
            return "\n\n".join(texts)
    return None


def extract_hermes_session_id(text: str) -> str | None:
    """Extract the most recent Hermes session ID from ``hermes sessions list``."""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Title") or set(stripped) <= {"─", "-"}:
            continue
        parts = stripped.split()
        if parts:
            candidate = parts[-1]
            if candidate.startswith("20") and "_" in candidate:
                return candidate
    return None


def create_native_adapter(name: str | None) -> NativeSessionAdapter | None:
    adapters: dict[str, type[NativeSessionAdapter]] = {
        "openclaw-session-key": OpenClawSessionKeyNativeAdapter,
        "claude-session-uuid": ClaudeCodeNativeAdapter,
        "codex-conversation-uuid": CodexConversationNativeAdapter,
        "hermes-session-id": HermesSessionNativeAdapter,
    }
    if name is None:
        return None
    try:
        return adapters[name]()
    except KeyError as exc:
        raise ValueError(f"unknown native session adapter: {name}") from exc


def _find_string(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        for child in value.values():
            found = _find_string(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string(child, keys)
            if found:
                return found
    return None


def openclaw_session_key(logical_session_id: str, agent: str = "main") -> str:
    """Build a stable, shell-safe OpenClaw routing key from a logical session."""

    digest = hashlib.sha256(logical_session_id.encode("utf-8")).hexdigest()[:24]
    return f"agent:{agent}:dynacomsec-{digest}"


def extract_openclaw_session_id(payload: dict[str, Any]) -> str | None:
    meta = payload.get("meta")
    if isinstance(meta, dict):
        agent_meta = meta.get("agentMeta")
        if isinstance(agent_meta, dict) and agent_meta.get("sessionId"):
            return str(agent_meta["sessionId"])
    return _find_string(payload, ("sessionId", "session_id"))


def extract_openclaw_response(payload: dict[str, Any]) -> str | None:
    """Extract final text from OpenClaw's version-dependent JSON envelope."""

    for key in ("payloads", "messages"):
        values = payload.get(key)
        if isinstance(values, list):
            texts: list[str] = []
            for item in reversed(values):
                if isinstance(item, dict):
                    for text_key in ("text", "content", "message", "response"):
                        value = item.get(text_key)
                        if isinstance(value, str) and value.strip():
                            texts.append(value.strip())
                            break
            if texts:
                return "\n\n".join(reversed(texts))
    for key in ("response", "content", "message", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    result = payload.get("result")
    if isinstance(result, dict):
        return extract_openclaw_response(result)
    return None


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    found = None
    best_score = (-1, -1)
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            score = (index + consumed, consumed)
            if score > best_score:
                found = value
                best_score = score
    return found


def _last_openclaw_response_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    fallback = None
    found = None
    best_score = (-1, -1, -1)
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        fallback = value
        score = (_openclaw_response_score(value), index + consumed, consumed)
        if score > best_score:
            found = value
            best_score = score
    if found is not None and best_score[0] > 0:
        return found
    return fallback


def _openclaw_response_score(value: dict[str, Any]) -> int:
    payloads = value.get("payloads")
    if isinstance(payloads, list) and any(
        isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
        for item in payloads
    ):
        return 4
    messages = value.get("messages")
    if isinstance(messages, list):
        return 3
    if isinstance(value.get("result"), dict):
        return 2
    if any(isinstance(value.get(key), str) and value.get(key, "").strip() for key in ("response", "content", "text")):
        return 1
    return 0
