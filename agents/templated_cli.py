"""Reusable command-template harness for CLI agents."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dynacomsec.agents.base import AgentHarness
from dynacomsec.runtime.base import AgentRequest, AgentResponse


class TemplatedCliHarness(AgentHarness):
    """Base class for first-class CLI harnesses.

    Subclasses provide a stable harness name and a default command template.
    The user can override the command with ``--agent-command``.

    Supported template variables:
    - ``{instruction}``: shell-quoted rendered prompt
    - ``{instruction_file}``: shell-quoted path to rendered prompt
    - ``{request_json}``: shell-quoted path to the full AgentRequest JSON
    - ``{model}``: shell-quoted model name
    - ``{session_id}``: shell-quoted DynaComSec session id
    - ``{community_id}``: shell-quoted active community id
    - ``{agent_id}``: shell-quoted target agent id
    - ``{workdir}``: shell-quoted harness working directory
    """

    DEFAULT_COMMAND_TEMPLATE: str | None = None
    COMMAND_ENV_VAR: str = "DYNACOMSEC_AGENT_COMMAND"
    NAME: str = "templated-cli"

    def __init__(
        self,
        *,
        command: str | None = None,
        model: str | None = None,
        workdir: str | Path | None = None,
        timeout: int = 3600,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, workdir=workdir, timeout=timeout, **kwargs)
        self.command_template = command or os.environ.get(self.COMMAND_ENV_VAR) or self.DEFAULT_COMMAND_TEMPLATE
        if not self.command_template:
            raise ValueError(
                f"{self.NAME} requires --agent-command or {self.COMMAND_ENV_VAR}. "
                "Use a command template that prints the final agent response to stdout."
            )

    @classmethod
    def harness_name(cls) -> str:
        return cls.NAME

    def send(self, request: AgentRequest) -> AgentResponse:
        payload = _payload_for_request(request, self.model)
        instruction = _render_instruction(request)
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as req_f:
            json.dump(payload, req_f, ensure_ascii=False, indent=2)
            request_path = req_f.name
        with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as inst_f:
            inst_f.write(instruction)
            instruction_path = inst_f.name

        env = dict(os.environ)
        env["DYNACOMSEC_REQUEST_JSON"] = request_path
        env["DYNACOMSEC_INSTRUCTION_FILE"] = instruction_path
        if self.model:
            env["DYNACOMSEC_AGENT_MODEL"] = self.model

        command = self._format_command(
            request=request,
            instruction=instruction,
            instruction_path=instruction_path,
            request_path=request_path,
        )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        finally:
            Path(request_path).unlink(missing_ok=True)
            Path(instruction_path).unlink(missing_ok=True)

        content = _parse_cli_content(proc.stdout)
        if proc.returncode != 0 and not content:
            content = proc.stderr.strip()
        raw: dict[str, Any] = {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        return AgentResponse(
            content=content,
            runtime=self.name,
            model=self.model or self.name,
            session_id=request.session_id,
            raw=raw,
        )

    def _format_command(
        self,
        *,
        request: AgentRequest,
        instruction: str,
        instruction_path: str,
        request_path: str,
    ) -> str:
        model = self.model or os.environ.get("DYNACOMSEC_AGENT_MODEL", "")
        values = {
            "instruction": shlex.quote(instruction),
            "instruction_file": shlex.quote(instruction_path),
            "request_json": shlex.quote(request_path),
            "model": shlex.quote(model),
            "session_id": shlex.quote(request.session_id),
            "community_id": shlex.quote(request.community_id),
            "agent_id": shlex.quote(request.agent_id),
            "workdir": shlex.quote(str(self.workdir)),
        }
        command = self.command_template
        for key, value in values.items():
            command = command.replace("{" + key + "}", value)
        return command


def _payload_for_request(request: AgentRequest, model: str | None) -> dict:
    return {
        "messages": request.messages,
        "session_id": request.session_id,
        "community_id": request.community_id,
        "agent_id": request.agent_id,
        "metadata": request.metadata or {},
        "model": model,
    }


def _render_instruction(request: AgentRequest) -> str:
    parts = []
    for message in request.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append(f"SESSION_ID: {request.session_id}")
    parts.append(f"COMMUNITY_ID: {request.community_id}")
    parts.append(f"TARGET_AGENT_ID: {request.agent_id}")
    return "\n\n".join(parts)


def _parse_cli_content(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    for candidate in reversed(text.splitlines()):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed.get("content") or parsed.get("response") or parsed.get("message")
            if value is not None:
                return str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        value = parsed.get("content") or parsed.get("response") or parsed.get("message")
        if value is not None:
            return str(value)
    return text

