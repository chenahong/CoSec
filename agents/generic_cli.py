"""Generic command-line harness for arbitrary local agents.

The command receives one JSON request file per turn through the
``DYNACOMSEC_REQUEST_JSON`` environment variable. Its stdout is treated as the
agent response unless it writes a JSON object with a ``content`` field.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dynacomsec.agents.base import AgentHarness
from dynacomsec.runtime.base import AgentRequest, AgentResponse


class GenericCliHarness(AgentHarness):
    """Run any CLI command as a DynaComSec target agent."""

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
        self.command = command or os.environ.get("DYNACOMSEC_AGENT_COMMAND")
        if not self.command:
            raise ValueError(
                "generic-cli requires --agent-command or DYNACOMSEC_AGENT_COMMAND. "
                "The command should read DYNACOMSEC_REQUEST_JSON and print a response."
            )

    @staticmethod
    def harness_name() -> str:
        return "generic-cli"

    def send(self, request: AgentRequest) -> AgentResponse:
        payload = {
            "messages": request.messages,
            "session_id": request.session_id,
            "community_id": request.community_id,
            "agent_id": request.agent_id,
            "metadata": request.metadata or {},
            "model": self.model,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            request_path = f.name
        env = dict(os.environ)
        env["DYNACOMSEC_REQUEST_JSON"] = request_path
        if self.model:
            env["DYNACOMSEC_AGENT_MODEL"] = self.model
        try:
            proc = subprocess.run(
                self.command,
                shell=True,
                cwd=str(self.workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        finally:
            Path(request_path).unlink(missing_ok=True)

        content = _parse_cli_content(proc.stdout)
        raw: dict[str, Any] = {
            "command": self.command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0 and not content:
            content = proc.stderr.strip()
        return AgentResponse(
            content=content,
            runtime=self.name,
            model=self.model or "generic-cli",
            session_id=request.session_id,
            raw=raw,
        )


def _parse_cli_content(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        value = parsed.get("content") or parsed.get("response") or parsed.get("message")
        if value is not None:
            return str(value)
    return text
