"""OpenClaw CLI harness.

This is separate from ``runtime.openclaw``. The runtime talks to an already
running OpenClaw Gateway; this harness runs the OpenClaw CLI as the target
agent, closer to chi-bench's full-agent integration style.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from dynacomsec.agents.base import AgentHarness
from dynacomsec.runtime.base import AgentRequest, AgentResponse


class OpenClawCliHarness(AgentHarness):
    """Invoke ``openclaw agent`` once per DynaComSec turn."""

    def __init__(
        self,
        *,
        agent: str = "main",
        model: str | None = None,
        workdir: str | Path | None = None,
        timeout: int = 3600,
        openclaw_bin: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, workdir=workdir, timeout=timeout, **kwargs)
        self.agent = agent
        self.openclaw_bin = openclaw_bin or os.environ.get("OPENCLAW_BIN", "openclaw")

    @staticmethod
    def harness_name() -> str:
        return "openclaw-cli"

    def send(self, request: AgentRequest) -> AgentResponse:
        message = _render_request_message(request)
        cmd = [
            self.openclaw_bin,
            "agent",
            "--local",
            "--agent",
            self.agent,
            "--message",
            message,
            "--json",
        ]
        env = dict(os.environ)
        if self.model:
            env["DYNACOMSEC_AGENT_MODEL"] = self.model
        proc = subprocess.run(
            cmd,
            cwd=str(self.workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        content = _parse_openclaw_output(proc.stdout) or proc.stdout.strip()
        if proc.returncode != 0 and not content:
            content = proc.stderr.strip()
        return AgentResponse(
            content=content,
            runtime=self.name,
            model=self.model or f"openclaw/{self.agent}",
            session_id=request.session_id,
            raw={
                "command": " ".join(shlex.quote(part) for part in cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )


def _render_request_message(request: AgentRequest) -> str:
    parts = []
    for message in request.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def _parse_openclaw_output(stdout: str) -> str:
    last_content = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            content = event.get("content") or event.get("message") or event.get("text")
            if content:
                last_content = str(content)
    return last_content
