"""Harbor-compatible command-template harnesses for DynaComSec."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import tempfile
from typing import TYPE_CHECKING

from dynacomsec.experiment.adaptive import (
    is_participant_seed_then_llm,
    is_adaptive_scenario,
    max_rounds,
    run_adaptive_loop,
)
from dynacomsec.experiment.dynamic_state_runtime import build_dynamic_state_snapshots
from dynacomsec.experiment.scenario_engine import (
    render_agent_request,
    run_scripted_scenario,
    target_agent,
)
from dynacomsec.experiment.session_manager import SessionManager
from dynacomsec.experiment.state_context import normalize_context_mode
from dynacomsec.experiment.native_session_adapters import (
    NativeSessionAdapter,
    create_native_adapter,
)

try:
    from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
except ModuleNotFoundError:  # pragma: no cover - imported by Harbor at runtime.
    BaseInstalledAgent = object  # type: ignore[assignment,misc]

    def with_prompt_template(func):  # type: ignore[no-untyped-def]
        return func

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext


if BaseInstalledAgent is object:
    class _BaseHarness:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("Harbor is required to instantiate DynaComSec Harbor harnesses.")
else:
    class _BaseHarness(BaseInstalledAgent):  # type: ignore[misc,no-redef]
        SUPPORTS_ATIF: bool = False
        NAME = "generic-cli"
        COMMAND_ENV_VAR = "DYNACOMSEC_AGENT_COMMAND"
        DEFAULT_COMMAND_TEMPLATE: str | None = None
        NATIVE_ADAPTER_NAME: str | None = None

        def __init__(
            self,
            timeout_seconds: int | None = None,
            command: str | None = None,
            session_mode: str = "replay",
            context_mode: str = "strong",
            speaker_mode: str = "llm",
            guardrail_mode: str = "none",
            transient_retry_attempts: int = 1,
            transient_retry_sleep_seconds: float = 0,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.timeout_seconds = int(timeout_seconds) if timeout_seconds is not None else 3600
            self.command_template = command
            self.session_mode = SessionManager(session_mode).mode
            self.context_mode = normalize_context_mode(context_mode)
            self.speaker_mode = _normalize_speaker_mode(speaker_mode)
            self.guardrail_mode = _normalize_guardrail_mode(guardrail_mode)
            self.transient_retry_attempts = max(1, int(transient_retry_attempts))
            self.transient_retry_sleep_seconds = max(0.0, float(transient_retry_sleep_seconds))
            self.native_adapter: NativeSessionAdapter | None = create_native_adapter(
                self.NATIVE_ADAPTER_NAME
            )
            if self.session_mode == "native" and self.native_adapter is None:
                raise ValueError(f"{self.NAME} does not provide a native session adapter")

        @staticmethod
        def name() -> str:
            return _BaseHarness.NAME

        def get_version_command(self) -> str | None:
            return None

        async def install(self, environment: "BaseEnvironment") -> None:
            return None

        @with_prompt_template
        async def run(self, instruction: str, environment: "BaseEnvironment", context: "AgentContext") -> None:
            template = (
                self.command_template
                or self._extra_env.get(self.COMMAND_ENV_VAR)
                or os.environ.get(self.COMMAND_ENV_VAR)
                or self.DEFAULT_COMMAND_TEMPLATE
            )
            if not template:
                raise RuntimeError(
                    f"{self.NAME} requires command=... or {self.COMMAND_ENV_VAR}. "
                    "The command should print the final response to stdout."
                )
            scenario = await _load_scenario(environment)
            if is_adaptive_scenario(scenario):
                await self._run_adaptive(instruction, scenario, template, environment)
                return
            await self._run_scripted(scenario, template, environment)

        async def _run_scripted(
            self,
            scenario: dict,
            template: str,
            environment: "BaseEnvironment",
        ) -> None:
            session_manager = SessionManager(self.session_mode)
            speaker_call = None
            default_speaker_model = ""
            speaker_models_by_role: dict[str, str] = {}
            speaker_generation_enabled = (
                self.speaker_mode in {"llm", "attacker"} and is_participant_seed_then_llm(scenario)
            )
            if speaker_generation_enabled:
                simulation = scenario.get("participant_simulation") or scenario.get("attacker_simulation", {})
                role_configs = simulation.get("roles") if isinstance(simulation.get("roles"), dict) else {}
                default_speaker_model = (
                    self._extra_env.get("DYNACOMSEC_SPEAKER_MODEL")
                    or os.environ.get("DYNACOMSEC_SPEAKER_MODEL")
                    or self._extra_env.get("DYNACOMSEC_ATTACKER_MODEL")
                    or os.environ.get("DYNACOMSEC_ATTACKER_MODEL")
                    or simulation.get("attacker_model")
                    or simulation.get("speaker_model")
                    or simulation.get("default_model")
                    or self._extra_env.get("OPENAI_MODEL")
                    or os.environ.get("OPENAI_MODEL")
                    or "gpt-4o"
                )
                default_speaker_api_key = (
                    self._extra_env.get("DYNACOMSEC_SPEAKER_API_KEY")
                    or os.environ.get("DYNACOMSEC_SPEAKER_API_KEY")
                    or self._extra_env.get("DYNACOMSEC_ATTACKER_API_KEY")
                    or os.environ.get("DYNACOMSEC_ATTACKER_API_KEY")
                    or self._extra_env.get("OPENAI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or ""
                )
                default_speaker_base_url = (
                    self._extra_env.get("DYNACOMSEC_SPEAKER_BASE_URL")
                    or os.environ.get("DYNACOMSEC_SPEAKER_BASE_URL")
                    or self._extra_env.get("DYNACOMSEC_ATTACKER_BASE_URL")
                    or os.environ.get("DYNACOMSEC_ATTACKER_BASE_URL")
                    or self._extra_env.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                )
                if not default_speaker_api_key:
                    raise RuntimeError(
                        "script_seed_then_llm requires DYNACOMSEC_SPEAKER_API_KEY or OPENAI_API_KEY"
                    )

                async def call_script_speaker(messages: list[dict[str, str]]) -> str:
                    role = _speaker_role_from_messages(messages)
                    role_config = role_configs.get(role, {}) if isinstance(role_configs, dict) else {}
                    model, api_key, base_url = _speaker_client_config(
                        role=role,
                        role_config=role_config if isinstance(role_config, dict) else {},
                        default_model=str(default_speaker_model),
                        default_api_key=default_speaker_api_key,
                        default_base_url=default_speaker_base_url,
                        extra_env=self._extra_env,
                    )
                    speaker_models_by_role[role] = model
                    payload = json.dumps(
                        {
                            "mode": "message",
                            "messages": messages,
                            "temperature": float(role_config.get("temperature", simulation.get("temperature", 0.8))),
                        },
                        ensure_ascii=False,
                    )
                    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
                    result = await environment.exec(
                        f"printf '%s' {shlex.quote(encoded)} | base64 -d | "
                        "python -m dynacomsec.experiment.speaker_cli",
                        env={
                            "DYNACOMSEC_SPEAKER_MODEL": model,
                            "DYNACOMSEC_SPEAKER_API_KEY": api_key,
                            "DYNACOMSEC_SPEAKER_BASE_URL": base_url or "",
                        },
                        timeout_sec=self.timeout_seconds,
                    )
                    if getattr(result, "return_code", 1) != 0:
                        raise RuntimeError(
                            "scripted speaker failed: "
                            + str(getattr(result, "stderr", "") or getattr(result, "stdout", ""))
                        )
                    try:
                        response = json.loads(str(result.stdout).strip())
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("scripted speaker returned invalid JSON") from exc
                    message = str(response.get("message") or "").strip()
                    if not message:
                        raise RuntimeError("scripted speaker returned an empty message")
                    return message

                speaker_call = call_script_speaker

            async def dynamic_state_update_call(
                runtime_scenario: dict,
                applied_event_ids: list[str],
                index: int,
            ) -> list[dict]:
                updates = []
                for snapshot in build_dynamic_state_snapshots(
                    runtime_scenario,
                    applied_event_ids=applied_event_ids,
                ):
                    await _write_json(environment, str(snapshot["path"]), snapshot["payload"])
                    updates.append(
                        {
                            "step": index,
                            "community_id": snapshot["community_id"],
                            "path": snapshot["path"],
                            "applied_event_ids": applied_event_ids,
                        }
                    )
                return updates

            async def step_call(
                messages: list[dict[str, str]],
                session_id: str,
                community_id: str,
                agent_id: str,
                step: dict,
                index: int,
            ) -> str:
                prompt = render_agent_request(
                    messages,
                    session_id=session_id,
                    community_id=community_id,
                    agent_id=agent_id,
                )
                return await self._run_victim_once(
                    prompt,
                    template,
                    environment,
                    log_path=f"/logs/agent/step-{index}.txt",
                    session_id=session_id,
                    community_id=community_id,
                    agent_id=agent_id,
                    session_manager=session_manager,
                )

            steps, interaction_metadata = await run_scripted_scenario(
                scenario,
                step_call=step_call,
                session_manager=session_manager,
                speaker_call=speaker_call,
                context_mode=self.context_mode,
                speaker_generation_enabled=speaker_generation_enabled,
                speaker_generation_mode=self.speaker_mode,
                guardrail_mode=self.guardrail_mode,
                dynamic_state_update_call=dynamic_state_update_call,
            )
            interaction_metadata.update(
                {
                    "victim_runtime": self.NAME,
                    "victim_model": self.model_name or "",
                    "speaker_mode": "script_seed_then_llm"
                    if self.speaker_mode == "llm" and speaker_generation_enabled
                    else "attacker_seed_then_llm"
                    if self.speaker_mode == "attacker" and speaker_generation_enabled
                    else "seed_message",
                    "guardrail_mode": self.guardrail_mode,
                    "speaker_model": str(default_speaker_model)
                    if speaker_generation_enabled
                    else "",
                    "speaker_models": speaker_models_by_role,
                }
            )
            for step in steps:
                speaker_input = step.get("speaker_model_input")
                if isinstance(speaker_input, dict) and speaker_input.get("source") == "script_seed_then_llm":
                    role = str(speaker_input.get("speaker_role") or "speaker")
                    speaker_input["speaker_model"] = speaker_models_by_role.get(role, str(default_speaker_model))
                step.update({"runtime": self.NAME, "model": self.model_name or ""})
            trajectory = {
                "schema_version": "dynacomsec.trajectory.v1",
                "scenario_id": scenario.get("scenario_id"),
                "interaction_metadata": interaction_metadata,
                "steps": steps,
            }
            await _write_json(environment, "/logs/agent/trajectory.json", trajectory)
            await _write_json(environment, "/logs/agent/sessions.json", session_manager.summary())
            await _write_text(environment, "/logs/agent/run_log.txt", _combined_log(steps))

        async def _run_adaptive(
            self,
            instruction: str,
            scenario: dict,
            template: str,
            environment: "BaseEnvironment",
        ) -> None:
            session_manager = SessionManager(self.session_mode)
            attacker_model = (
                self._extra_env.get("DYNACOMSEC_ATTACKER_MODEL")
                or os.environ.get("DYNACOMSEC_ATTACKER_MODEL")
                or self._extra_env.get("OPENAI_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or "gpt-4o"
            )
            attacker_api_key = (
                self._extra_env.get("DYNACOMSEC_ATTACKER_API_KEY")
                or os.environ.get("DYNACOMSEC_ATTACKER_API_KEY")
                or self._extra_env.get("OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
            attacker_base_url = (
                self._extra_env.get("DYNACOMSEC_ATTACKER_BASE_URL")
                or os.environ.get("DYNACOMSEC_ATTACKER_BASE_URL")
                or self._extra_env.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            if not attacker_api_key:
                raise RuntimeError(
                    "adaptive scenarios require DYNACOMSEC_ATTACKER_API_KEY or OPENAI_API_KEY"
                )

            round_limit = max_rounds(scenario)
            victim = target_agent(scenario)
            victim_agent_id = str(victim.get("agent_id", ""))

            async def victim_call(
                prompt: str,
                round_number: int,
                session_id: str,
                community_id: str,
                attacker: dict,
            ) -> str:
                return await self._run_victim_once(
                    prompt,
                    template,
                    environment,
                    log_path=f"/logs/agent/round-{round_number}.txt",
                    session_id=session_id,
                    community_id=community_id,
                    agent_id=victim_agent_id,
                    session_manager=session_manager,
                )

            async def attacker_call(messages: list[dict[str, str]]) -> dict[str, str]:
                payload = json.dumps({"messages": messages}, ensure_ascii=False)
                encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
                result = await environment.exec(
                    f"printf '%s' {shlex.quote(encoded)} | base64 -d | "
                    "python -m dynacomsec.experiment.attacker_cli",
                    env={
                        "DYNACOMSEC_ATTACKER_MODEL": attacker_model,
                        "DYNACOMSEC_ATTACKER_API_KEY": attacker_api_key,
                        "DYNACOMSEC_ATTACKER_BASE_URL": attacker_base_url or "",
                    },
                    timeout_sec=self.timeout_seconds,
                )
                if getattr(result, "return_code", 1) != 0:
                    raise RuntimeError(
                        "adaptive attacker failed: "
                        + str(getattr(result, "stderr", "") or getattr(result, "stdout", ""))
                    )
                try:
                    return json.loads(str(result.stdout).strip())
                except json.JSONDecodeError as exc:
                    raise RuntimeError("adaptive attacker returned invalid JSON") from exc

            transcript, stop_reason = await run_adaptive_loop(
                scenario,
                instruction,
                victim_call=victim_call,
                attacker_call=attacker_call,
                session_manager=session_manager,
            )
            for step in transcript:
                step.update(
                    {
                        "victim_runtime": self.NAME,
                        "victim_model": self.model_name or "",
                        "attacker_model": attacker_model,
                    }
                )

            trajectory = {
                "schema_version": "dynacomsec.trajectory.v1",
                "scenario_id": scenario.get("scenario_id"),
                "interaction_metadata": {
                    "mode": "multi_agent_adaptive",
                    "attacker_model": attacker_model,
                    "victim_runtime": self.NAME,
                    "victim_model": self.model_name or "",
                    "max_rounds": round_limit,
                    "rounds_completed": len(transcript),
                    "stop_reason": stop_reason,
                    "attackers": sorted(
                        {
                            str(step.get("speaker_agent_id"))
                            for step in transcript
                            if step.get("speaker_agent_id")
                        }
                    ),
                    "communities": sorted(
                        {str(step.get("community")) for step in transcript if step.get("community")}
                    ),
                    "session_manager": session_manager.summary(),
                },
                "steps": transcript,
            }
            await _write_json(environment, "/logs/agent/trajectory.json", trajectory)
            await _write_json(environment, "/logs/agent/sessions.json", session_manager.summary())
            await _write_text(environment, "/logs/agent/run_log.txt", _combined_log(transcript))

        async def _run_victim_once(
            self,
            instruction: str,
            template: str,
            environment: "BaseEnvironment",
            *,
            log_path: str,
            session_id: str = "",
            community_id: str = "",
            agent_id: str = "",
            session_manager: SessionManager | None = None,
        ) -> str:
            instruction_file = "/tmp/dynacomsec_instruction.md"
            if self.session_mode == "native":
                if self.native_adapter is None or session_manager is None:
                    raise RuntimeError(f"{self.NAME} native session adapter is not initialized")
                invocation = self.native_adapter.build_invocation(
                    manager=session_manager,
                    session_id=session_id,
                    community_id=community_id,
                    agent_id=agent_id,
                    instruction_file=instruction_file,
                    model=self.model_name or "",
                    usage_file=_usage_file_for_log(log_path),
                )
                command = invocation.command
            else:
                command = _format_command(
                    template,
                    instruction=instruction,
                    instruction_file=instruction_file,
                    model=self.model_name or "",
                    session_id=session_id,
                    community_id=community_id,
                    agent_id=agent_id,
                    usage_file=_usage_file_for_log(log_path) or "",
                )
            for attempt in range(1, self.transient_retry_attempts + 1):
                try:
                    result = await self.exec_as_agent(
                        environment,
                        command=(
                            "set -o pipefail; "
                            f"printf '%s' {shlex.quote(instruction)} > {instruction_file} && "
                            f"( {command} ) 2>&1 | tee {shlex.quote(log_path)}"
                        ),
                        env={
                            "DYNACOMSEC_AGENT_MODEL": self.model_name or "",
                            "DYNACOMSEC_INSTRUCTION": instruction,
                            "DYNACOMSEC_SESSION_ID": session_id,
                            "DYNACOMSEC_COMMUNITY_ID": community_id,
                            "DYNACOMSEC_TARGET_AGENT_ID": agent_id,
                        },
                        timeout_sec=self.timeout_seconds,
                    )
                    stdout = str(getattr(result, "stdout", "") or "")
                    if getattr(result, "return_code", 1) != 0:
                        raise RuntimeError(
                            f"{self.NAME} turn failed: "
                            + str(getattr(result, "stderr", "") or stdout).strip()
                        )
                    if self.session_mode == "native":
                        return self.native_adapter.complete_invocation(
                            manager=session_manager,
                            session_id=session_id,
                            stdout=stdout,
                        )
                    return stdout.strip()
                except Exception as exc:
                    if attempt >= self.transient_retry_attempts or not _is_transient_model_error(exc):
                        raise
                    await asyncio.sleep(self.transient_retry_sleep_seconds)


def _format_command(
    template: str,
    *,
    instruction: str,
    instruction_file: str,
    model: str,
    session_id: str = "",
    community_id: str = "",
    agent_id: str = "",
    usage_file: str = "",
) -> str:
    replacements = {
        "{instruction}": shlex.quote(instruction),
        "{instruction_file}": shlex.quote(instruction_file),
        "{model}": shlex.quote(model),
        "{session_id}": shlex.quote(session_id),
        "{community_id}": shlex.quote(community_id),
        "{agent_id}": shlex.quote(agent_id),
        "{usage_file}": shlex.quote(usage_file),
    }
    command = template
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


def _is_transient_model_error(error: Exception) -> bool:
    """Return true only for provider failures safe to retry at the same step."""

    text = str(error).lower()
    markers = (
        "currently overloaded",
        "temporarily overloaded",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "too many requests",
        # B.AI gate: "Concurrency limit exceeded for user, please retry later"
        "concurrency limit",
        # B.AI gate: "Upstream request failed" — provider's upstream model call
        # failed; transient, safe to retry at the same step.
        "upstream request failed",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "status=429",
        "status=500",
        "status=502",
        "status=503",
    )
    return any(marker in text for marker in markers)


async def _load_scenario(environment: "BaseEnvironment") -> dict:
    result = await environment.exec('cat "$DYNACOMSEC_SCENARIO"', timeout_sec=10)
    if getattr(result, "return_code", 1) != 0:
        raise RuntimeError(f"failed to read scenario: {getattr(result, 'stderr', '')}")
    return json.loads(result.stdout)


async def _write_text(environment: "BaseEnvironment", path: str, content: str) -> None:
    # Dynamic state updates can occur after an agent has initialized or
    # recreated its workspace. Ensure the target directory exists for both
    # small shell writes and larger docker uploads.
    parent = os.path.dirname(path)
    if parent:
        result = await environment.exec(
            f"mkdir -p {shlex.quote(parent)}",
            timeout_sec=10,
        )
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError(f"failed to create parent directory for {path}: {getattr(result, 'stderr', '')}")
    if len(content.encode("utf-8")) > 100_000 and hasattr(environment, "upload_file"):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(content)
            temp_path = handle.name
        try:
            await environment.upload_file(temp_path, path)
            return
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    command = f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
    result = await environment.exec(command, timeout_sec=10)
    if getattr(result, "return_code", 1) != 0:
        raise RuntimeError(f"failed to write {path}: {getattr(result, 'stderr', '')}")


async def _write_json(environment: "BaseEnvironment", path: str, value: dict) -> None:
    await _write_text(environment, path, json.dumps(value, ensure_ascii=False, indent=2))


def _combined_log(steps: list[dict]) -> str:
    chunks = []
    for step in steps:
        number = step.get("round") or step.get("step")
        chunks.append(
            f"Turn {number} [{step.get('community', '')}] session={step.get('session_id', '')}\n"
            f"Speaker ({step.get('speaker_role', 'unknown')}):\n{step.get('message', '')}\n\n"
            f"Victim:\n{step.get('response', '')}"
        )
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def _usage_file_for_log(log_path: str) -> str | None:
    if not log_path.startswith("/logs/agent/") or not log_path.endswith(".txt"):
        return None
    stem = log_path.rsplit("/", 1)[-1][:-4]
    return f"/logs/agent/usage-{stem}.json"


def _speaker_role_from_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "speaker"
    try:
        payload = json.loads(messages[-1].get("content", "{}"))
    except json.JSONDecodeError:
        return "speaker"
    if not isinstance(payload, dict):
        return "speaker"
    speaker = payload.get("speaker") if isinstance(payload.get("speaker"), dict) else {}
    current = payload.get("current_step") if isinstance(payload.get("current_step"), dict) else {}
    return str(speaker.get("speaker_role") or current.get("speaker_role") or "speaker")


def _normalize_speaker_mode(mode: str | None) -> str:
    normalized = str(mode or "llm").strip().lower().replace("_", "-")
    if normalized == "attacker-only":
        normalized = "attacker"
    if normalized not in {"llm", "seed", "attacker"}:
        raise ValueError("speaker_mode must be 'llm', 'seed', or 'attacker'")
    return normalized


def _normalize_guardrail_mode(mode: str | None) -> str:
    normalized = str(mode or "none").strip().lower().replace("-", "_")
    if normalized not in {"none", "first_turn"}:
        raise ValueError("guardrail_mode must be 'none' or 'first_turn'")
    return normalized


def _speaker_client_config(
    *,
    role: str,
    role_config: dict,
    default_model: str,
    default_api_key: str,
    default_base_url: str | None,
    extra_env: dict[str, str],
) -> tuple[str, str, str | None]:
    prefix = "DYNACOMSEC_NORMAL" if role == "normal" else "DYNACOMSEC_ATTACKER"
    model = (
        extra_env.get(f"{prefix}_MODEL")
        or os.environ.get(f"{prefix}_MODEL")
        or role_config.get("model")
        or default_model
    )
    api_key = (
        extra_env.get(f"{prefix}_API_KEY")
        or os.environ.get(f"{prefix}_API_KEY")
        or role_config.get("api_key")
        or default_api_key
    )
    base_url = (
        extra_env.get(f"{prefix}_BASE_URL")
        or os.environ.get(f"{prefix}_BASE_URL")
        or role_config.get("base_url")
        or default_base_url
    )
    return str(model), str(api_key), str(base_url) if base_url else None


class DynaComSecGenericCliHarness(_BaseHarness):
    NAME = "generic-cli"
    COMMAND_ENV_VAR = "DYNACOMSEC_AGENT_COMMAND"

    @staticmethod
    def name() -> str:
        return "generic-cli"


class DynaComSecClaudeCodeCliHarness(_BaseHarness):
    NAME = "claude-code"
    COMMAND_ENV_VAR = "DYNACOMSEC_CLAUDE_CODE_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = "claude --model {model} --print < {instruction_file}"
    NATIVE_ADAPTER_NAME = "claude-session-uuid"

    @staticmethod
    def name() -> str:
        return "claude-code"


class DynaComSecCodexCliHarness(_BaseHarness):
    NAME = "codex"
    COMMAND_ENV_VAR = "DYNACOMSEC_CODEX_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = (
        'response_file="/tmp/dynacomsec-codex-last-message-$$.txt"; '
        'event_file="/tmp/dynacomsec-codex-events-$$.jsonl"; '
        'mkdir -p "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        'cd "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        "codex exec --json --skip-git-repo-check --sandbox danger-full-access "
        '--model {model} --output-last-message "$response_file" '
        '< {instruction_file} > "$event_file" 2>&1; '
        "status=$?; "
        'cat "$response_file" 2>/dev/null || true; '
        'rm -f "$response_file" "$event_file"; '
        "exit $status"
    )
    NATIVE_ADAPTER_NAME = "codex-conversation-uuid"

    @staticmethod
    def name() -> str:
        return "codex"


class DynaComSecHermesCliHarness(_BaseHarness):
    NAME = "hermes"
    COMMAND_ENV_VAR = "DYNACOMSEC_HERMES_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = (
        'mkdir -p "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        'cd "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}"; '
        "model={model}; "
        "args=''; "
        "if [ -n \"$model\" ]; then args=\"--model $model\"; fi; "
        "usage_file={usage_file}; "
        "if [ -n \"$usage_file\" ]; then args=\"$args --usage-file $usage_file\"; fi; "
        'hermes -z "$(cat {instruction_file})" --provider custom $args '
        '--in "${DYNACOMSEC_WORKSPACE_ROOT:-/workspace/agent}" '
        "--ignore-rules"
    )
    NATIVE_ADAPTER_NAME = "hermes-session-id"

    @staticmethod
    def name() -> str:
        return "hermes"


class DynaComSecOpenClawCliHarness(_BaseHarness):
    NAME = "openclaw"
    COMMAND_ENV_VAR = "DYNACOMSEC_OPENCLAW_COMMAND"
    DEFAULT_COMMAND_TEMPLATE = "openclaw agent --local --agent main --message {instruction} --json"
    NATIVE_ADAPTER_NAME = "openclaw-session-key"

    @staticmethod
    def name() -> str:
        return "openclaw"
