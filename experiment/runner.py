"""Harbor-style experiment runner for DynaComSec."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dynacomsec.core.scenario import communities, info_items
from dynacomsec.experiment.agents.registry import IN_TREE_AGENT_IMPORT_PATHS, available_agents
from dynacomsec.experiment.docker_env import DEFAULT_IMAGE
from dynacomsec.experiment.model_registry import (
    materialize_codex_profile,
    materialize_hermes_profile,
    load_model_profile,
    materialize_openclaw_profile,
    profile_env,
    profile_unset,
)
from dynacomsec.experiment.task_builder import build_task_dir, temporary_task_dir
from dynacomsec.scripts.export_data_flow_report import write_reports


DOCKER_ENVIRONMENT_IMPORT_PATH = "dynacomsec.experiment.docker_env:DynaComSecDockerEnvironment"

AGENT_ENV_ALLOWLIST: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "GPTOSS_ZZ_API_KEY",
    "HERMES_GPTOSS_ZZ_API_KEY",
    "CODEX_GPTOSS_ZZ_API_KEY",
    "OPENCLAW_LLAMA4_ZZ_API_KEY",
    "HERMES_LLAMA4_ZZ_API_KEY",
    "OPENCLAW_OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_KIMI_API_KEY",
    "ALIYUN_MINIMAX_API_KEY",
    "ALIYUN_GLM52_API_KEY",
    "DEEPROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
    "TOGETHER_API_KEY",
    "DYNACOMSEC_AGENT_COMMAND",
    "DYNACOMSEC_CLAUDE_CODE_COMMAND",
    "DYNACOMSEC_CODEX_COMMAND",
    "DYNACOMSEC_HERMES_COMMAND",
    "DYNACOMSEC_OPENCLAW_COMMAND",
    "DYNACOMSEC_SPEAKER_API_KEY",
    "DYNACOMSEC_SPEAKER_BASE_URL",
    "DYNACOMSEC_SPEAKER_MODEL",
    "DYNACOMSEC_NORMAL_API_KEY",
    "DYNACOMSEC_NORMAL_BASE_URL",
    "DYNACOMSEC_NORMAL_MODEL",
    "DYNACOMSEC_ATTACKER_API_KEY",
    "DYNACOMSEC_ATTACKER_BASE_URL",
    "DYNACOMSEC_ATTACKER_MODEL",
    "DYNACOMSEC_JUDGE_API_KEY",
    "DYNACOMSEC_JUDGE_BASE_URL",
    "DYNACOMSEC_JUDGE_MODEL",
    "DYNACOMSEC_OPENCLAW_HOME_HOST",
    "DYNACOMSEC_CODEX_HOME_HOST",
    "DYNACOMSEC_CLAUDE_HOME_HOST",
    "DYNACOMSEC_HERMES_HOME_HOST",
)


def build_harbor_command(
    *,
    task_dir: str | Path,
    agent: str,
    model: str | None = None,
    trials_dir: str | None = None,
    image: str = DEFAULT_IMAGE,
    force_build: bool = False,
    agent_command: str | None = None,
    session_mode: str = "replay",
    context_mode: str = "strong",
    speaker_mode: str = "llm",
    guardrail_mode: str = "none",
    transient_retry_attempts: int = 1,
    transient_retry_sleep_seconds: float = 0,
    env: dict[str, str] | None = None,
    openclaw_home_host: str | None = None,
    codex_home_host: str | None = None,
    claude_home_host: str | None = None,
    hermes_home_host: str | None = None,
    unset_env: list[str] | None = None,
) -> list[str]:
    cmd = [_resolve_harbor_bin(), "trials", "start", "-p", str(task_dir)]

    import_path = IN_TREE_AGENT_IMPORT_PATHS.get(agent)
    if not import_path:
        options = ", ".join(available_agents())
        raise ValueError(f"unknown DynaComSec experiment agent: {agent}. Available: {options}")
    cmd += ["--agent-import-path", import_path]

    if model:
        cmd += ["-m", model]
    cmd += ["--environment-import-path", DOCKER_ENVIRONMENT_IMPORT_PATH]
    cmd += ["--environment-kwarg", f"image={image}"]
    for key, value in (
        ("openclaw_home_host", openclaw_home_host),
        ("codex_home_host", codex_home_host),
        ("claude_home_host", claude_home_host),
        ("hermes_home_host", hermes_home_host),
    ):
        if value:
            cmd += ["--environment-kwarg", f"{key}={value}"]
    if unset_env:
        cmd += ["--environment-kwarg", f"unset_env={json.dumps(unset_env, separators=(',', ':'))}"]
    if force_build:
        cmd += ["--environment-kwarg", "force_build=true"]
    if trials_dir:
        cmd += ["--trials-dir", trials_dir]
    if agent_command:
        cmd += ["--agent-kwarg", f"command={agent_command}"]
    if session_mode not in {"replay", "native"}:
        raise ValueError(f"unsupported session mode: {session_mode}")
    cmd += ["--agent-kwarg", f"session_mode={session_mode}"]
    if context_mode not in {"strong", "natural", "light"}:
        raise ValueError(f"unsupported context mode: {context_mode}")
    cmd += ["--agent-kwarg", f"context_mode={context_mode}"]
    if speaker_mode not in {"llm", "seed", "attacker"}:
        raise ValueError(f"unsupported speaker mode: {speaker_mode}")
    cmd += ["--agent-kwarg", f"speaker_mode={speaker_mode}"]
    if guardrail_mode not in {"none", "first_turn"}:
        raise ValueError(f"unsupported guardrail mode: {guardrail_mode}")
    cmd += ["--agent-kwarg", f"guardrail_mode={guardrail_mode}"]
    cmd += ["--agent-kwarg", f"transient_retry_attempts={int(transient_retry_attempts)}"]
    cmd += ["--agent-kwarg", f"transient_retry_sleep_seconds={float(transient_retry_sleep_seconds)}"]
    for key, value in _forward_agent_env(env or dict(os.environ)):
        cmd += ["--ae", f"{key}={value}"]
    return cmd


def run_experiment(
    *,
    scenario: str,
    agent: str = "codex",
    model: str | None = None,
    model_profile: str | None = None,
    model_registry: list[str | Path] | None = None,
    task_dir: str | None = None,
    trials_dir: str = "dynacomsec/results/experiments",
    image: str = DEFAULT_IMAGE,
    force_build: bool = False,
    execute_setup: bool = True,
    judge: bool = False,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    agent_command: str | None = None,
    session_mode: str = "replay",
    context_mode: str = "strong",
    speaker_mode: str = "llm",
    guardrail_mode: str = "none",
    transient_retry_attempts: int = 1,
    transient_retry_sleep_seconds: float = 0,
    speaker_model: str | None = None,
    speaker_base_url: str | None = None,
    normal_model: str | None = None,
    normal_base_url: str | None = None,
    attacker_model: str | None = None,
    attacker_base_url: str | None = None,
    enable_mcp: bool = False,
    show: bool = False,
    dry_run: bool = False,
) -> list[str]:
    agent_env = dict(os.environ)
    generated_profile_home: str | None = None
    profile_home_kwargs: dict[str, str] = {}
    profile_unset_keys: list[str] = []
    if model_profile:
        profile = load_model_profile(model_profile, registry_paths=model_registry)
        agent = str(profile.get("agent") or agent)
        model = model or str(profile.get("model") or "")
        image = str(profile.get("image") or image)
        session_mode = str(profile.get("session_mode") or session_mode)
        context_mode = str(profile.get("context_mode") or context_mode)
        profile_unset_keys = profile_unset(profile)
        for key in profile_unset_keys:
            agent_env.pop(key, None)
        agent_env.update(profile_env(profile))
        if agent == "openclaw":
            generated_profile_home = materialize_openclaw_profile(profile, agent_env)
            if generated_profile_home:
                agent_env["DYNACOMSEC_OPENCLAW_HOME_HOST"] = generated_profile_home
                profile_home_kwargs["openclaw_home_host"] = generated_profile_home
        elif agent == "hermes":
            generated_profile_home = materialize_hermes_profile(profile, agent_env)
            if generated_profile_home:
                agent_env["DYNACOMSEC_HERMES_HOME_HOST"] = generated_profile_home
                profile_home_kwargs["hermes_home_host"] = generated_profile_home
        elif agent == "codex":
            generated_profile_home = materialize_codex_profile(profile, agent_env)
            if generated_profile_home:
                agent_env["DYNACOMSEC_CODEX_HOME_HOST"] = generated_profile_home
                profile_home_kwargs["codex_home_host"] = generated_profile_home

    if speaker_model or attacker_model:
        agent_env["DYNACOMSEC_SPEAKER_MODEL"] = speaker_model or attacker_model or ""
    if speaker_base_url or attacker_base_url:
        agent_env["DYNACOMSEC_SPEAKER_BASE_URL"] = speaker_base_url or attacker_base_url or ""
    if normal_model:
        agent_env["DYNACOMSEC_NORMAL_MODEL"] = normal_model
    if normal_base_url:
        agent_env["DYNACOMSEC_NORMAL_BASE_URL"] = normal_base_url
    if judge_base_url:
        agent_env["DYNACOMSEC_JUDGE_BASE_URL"] = judge_base_url
    if judge_model:
        agent_env["DYNACOMSEC_JUDGE_MODEL"] = judge_model
    if task_dir:
        task = build_task_dir(
            scenario,
            task_dir,
            execute_setup=execute_setup,
            judge=judge,
            judge_model=judge_model,
            enable_mcp=enable_mcp,
        )
        cmd = build_harbor_command(
            task_dir=task,
            agent=agent,
            model=model,
            trials_dir=trials_dir,
            image=image,
            force_build=force_build,
            agent_command=agent_command,
            session_mode=session_mode,
            context_mode=context_mode,
            speaker_mode=speaker_mode,
            guardrail_mode=guardrail_mode,
            transient_retry_attempts=transient_retry_attempts,
            transient_retry_sleep_seconds=transient_retry_sleep_seconds,
            env=agent_env,
            unset_env=profile_unset_keys,
            **profile_home_kwargs,
        )
        return _maybe_run(cmd, dry_run=dry_run, show=show, scenario_path=scenario, trials_dir=trials_dir)

    with temporary_task_dir(
        scenario,
        execute_setup=execute_setup,
        judge=judge,
        judge_model=judge_model,
        enable_mcp=enable_mcp,
    ) as task:
        cmd = build_harbor_command(
            task_dir=task,
            agent=agent,
            model=model,
            trials_dir=trials_dir,
            image=image,
            force_build=force_build,
            agent_command=agent_command,
            session_mode=session_mode,
            context_mode=context_mode,
            speaker_mode=speaker_mode,
            guardrail_mode=guardrail_mode,
            transient_retry_attempts=transient_retry_attempts,
            transient_retry_sleep_seconds=transient_retry_sleep_seconds,
            env=agent_env,
            unset_env=profile_unset_keys,
            **profile_home_kwargs,
        )
        return _maybe_run(cmd, dry_run=dry_run, show=show, scenario_path=scenario, trials_dir=trials_dir)


def _maybe_run(
    cmd: list[str],
    *,
    dry_run: bool,
    show: bool = False,
    scenario_path: str | Path | None = None,
    trials_dir: str | Path | None = None,
) -> list[str]:
    print("Running:", _redact(cmd))
    if dry_run:
        return cmd
    before = _trial_dirs(trials_dir) if trials_dir else set()
    result = subprocess.run(cmd, check=False, env=_child_env())
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    trial_dir = _latest_new_trial_dir(trials_dir, before) if trials_dir else None
    if trial_dir is not None:
        try:
            json_report, md_report = write_reports(trial_dir)
            print(f"[data-flow] wrote {json_report}")
            print(f"[data-flow] wrote {md_report}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[data-flow] warning: failed to write report: {exc}", file=sys.stderr)
    if show and scenario_path and trials_dir:
        if trial_dir is None:
            print("\n[show] Trial completed, but no result directory was found.")
        else:
            show_trial_conversation(scenario_path, trial_dir)
    return cmd


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(Path.cwd())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _forward_agent_env(env: dict[str, str]) -> list[tuple[str, str]]:
    return [(key, env[key]) for key in AGENT_ENV_ALLOWLIST if env.get(key)]


def _resolve_harbor_bin() -> str:
    sibling = Path(sys.executable).parent / "harbor"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("harbor")
    return found or "harbor"


def _redact(cmd: list[str]) -> str:
    redacted = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--ae" and i + 1 < len(cmd):
            key = cmd[i + 1].split("=", 1)[0]
            redacted.extend(["--ae", f"{key}=***"])
            i += 2
            continue
        redacted.append(cmd[i])
        i += 1
    return " ".join(redacted)


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "ture", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("--show expects true/false")


def _trial_dirs(trials_dir: str | Path | None) -> set[Path]:
    root = Path(trials_dir or "")
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def _latest_new_trial_dir(trials_dir: str | Path, before: set[Path]) -> Path | None:
    after = _trial_dirs(trials_dir)
    candidates = list(after - before) or list(after)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


_ANSI_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "purple": "\033[35m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _show_uses_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _show_color(text: str, color: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_ANSI_COLORS[color]}{text}{_ANSI_COLORS['reset']}"


def show_trial_conversation(scenario_path: str | Path, trial_dir: str | Path) -> None:
    scenario = _load_json_file(Path(scenario_path), {})
    trial = Path(trial_dir)
    trajectory = _load_json_file(trial / "agent" / "trajectory.json", {})
    scorecard = _load_json_file(trial / "verifier" / "scorecard.json", {})
    use_color = _show_uses_color()
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else []
    if not isinstance(steps, list):
        steps = []

    print("\n" + "=" * 88)
    print("DynaComSec Conversation")
    print("=" * 88)
    print(f"Trial: {trial.name}")
    print(f"Scenario: {scenario.get('scenario_id') or trajectory.get('scenario_id') or Path(scenario_path).stem}")
    print(f"Attack type: {scenario.get('attack_type') or scenario.get('task_family') or ''}")
    if scenario.get("description"):
        print(f"Description: {scenario.get('description')}")

    metadata = trajectory.get("interaction_metadata") if isinstance(trajectory, dict) else {}
    if isinstance(metadata, dict):
        print(f"Mode: {metadata.get('mode', '')}")
        if metadata.get("victim_runtime") or metadata.get("victim_model"):
            print(f"Target runtime: {metadata.get('victim_runtime', '')} / {metadata.get('victim_model', '')}")
        if metadata.get("speaker_model") or metadata.get("attacker_model"):
            print(f"Speaker model: {metadata.get('speaker_model') or metadata.get('attacker_model')}")
        if metadata.get("stop_reason"):
            print(f"Stop reason: {metadata.get('stop_reason')}")
    if scorecard:
        print(
            "Score: "
            f"reward={scorecard.get('binary_reward')} "
            f"status={scorecard.get('status')} "
            f"decision_source={scorecard.get('decision_source')}"
        )

    print("\nParticipants")
    print("-" * 88)
    _print_participants(scenario, steps)
    print("\nCommunities")
    print("-" * 88)
    _print_communities(scenario)
    print("\nSensitive Boundary")
    print("-" * 88)
    _print_sensitive_boundary(scenario)
    print("\nConversation")
    print("-" * 88)
    if not steps:
        print("(no trajectory steps found)")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        turn = step.get("round") or step.get("step") or index
        community = step.get("community", "")
        session_id = step.get("session_id", "")
        backend = step.get("backend_session_id") or step.get("backend_session_key") or ""
        attacker = step.get("speaker_agent_id") or step.get("speaker") or "attacker"
        victim = step.get("recipient_agent_id") or "target"
        attacker_name = _display_name(scenario, str(attacker), include_agent_id=False)
        victim_name = _display_name(scenario, str(victim), include_agent_id=False)
        attacker_text = str(step.get("message", "")).strip()
        victim_text = str(step.get("response", "")).strip()
        speaker_role = str(step.get("speaker_role") or "")
        is_system_event = speaker_role == "system_event" or str(attacker) in {"system_event", "trusted_event", "event"}
        print(f"\nTurn {turn} | community={community}")
        if session_id:
            print(f"Session: {session_id}")
        if backend:
            print(f"Native session: {backend}")
        if is_system_event:
            print(_show_color(f"\n[SYSTEM EVENT] {attacker_name} -> {victim_name}", "purple", enabled=use_color))
            print(_show_color(attacker_text, "purple", enabled=use_color))
        else:
            label = "NORMAL" if speaker_role == "normal" else "SPEAKER"
            color = "blue" if speaker_role == "normal" else "red"
            if speaker_role == "attacker":
                label = "ATTACKER"
            print(_show_color(f"\n[{label}] {attacker_name} -> {victim_name}", color, enabled=use_color))
            print(_show_color(attacker_text, color, enabled=use_color))
        print(_show_color(f"\n[TARGET AGENT] {victim_name} -> {attacker_name}", "green", enabled=use_color))
        print(_show_color(victim_text, "green", enabled=use_color))
    print("\n" + "=" * 88)


def _load_json_file(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _display_name(scenario: dict, identifier: str, *, include_agent_id: bool = True) -> str:
    agents_by_user = {
        str(agent.get("user_id", "")): agent
        for agent in scenario.get("agents", [])
        if isinstance(agent, dict)
    }
    for user in scenario.get("users", []):
        if not isinstance(user, dict):
            continue
        if identifier == str(user.get("user_id", "")):
            name = user.get("display_name") or user.get("user_id") or identifier
            agent_id = agents_by_user.get(str(user.get("user_id", "")), {}).get("agent_id")
            if include_agent_id and agent_id:
                return f"{name} ({agent_id})"
            return str(name)
    for agent in scenario.get("agents", []):
        if not isinstance(agent, dict):
            continue
        if identifier in {str(agent.get("agent_id", "")), str(agent.get("user_id", ""))}:
            name = agent.get("display_name") or agent.get("user_id") or identifier
            agent_id = agent.get("agent_id")
            if include_agent_id and agent_id:
                return f"{name} ({agent_id})"
            return str(name)
    return identifier


def _print_participants(scenario: dict, steps: list) -> None:
    target = scenario.get("target_agent") if isinstance(scenario.get("target_agent"), dict) else {}
    if target:
        print(
            "Target: "
            f"{target.get('display_name') or target.get('user_id') or ''} "
            f"({target.get('agent_id', '')})"
        )
    attackers = [
        agent for agent in scenario.get("agents", [])
        if isinstance(agent, dict) and agent.get("role") == "attacker"
    ]
    if not attackers:
        seen = []
        for step in steps:
            if isinstance(step, dict) and step.get("speaker_agent_id") not in seen:
                seen.append(step.get("speaker_agent_id"))
        attackers = [{"agent_id": item} for item in seen if item]
    for attacker in attackers:
        communities = ", ".join(str(item) for item in attacker.get("communities", []))
        print(
            "Attacker: "
            f"{attacker.get('display_name') or attacker.get('user_id') or attacker.get('agent_id')} "
            f"({attacker.get('agent_id', '')})"
            + (f" communities=[{communities}]" if communities else "")
        )
        if attacker.get("private_goal"):
            print(f"  Goal: {attacker.get('private_goal')}")
    owners = [
        agent for agent in scenario.get("agents", [])
        if isinstance(agent, dict) and agent.get("role") in {"owner", "information_owner", "third"}
    ]
    for owner in owners:
        communities = ", ".join(str(item) for item in owner.get("communities", []))
        print(
            "Information owner/third: "
            f"{owner.get('display_name') or owner.get('user_id') or owner.get('agent_id')} "
            f"({owner.get('agent_id', '')})"
            + (f" communities=[{communities}]" if communities else "")
        )


def _print_communities(scenario: dict) -> None:
    for community in communities(scenario):
        if not isinstance(community, dict):
            continue
        members = []
        for member in community.get("members", []):
            if not isinstance(member, dict):
                continue
            label = _display_name(scenario, str(member.get("user_id", "")))
            role = member.get("role")
            status = member.get("status")
            details = ", ".join(str(item) for item in (role, status) if item)
            members.append(f"{label}" + (f" [{details}]" if details else ""))
        print(f"{community.get('community_id')}: {community.get('name', '')}")
        if members:
            print("  Members: " + "; ".join(members))
        norms = community.get("privacy_norms")
        if isinstance(norms, dict):
            for key, value in norms.items():
                print(f"  {key}: {value}")


def _print_sensitive_boundary(scenario: dict) -> None:
    items = info_items(scenario)
    if not isinstance(items, list) or not items:
        print("(no info_items declared)")
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        owner = _display_name(scenario, str(item.get("owner", "")))
        print(f"{item.get('info_id', '')}: owner={owner} source={item.get('source_community', '')}")
        if item.get("sensitivity") or item.get("sensitivity_domain"):
            print(f"  Sensitivity: {item.get('sensitivity') or item.get('sensitivity_domain')}")
        if item.get("allowed_communities"):
            print(f"  Allowed communities: {', '.join(map(str, item.get('allowed_communities', [])))}")
        if item.get("forbidden_communities"):
            print(f"  Forbidden communities: {', '.join(map(str, item.get('forbidden_communities', [])))}")
        if item.get("forbidden_recipients"):
            recipients = [_display_name(scenario, str(recipient)) for recipient in item.get("forbidden_recipients", [])]
            print(f"  Forbidden recipients: {', '.join(recipients)}")
        if item.get("safe_abstraction"):
            print(f"  Safe abstraction: {item.get('safe_abstraction')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DynaComSec scenario through Harbor/Docker.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent", choices=available_agents(), default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Named profile from dynacomsec/configs/model_registry*.toml. Sets agent/image/model/env defaults.",
    )
    parser.add_argument(
        "--model-registry",
        action="append",
        default=None,
        help="Extra TOML registry path. Repeatable; later files override default registry entries.",
    )
    parser.add_argument("--task-dir", default=None, help="Persist generated Harbor task here.")
    parser.add_argument("--trials-dir", default="dynacomsec/results/experiments")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-execute-setup", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--agent-command", default=None)
    parser.add_argument(
        "--session-mode",
        choices=("replay", "native"),
        default="replay",
        help="Replay isolated DynaComSec history or rely on the Agent CLI's native session/resume support.",
    )
    parser.add_argument(
        "--context-mode",
        choices=("strong", "natural", "light"),
        default="strong",
        help="Control victim-agent context detail: strong repeats explicit policy, natural uses workspace-style guidance, light keeps later native turns minimal.",
    )
    parser.add_argument(
        "--speaker-llm",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Use LLM-generated scripted participants. false uses only the fixed JSON messages.",
    )
    parser.add_argument(
        "--speaker-mode",
        choices=("seed", "llm", "attacker"),
        default=None,
        help=(
            "Scripted participant generation mode. seed uses fixed JSON messages; "
            "llm generates all eligible participants; attacker generates only attacker turns. "
            "Overrides --speaker-llm when set."
        ),
    )
    parser.add_argument(
        "--guardrail-mode",
        choices=("none", "first_turn"),
        default="none",
        help="Optional compact safety guardrail injected into the first victim-agent turn per session.",
    )
    parser.add_argument("--transient-retry-attempts", type=int, default=1)
    parser.add_argument("--transient-retry-sleep", type=float, default=0)
    parser.add_argument(
        "--enable-mcp",
        action="store_true",
        help="Enable the optional DynaComSec community MCP server on :8120.",
    )
    parser.add_argument(
        "--speaker-model",
        default=None,
        help="LLM used by generated scripted participants (default: DYNACOMSEC_SPEAKER_MODEL/OPENAI_MODEL/gpt-4o).",
    )
    parser.add_argument(
        "--speaker-base-url",
        default=None,
        help="OpenAI-compatible endpoint for generated scripted participants.",
    )
    parser.add_argument(
        "--normal-model",
        default=None,
        help="Optional LLM override for generated normal participants.",
    )
    parser.add_argument(
        "--normal-base-url",
        default=None,
        help="Optional OpenAI-compatible endpoint override for generated normal participants.",
    )
    parser.add_argument(
        "--attacker-model",
        default=None,
        help="Deprecated alias for --speaker-model.",
    )
    parser.add_argument(
        "--attacker-base-url",
        default=None,
        help="Deprecated alias for --speaker-base-url.",
    )
    parser.add_argument(
        "--show",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Print scenario participants, sensitive boundary, and full conversation after the trial. Accepts true/false; 'ture' is accepted as a common typo.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print Harbor command without executing it.")
    args = parser.parse_args()
    speaker_mode = args.speaker_mode or ("llm" if args.speaker_llm else "seed")
    run_experiment(
        scenario=args.scenario,
        agent=args.agent,
        model=args.model,
        model_profile=args.model_profile,
        model_registry=args.model_registry,
        task_dir=args.task_dir,
        trials_dir=args.trials_dir,
        image=args.image,
        force_build=args.force_build,
        execute_setup=not args.no_execute_setup,
        judge=args.judge,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        agent_command=args.agent_command,
        session_mode=args.session_mode,
        context_mode=args.context_mode,
        speaker_mode=speaker_mode,
        guardrail_mode=args.guardrail_mode,
        transient_retry_attempts=args.transient_retry_attempts,
        transient_retry_sleep_seconds=args.transient_retry_sleep,
        speaker_model=args.speaker_model,
        speaker_base_url=args.speaker_base_url,
        normal_model=args.normal_model,
        normal_base_url=args.normal_base_url,
        attacker_model=args.attacker_model,
        attacker_base_url=args.attacker_base_url,
        enable_mcp=args.enable_mcp,
        show=args.show,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
