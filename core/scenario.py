"""Scenario loading, placeholder expansion, and lightweight validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PATH_VARS: dict[str, tuple[str, str]] = {
    # Backward-compatible mapping for existing scenario JSON files. This must
    # precede the generic AGENT_HOME_DIR replacement.
    "${AGENT_HOME_DIR}workspace/": ("DYNACOMSEC_WORKSPACE_ROOT", "/workspace/agent/"),
    "${FIXTURE_DIR}": ("FIXTURE_DIR", "/root/workspace/"),
    "${AGENT_HOME_DIR}": ("AGENT_HOME_DIR", "/workspace/"),
    "${RUNTIME_TMP_DIR}": ("RUNTIME_TMP_DIR", "/tmp/"),
}


V2_SCHEMA_VERSION = "dynacomsec.scenario.v2"


def load_scenario(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        scenario = json.load(f)
    errors = validate_scenario(scenario)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid DynaComSec scenario {path}:\n{joined}")
    return scenario


def validate_scenario(scenario: dict) -> list[str]:
    errors = []
    if scenario.get("schema_version") != V2_SCHEMA_VERSION:
        errors.append(f"schema_version must be {V2_SCHEMA_VERSION}")
    required = [
        "scenario_id",
        "category",
        "task_family",
        "attack_type",
        "users",
        "agents",
        "target_agent",
        "world_state",
        "agent_views",
        "environment_setup",
        "participant_simulation",
        "conversation_script",
        "evaluator_private",
    ]
    for field in required:
        if field not in scenario:
            errors.append(f"missing required field: {field}")
    has_script = bool(scenario.get("conversation_script"))
    has_adaptive = bool(scenario.get("agents")) and bool(scenario.get("interaction_protocol"))
    if not has_script and not has_adaptive:
        errors.append("scenario must include either conversation_script or agents + interaction_protocol")
    target = scenario.get("target_agent", {})
    if has_script and (not isinstance(target, dict) or not target.get("agent_id")):
        errors.append("target_agent.agent_id is required when using conversation_script")
    runtime = scenario.get("dynacomsec_runtime", {})
    if isinstance(runtime, dict) and runtime.get("target_agent_context") is not None:
        errors.append(
            "dynacomsec_runtime.target_agent_context is not allowed; "
            "use agent_views.<community>.system_context for victim-visible system prompts"
        )
    allowed_speaker_roles = {"attacker", "normal", "owner", "system_event"}
    for index, step in enumerate(scenario.get("conversation_script", []), start=1):
        if not step.get("message"):
            errors.append(f"conversation_script[{index}] missing message")
        if not step.get("community"):
            errors.append(f"conversation_script[{index}] missing community")
        role = step.get("speaker_role")
        if not role:
            errors.append(f"conversation_script[{index}] missing speaker_role")
        elif role not in allowed_speaker_roles:
            errors.append(
                f"conversation_script[{index}].speaker_role must be one of "
                f"{sorted(allowed_speaker_roles)}"
            )
    if has_adaptive:
        agents = scenario.get("agents", [])
        roles = {agent.get("role") for agent in agents if isinstance(agent, dict)}
        if "attacker" not in roles:
            errors.append("agents must include one role='attacker' entry")
        if "victim" not in roles:
            errors.append("agents must include one role='victim' entry")
        protocol = scenario.get("interaction_protocol", {})
        if protocol.get("mode") != "multi_agent_adaptive":
            errors.append("interaction_protocol.mode must be multi_agent_adaptive")
        has_route = bool(protocol.get("community") or protocol.get("turn_schedule"))
        if not has_route:
            attacker_has_route = any(
                isinstance(agent, dict)
                and agent.get("role") == "attacker"
                and (agent.get("community") or agent.get("communities"))
                for agent in agents
            )
            if not attacker_has_route:
                errors.append(
                    "adaptive scenarios require interaction_protocol.community, "
                    "turn_schedule, or attacker community metadata"
                )
    return errors


def world_state(scenario: dict) -> dict:
    value = scenario.get("world_state")
    return value if isinstance(value, dict) else {}


def evaluator_private(scenario: dict) -> dict:
    value = scenario.get("evaluator_private")
    return value if isinstance(value, dict) else {}


def communities(scenario: dict) -> list:
    value = world_state(scenario).get("communities", [])
    return value if isinstance(value, list) else []


def community_relations(scenario: dict) -> list:
    value = world_state(scenario).get("community_relations", [])
    return value if isinstance(value, list) else []


def dynamic_events(scenario: dict) -> list:
    value = world_state(scenario).get("dynamic_events", [])
    return value if isinstance(value, list) else []


def info_items(scenario: dict) -> list:
    value = evaluator_private(scenario).get("info_items", [])
    return value if isinstance(value, list) else []


def leakage_labels(scenario: dict) -> dict:
    value = evaluator_private(scenario).get("leakage_labels", {})
    return value if isinstance(value, dict) else {}


def success_criteria(scenario: dict) -> dict:
    value = evaluator_private(scenario).get("success_criteria", {})
    return value if isinstance(value, dict) else {}


def oracles(scenario: dict) -> dict:
    value = evaluator_private(scenario).get("oracles", {})
    return value if isinstance(value, dict) else {}


def resolved_path_vars() -> dict[str, str]:
    return {
        placeholder: _ensure_trailing_slash(os.environ.get(env_name, default))
        for placeholder, (env_name, default) in PATH_VARS.items()
    }


def _ensure_trailing_slash(path: str) -> str:
    return path if path.endswith("/") else path + "/"


def expand_placeholders(value, mapping: dict[str, str]):
    if isinstance(value, str):
        for placeholder, replacement in mapping.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [expand_placeholders(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: expand_placeholders(item, mapping) for key, item in value.items()}
    return value


def session_id_for(scenario_id: str, agent_id: str, community_id: str) -> str:
    """Stable per-community session key for one target personal agent."""

    digest = hashlib.sha256(f"{scenario_id}::{agent_id}::{community_id}".encode("utf-8")).hexdigest()[:16]
    return f"session-{digest}::{agent_id}::{community_id}"
