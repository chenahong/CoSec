"""Model profile registry for DynaComSec experiment runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATHS = (
    _PACKAGE_ROOT / "configs/model_registry.local.toml",
)

_RESERVED_PROFILE_ENV_PREFIXES = (
    "DYNACOMSEC_ATTACKER_",
    "DYNACOMSEC_JUDGE_",
)
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_model_profile(
    name: str,
    *,
    registry_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Load a named model profile from public and local registry files."""

    merged: dict[str, Any] = {}
    configured_paths = (
        list(DEFAULT_REGISTRY_PATHS) + list(registry_paths)
        if registry_paths
        else list(DEFAULT_REGISTRY_PATHS)
    )
    paths = [Path(path).expanduser() for path in configured_paths]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        formatted = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"no model registry file found. Checked: {formatted}")

    for path in existing:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        _deep_merge(merged, data)

    profiles = merged.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("model registry must contain a [profiles] table")
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles)) or "<none>"
        raise KeyError(f"unknown model profile {name!r}. Available profiles: {available}")
    profile = _resolve_profile(name, profiles)
    _validate_profile_env(name, profile)
    result = dict(profile)
    result["profile_name"] = name
    return result


def materialize_openclaw_profile(
    profile: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Write a temporary OpenClaw home for profiles with inline OpenClaw config."""

    openclaw = profile.get("openclaw")
    if not isinstance(openclaw, Mapping):
        return None
    models = openclaw.get("models")
    if not isinstance(models, Mapping):
        return None
    models = dict(_expand_env_refs(models, env))
    primary_model = str(models.pop("default", "") or profile.get("model") or "")
    if "mode" not in models:
        models["mode"] = "replace"

    profile_name = str(profile.get("profile_name") or "profile").replace("/", "_")
    root = Path(tempfile.mkdtemp(prefix=f"dynacomsec-openclaw-{profile_name}-")).resolve()
    root.chmod(0o755)

    openclaw_json: dict[str, Any] = {"models": models}
    if primary_model:
        openclaw_json["agents"] = {
            "defaults": {
                "model": {"primary": primary_model},
                "models": {primary_model: {}},
            }
        }
    (root / "openclaw.json").write_text(
        json.dumps(openclaw_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # OpenClaw resolves provider credentials from the agent auth store, not
    # from the model catalog's apiKey field alone.  The store entry has to be
    # keyed by the same provider name the model catalog uses, otherwise the
    # credential is never matched.  Derive that name from the profile's own
    # model id ("<provider>/<model>") rather than pinning it to a single
    # vendor, so any profile can be repointed at any vendor by config alone.
    resolved_env = env or os.environ
    auth_provider = _openclaw_auth_provider(models, primary_model)
    api_key = str(
        resolved_env.get("DEEPROUTER_API_KEY")
        or resolved_env.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if auth_provider and api_key:
        auth_target = root / "agents/main/agent/auth-profiles.json"
        auth_target.parent.mkdir(parents=True, exist_ok=True)
        entry = f"{auth_provider}:default"
        auth_target.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        entry: {
                            "type": "api_key",
                            "provider": auth_provider,
                            "key": api_key,
                        }
                    },
                    "order": {auth_provider: [entry]},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for rel_path in ("agents/main/agent/models.json", "config/models.json"):
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(models, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return str(root)


def _openclaw_auth_provider(models: Mapping[str, Any], primary_model: str) -> str:
    """Provider name the OpenClaw auth store must be keyed by.

    Mirrors the model catalog: profiles name themselves after their vendor
    ("openai/<model>", "zhizengzeng/<model>", ...), so the prefix of the active
    model is the provider name.  Single-provider catalogs that omit the prefix
    (e.g. model "deepseek-v4-flash" under providers.deepseek) fall back to that
    one provider.  "deeprouter" remains the last resort for catalogs that
    declare no provider at all.
    """

    providers = models.get("providers")
    names = [str(name) for name in providers] if isinstance(providers, Mapping) else []
    prefix = primary_model.strip().split("/", 1)[0] if "/" in primary_model else ""
    if prefix and prefix in names:
        return prefix
    if len(names) == 1:
        return names[0]
    return prefix or "deeprouter"


def materialize_hermes_profile(profile: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    """Write a temporary Hermes home whose custom provider matches the profile."""

    if str(profile.get("agent") or "") != "hermes":
        return None
    model = str(profile.get("model") or env.get("OPENAI_MODEL") or "").strip()
    base_url = str(env.get("OPENAI_BASE_URL") or "").strip()
    api_key = str(env.get("OPENAI_API_KEY") or "").strip()
    # Hermes >=0.20 supports native Responses transport.  Keep the legacy
    # Chat Completions default for existing profiles, but allow a profile to
    # opt into the Responses wire format explicitly.
    api_mode = str(
        profile.get("hermes_api_mode")
        or profile.get("hermes_transport")
        or "chat_completions"
    ).strip()
    streaming_enabled = bool(profile.get("hermes_streaming", False))
    if not model or not base_url or not api_key:
        return None

    profile_name = str(profile.get("profile_name") or "profile").replace("/", "_")
    root = Path(tempfile.mkdtemp(prefix=f"dynacomsec-hermes-{profile_name}-")).resolve()
    root.chmod(0o700)

    config = "\n".join(
        [
            "model:",
            f"  default: {_yaml_string(model)}",
            "  provider: custom",
            f"  base_url: {_yaml_string(base_url)}",
            f"  api_key: {_yaml_string(api_key)}",
            f"  api_mode: {_yaml_string(api_mode)}",
            "agent:",
            "  max_turns: 90",
            "  disabled_toolsets:",
            "    - browser",
            "    - computer_use",
            "    - delegation",
            "    - image_gen",
            "    - memory",
            "    - search",
            "    - skills",
            "    - web",
            "  verbose: false",
            "streaming:",
            f"  enabled: {'true' if streaming_enabled else 'false'}",
            "custom_providers:",
            "  - name: DynaComSecProfile",
            f"    base_url: {_yaml_string(base_url)}",
            "    key_env: OPENAI_API_KEY",
            f"    model: {_yaml_string(model)}",
            f"    api_mode: {_yaml_string(api_mode)}",
            "",
        ]
    )
    (root / "config.yaml").write_text(config, encoding="utf-8")
    (root / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "active_provider": None,
                "credential_pool": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(root)


def materialize_codex_profile(profile: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    """Write a temporary Codex home for OpenAI-compatible model profiles."""

    if str(profile.get("agent") or "") != "codex":
        return None
    model = str(profile.get("model") or env.get("OPENAI_MODEL") or "").strip()
    base_url = str(env.get("OPENAI_BASE_URL") or "").strip()
    api_key = str(env.get("OPENAI_API_KEY") or "").strip()
    if not model or not base_url or not api_key:
        return None

    provider = str(profile.get("codex_provider") or "zhizengzeng").strip()
    wire_api = str(profile.get("codex_wire_api") or "responses").strip()
    profile_name = str(profile.get("profile_name") or "profile").replace("/", "_")
    root = Path(tempfile.mkdtemp(prefix=f"dynacomsec-codex-{profile_name}-")).resolve()
    root.chmod(0o700)

    config = "\n".join(
        [
            f"model_provider = {_toml_string(provider)}",
            "",
            f"[model_providers.{provider}]",
            f"name = {_toml_string(provider)}",
            f"base_url = {_toml_string(base_url)}",
            'env_key = "OPENAI_API_KEY"',
            f"wire_api = {_toml_string(wire_api)}",
            "",
        ]
    )
    (root / "config.toml").write_text(config, encoding="utf-8")
    return str(root)


def profile_env(profile: Mapping[str, Any]) -> dict[str, str]:
    env = profile.get("env")
    if not isinstance(env, Mapping):
        return {}
    expanded = {}
    for key, value in env.items():
        if value is None:
            continue
        resolved = str(_expand_env_refs(value))
        if resolved:
            expanded[str(key)] = resolved
    return expanded


def profile_unset(profile: Mapping[str, Any]) -> list[str]:
    unset = profile.get("unset")
    if not isinstance(unset, list):
        return []
    return [str(key) for key in unset]


def _validate_profile_env(name: str, profile: Mapping[str, Any]) -> None:
    env = profile.get("env")
    if not isinstance(env, Mapping):
        return
    blocked = [
        str(key)
        for key in env
        if any(str(key).startswith(prefix) for prefix in _RESERVED_PROFILE_ENV_PREFIXES)
    ]
    if blocked:
        joined = ", ".join(blocked)
        raise ValueError(
            f"model profile {name!r} must not set attacker/judge env: {joined}. "
            "Keep attacker and judge fixed in common.env."
        )


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _resolve_profile(
    name: str,
    profiles: Mapping[str, Any],
    stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if name in stack:
        chain = " -> ".join((*stack, name))
        raise ValueError(f"cyclic model profile inheritance: {chain}")
    profile = profiles.get(name)
    if not isinstance(profile, Mapping):
        raise KeyError(f"unknown model profile {name!r}")
    parent_name = profile.get("extends")
    if not parent_name:
        return dict(profile)
    parent = _resolve_profile(str(parent_name), profiles, (*stack, name))
    merged = dict(parent)
    _deep_merge(merged, {key: value for key, value in profile.items() if key != "extends"})
    return merged


def _expand_env_refs(value: Any, env: Mapping[str, str] | None = None) -> Any:
    lookup = env or os.environ
    if isinstance(value, str):
        return _ENV_REF.sub(lambda match: lookup.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_refs(item, lookup) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_env_refs(item, lookup) for key, item in value.items()}
    return value


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
