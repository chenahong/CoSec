"""Shared execution helpers for DynaComSec setup and verification."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from dynacomsec.core.commands import run_commands


def collect_commands(section: dict | list | None, key: str) -> list[str]:
    if section is None:
        return []
    if isinstance(section, list):
        return [str(item) for item in section]
    if isinstance(section, dict):
        commands = []
        for value in section.values():
            if isinstance(value, dict):
                commands.extend(str(item) for item in value.get(key, []))
            elif isinstance(value, list) and key == "commands":
                commands.extend(str(item) for item in value)
        return commands
    return []


def physical_snapshot(scenario: dict, phase: str, dry_run: bool) -> list[dict]:
    verification = scenario.get("verification", {})
    checks = verification.get("physical_checks", {})
    phase_checks = checks.get(phase, {})
    commands = []
    if isinstance(phase_checks, dict):
        for command in phase_checks.values():
            if isinstance(command, str):
                commands.append(command)
            elif isinstance(command, dict):
                commands.extend(str(item) for item in command.get("commands", []))
    elif isinstance(phase_checks, list):
        commands = [str(item) for item in phase_checks]
    return run_commands(commands, dry_run=dry_run)


def benign_completion_artifacts(scenario: dict, dry_run: bool) -> list[dict]:
    """Fingerprint targets referenced by simple ``test -s`` completion checks."""

    completion = ((scenario.get("verification") or {}).get("physical_checks") or {}).get(
        "benign_completion"
    )
    paths: list[str] = []
    for command in collect_commands(completion, "commands"):
        match = re.fullmatch(r"\s*test\s+-s\s+([^\s]+)\s*", command)
        if match and match.group(1) not in paths:
            paths.append(match.group(1))

    snapshots: list[dict] = []
    for path_text in paths:
        item = {
            "path": path_text,
            "exists": False,
            "size": 0,
            "sha256": None,
            "skipped": dry_run,
        }
        if not dry_run:
            path = Path(path_text)
            try:
                if path.is_file():
                    content = path.read_bytes()
                    item.update(
                        {
                            "exists": True,
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
            except OSError as exc:
                item["error"] = str(exc)
        snapshots.append(item)
    return snapshots


def compare_benign_completion_artifacts(
    before: list[dict] | None,
    after: list[dict] | None,
) -> list[dict]:
    """Return whether each completion target changed during the trial."""

    before_by_path = {
        str(item.get("path")): item for item in (before or []) if isinstance(item, dict)
    }
    comparisons: list[dict] = []
    for item in after or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path"))
        baseline = before_by_path.get(path)
        changed = bool(
            baseline
            and item.get("exists")
            and int(item.get("size") or 0) > 0
            and baseline.get("sha256") != item.get("sha256")
        )
        comparisons.append(
            {"path": path, "before": baseline, "after": item, "changed": changed}
        )
    return comparisons
