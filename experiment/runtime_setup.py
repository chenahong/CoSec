"""In-container scenario setup for DynaComSec experiment trials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dynacomsec.core.execution import benign_completion_artifacts, physical_snapshot
from dynacomsec.core.scenario import (
    community_relations,
    dynamic_events,
    expand_placeholders,
    info_items,
    load_scenario,
    resolved_path_vars,
    communities,
)
from dynacomsec.experiment.state_context import build_agent_visible_dynamic_state
from dynacomsec.experiment.dynamic_state_runtime import initial_runtime_scenario


def setup_scenario(scenario_path: str, *, execute_setup: bool) -> list[dict]:
    scenario = expand_placeholders(load_scenario(scenario_path), resolved_path_vars())
    results = _run_structured_environment_setup(scenario, dry_run=not execute_setup)
    results.extend(
        _write_agent_visible_dynamic_states(
            initial_runtime_scenario(scenario),
            dry_run=not execute_setup,
            applied_event_ids=[],
        )
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a DynaComSec trial workspace.")
    parser.add_argument("--scenario", default=os.environ.get("DYNACOMSEC_SCENARIO"))
    parser.add_argument("--execute-setup", action="store_true")
    parser.add_argument("--output-dir", default="/logs/artifacts")
    args = parser.parse_args()
    if not args.scenario:
        raise SystemExit("--scenario or DYNACOMSEC_SCENARIO is required")
    execute_setup = args.execute_setup or os.environ.get("DYNACOMSEC_EXECUTE_SETUP", "true") == "true"
    scenario = expand_placeholders(load_scenario(args.scenario), resolved_path_vars())
    results = setup_scenario(args.scenario, execute_setup=execute_setup)
    before_attack = physical_snapshot(scenario, "before_attack", dry_run=not execute_setup)
    benign_completion_before = benign_completion_artifacts(
        scenario,
        dry_run=not execute_setup,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "setup.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "before_attack.json").write_text(
        json.dumps(before_attack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "benign_completion_before.json").write_text(
        json.dumps(benign_completion_before, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "world_state_before.json").write_text(
        json.dumps(_initial_world_state(scenario), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        status = "skipped" if result.get("skipped") else f"rc={result.get('returncode')}"
        print(f"[setup] {status}: {result.get('command')}")


def _initial_world_state(scenario: dict) -> dict:
    """Persist the policy-bearing state without duplicating secret contents."""

    return {
        "scenario_id": scenario.get("scenario_id"),
        "communities": communities(scenario),
        "community_relations": community_relations(scenario),
        "info_items": [
            {
                "info_id": item.get("info_id"),
                "owner": item.get("owner"),
                "source_community": item.get("source_community"),
                "allowed_recipients": item.get("allowed_recipients", []),
                "allowed_communities": item.get("allowed_communities", []),
                "forbidden_recipients": item.get("forbidden_recipients", []),
                "forbidden_communities": item.get("forbidden_communities", []),
                "status": item.get("status", "active"),
            }
            for item in info_items(scenario)
            if isinstance(item, dict)
        ],
        "events_pending": dynamic_events(scenario),
    }


def _write_agent_visible_dynamic_states(
    scenario: dict,
    *,
    dry_run: bool,
    applied_event_ids: list[str] | None = None,
) -> list[dict]:
    results = []
    for community_id, path in _dynamic_state_targets(scenario):
        command = f"write sanitized dynamic state: {path}"
        if dry_run:
            results.append(
                {
                    "command": command,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "skipped": True,
                }
            )
            continue
        try:
            payload = build_agent_visible_dynamic_state(
                scenario,
                community_id=community_id,
                applied_event_ids=applied_event_ids or [],
            )
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            results.append(
                {
                    "command": command,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "skipped": False,
                }
            )
        except Exception as exc:  # pragma: no cover - surfaced in setup logs
            results.append(
                {
                    "command": command,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": str(exc),
                    "skipped": False,
                }
            )
    return results


def _dynamic_state_targets(scenario: dict) -> list[tuple[str, str]]:
    setup = scenario.get("environment_setup", {})
    targets = []
    seen: set[str] = set()
    for item in setup.get("agent_visible_dynamic_state", []):
        if not isinstance(item, dict):
            continue
        community_id = str(item.get("community_id", ""))
        path = str(item.get("path", ""))
        if community_id and path and path not in seen:
            targets.append((community_id, path))
            seen.add(path)
    return targets


def _run_structured_environment_setup(scenario: dict, *, dry_run: bool) -> list[dict]:
    setup = scenario.get("environment_setup", {})
    results: list[dict] = []
    for path in setup.get("directories", []):
        results.append(_mkdir_result(str(path), dry_run=dry_run))
    for item in setup.get("files", []):
        if isinstance(item, dict):
            results.append(
                _write_file_result(
                    str(item.get("path", "")),
                    str(item.get("content", "")),
                    dry_run=dry_run,
                )
            )
    for path in setup.get("empty_files", []):
        results.append(_write_file_result(str(path), "", dry_run=dry_run))
    return results


def _mkdir_result(path: str, *, dry_run: bool) -> dict:
    command = f"mkdir: {path}"
    if dry_run:
        return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "skipped": True}
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "skipped": False}
    except Exception as exc:  # pragma: no cover - surfaced in setup logs
        return {"command": command, "returncode": 1, "stdout": "", "stderr": str(exc), "skipped": False}


def _write_file_result(path: str, content: str, *, dry_run: bool) -> dict:
    command = f"write file: {path}"
    if dry_run:
        return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "skipped": True}
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "skipped": False}
    except Exception as exc:  # pragma: no cover - surfaced in setup logs
        return {"command": command, "returncode": 1, "stdout": "", "stderr": str(exc), "skipped": False}


if __name__ == "__main__":
    main()
