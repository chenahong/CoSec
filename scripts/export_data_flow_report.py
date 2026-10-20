"""Export an end-to-end DynaComSec data-flow report for a completed trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ACTION_EVENT_SCHEMA = "dynacomsec.action_event.v1"


def export_report(trial_dir: Path) -> dict[str, Any]:
    trajectory = _load_json(trial_dir / "agent" / "trajectory.json", {})
    verifier = _load_json(trial_dir / "verifier" / "result.json", {})
    scorecard = _load_json(trial_dir / "verifier" / "scorecard.json", {})
    config = _load_json(trial_dir / "config.json", {})
    setup = _load_json(trial_dir / "artifacts" / "logs" / "artifacts" / "setup.json", [])
    before_attack = _load_json(trial_dir / "artifacts" / "logs" / "artifacts" / "before_attack.json", [])
    world_state_before = _load_json(
        trial_dir / "artifacts" / "logs" / "artifacts" / "world_state_before.json",
        {},
    )
    cleanup = _load_json(trial_dir / "verifier" / "cleanup.json", [])
    tool_calls = _load_tool_calls(trial_dir / "artifacts" / "logs" / "artifacts")

    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list):
        steps = verifier.get("steps") if isinstance(verifier, dict) else []
    steps = steps if isinstance(steps, list) else []

    conversation_flow = [_step_flow(step, trial_dir) for step in steps if isinstance(step, dict)]
    action_events = [
        event
        for step in conversation_flow
        for event in step.get("action_events", [])
        if isinstance(event, dict)
    ]

    report = {
        "schema": "dynacomsec.data_flow_report.v1",
        "trial_dir": str(trial_dir),
        "scenario_id": _first_nonempty(
            trajectory.get("scenario_id") if isinstance(trajectory, dict) else None,
            verifier.get("scenario_id") if isinstance(verifier, dict) else None,
            scorecard.get("scenario_id") if isinstance(scorecard, dict) else None,
        ),
        "config": config,
        "scorecard": scorecard,
        "session_manager": (trajectory.get("interaction_metadata") or {}).get("session_manager")
        if isinstance(trajectory, dict)
        else None,
        "environment_flow": {
            "setup": setup,
            "before_attack_snapshot": before_attack,
            "world_state_before": world_state_before,
            "cleanup": cleanup,
        },
        "conversation_flow": conversation_flow,
        "action_flow": _action_flow(action_events),
        "tool_flow": {
            "tool_calls": tool_calls,
            "note": (
                "Raw tool_calls are retained for compatibility. Use action_flow/action_events "
                "for normalized cross-agent process analysis."
            )
            if action_events
            else "No tool or action events were recorded for this trial.",
        },
        "verifier_flow": {
            "metrics": verifier.get("metrics") if isinstance(verifier, dict) else None,
            "decision_source": verifier.get("decision_source") if isinstance(verifier, dict) else None,
            "evidence": verifier.get("evidence") if isinstance(verifier, dict) else None,
            "snapshots": verifier.get("snapshots") if isinstance(verifier, dict) else None,
            "judge_result": verifier.get("judge_result") if isinstance(verifier, dict) else None,
        },
        "sanity_checks": _sanity_checks(steps),
    }
    return report


def write_reports(trial_dir: Path) -> tuple[Path, Path]:
    report = export_report(trial_dir)
    json_path = trial_dir / "data_flow_report.json"
    md_path = trial_dir / "data_flow_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _step_flow(step: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    index = step.get("step") or step.get("round")
    native_log_path = trial_dir / "agent" / f"step-{index}.txt"
    native_log = native_log_path.read_text(encoding="utf-8") if native_log_path.is_file() else None
    embedded_tool_calls = step.get("tool_calls") or step.get("tools") or step.get("actions") or []
    raw_tool_calls = embedded_tool_calls or _native_tool_calls(native_log)
    action_events = _action_events_for_step(
        step,
        native_log=native_log,
        raw_tool_calls=raw_tool_calls,
    )
    return {
        "step": index,
        "phase": step.get("phase"),
        "community": step.get("community"),
        "session_id": step.get("session_id"),
        "backend_session_id": step.get("backend_session_id"),
        "speaker": step.get("speaker"),
        "speaker_role": step.get("speaker_role"),
        "intent": step.get("intent"),
        "speaker_output_message_to_agent": step.get("delivered_message") or step.get("message"),
        "speaker_model_input": step.get("speaker_model_input") or step.get("attacker_model_input"),
        "victim_agent_input": step.get("victim_model_input"),
        "victim_agent_output": step.get("response"),
        "runtime": step.get("runtime"),
        "model": step.get("model"),
        "expected_agent_action": step.get("expected_agent_action"),
        "local_permission_check": step.get("local_permission_check"),
        "global_violation": step.get("global_violation"),
        "native_runtime_log_path": str(native_log_path) if native_log_path.is_file() else None,
        "native_runtime_log": native_log,
        "tool_calls": raw_tool_calls,
        "observability": _observability(action_events),
        "action_events": action_events,
    }


def _sanity_checks(steps: list[dict[str, Any]]) -> dict[str, Any]:
    victim_prompts = "\n".join(
        str(((step.get("victim_model_input") or {}).get("rendered_prompt") or ""))
        for step in steps
        if isinstance(step, dict)
    )
    bad_terms = [
        "DynaComSec",
        "benchmark",
        "attacker",
        "contains_attacker",
        "attacker_branch",
        "vestibular migraines",
        "topiramate adjustment",
    ]
    return {
        "victim_prompt_bad_terms": [
            term for term in bad_terms if term.casefold() in victim_prompts.casefold()
        ],
        "steps_with_victim_input": sum(
            1 for step in steps if isinstance((step.get("victim_model_input") or {}), dict)
        ),
        "steps_with_speaker_input": sum(
            1
            for step in steps
            if isinstance((step.get("speaker_model_input") or step.get("attacker_model_input") or {}), dict)
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DynaComSec Data Flow Report",
        "",
        f"- Trial: `{report.get('trial_dir')}`",
        f"- Scenario: `{report.get('scenario_id')}`",
    ]
    scorecard = report.get("scorecard") or {}
    if isinstance(scorecard, dict):
        lines.extend(
            [
                f"- Status: `{scorecard.get('status')}`",
                f"- Reward: `{scorecard.get('binary_reward')}`",
                f"- Decision source: `{scorecard.get('decision_source')}`",
                "",
            ]
        )
    session_manager = report.get("session_manager")
    lines.extend(["## Sessions", "", "```json", json.dumps(session_manager, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Sanity Checks", "", "```json", json.dumps(report.get("sanity_checks"), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Environment Flow", "", "```json", json.dumps(report.get("environment_flow"), ensure_ascii=False, indent=2), "```", ""])
    for step in report.get("conversation_flow", []):
        lines.extend(
            [
                f"## Step {step.get('step')}",
                "",
                f"- Phase: `{step.get('phase')}`",
                f"- Community: `{step.get('community')}`",
                f"- Session: `{step.get('session_id')}`",
                f"- Backend session: `{step.get('backend_session_id')}`",
                "",
                "### Speaker Model Input",
                "",
                "```json",
                json.dumps(step.get("speaker_model_input"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Speaker Output To Agent",
                "",
                "```text",
                str(step.get("speaker_output_message_to_agent") or ""),
                "```",
                "",
                "### Victim Agent Input",
                "",
                "```json",
                json.dumps(step.get("victim_agent_input"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Victim Agent Output",
                "",
                "```text",
                str(step.get("victim_agent_output") or ""),
                "```",
                "",
                "### Tool Calls",
                "",
                "```json",
                json.dumps(step.get("tool_calls"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Action Events",
                "",
                "```json",
                json.dumps(step.get("action_events"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Native Runtime Log",
                "",
                f"- Path: `{step.get('native_runtime_log_path')}`",
                "",
            ]
        )
    lines.extend(["## Action Flow", "", "```json", json.dumps(report.get("action_flow"), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Tool Flow", "", "```json", json.dumps(report.get("tool_flow"), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Verifier Flow", "", "```json", json.dumps(report.get("verifier_flow"), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines) + "\n"


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_tool_calls(artifact_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl = artifact_dir / "tool_calls.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    json_file = artifact_dir / "tool_calls.json"
    if json_file.is_file():
        value = _load_json(json_file, [])
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.append(value)
    return records


def _action_flow(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "dynacomsec.action_flow.v1",
        "summary": {
            "total_events": len(events),
            "by_runtime": _counts(event.get("agent_runtime") for event in events),
            "by_event_type": _counts(event.get("event_type") for event in events),
            "by_observability": _counts(event.get("observability") for event in events),
        },
        "events": events,
    }


def _action_events_for_step(
    step: dict[str, Any],
    *,
    native_log: str | None,
    raw_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runtime = str(step.get("runtime") or "unknown")
    events: list[dict[str, Any]] = []
    response = str(step.get("response") or "")
    if response:
        events.append(
            _action_event(
                step=step,
                event_index=len(events),
                event_type="agent_response",
                source="trajectory",
                source_tool="trajectory.response",
                status="completed",
                text_preview=_preview(response),
                observability="response",
            )
        )

    native_events = _native_action_events(step, native_log)
    if native_events:
        for event in native_events:
            event["event_index"] = len(events)
            events.append(event)
        return events

    embedded_events = _embedded_action_events(step, raw_tool_calls)
    if embedded_events:
        for event in embedded_events:
            event["event_index"] = len(events)
            events.append(event)
        return events

    if native_log:
        runtime_meta = _runtime_metadata_event(step, native_log)
        if runtime_meta:
            runtime_meta["event_index"] = len(events)
            events.append(runtime_meta)
        elif not response:
            events.append(
                _action_event(
                    step=step,
                    event_index=len(events),
                    event_type="agent_response",
                    source="native_runtime_log",
                    source_tool=f"native.{runtime}.stdout",
                    status="unknown",
                    text_preview=_preview(native_log),
                    observability="response_only",
                )
            )
    return events


def _native_action_events(step: dict[str, Any], native_log: str | None) -> list[dict[str, Any]]:
    if not native_log:
        return []
    events: list[dict[str, Any]] = []
    for record in _json_records_from_text(native_log):
        item = record.get("item") if isinstance(record, dict) else None
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            output = str(item.get("aggregated_output") or "")
            events.append(
                _action_event(
                    step=step,
                    event_index=len(events),
                    event_type="shell_command",
                    source="native_runtime_log",
                    source_tool="native.command_execution",
                    status=_optional_string(item.get("status")),
                    command=command,
                    exit_code=item.get("exit_code"),
                    output_preview=_preview(output),
                    observability="native_structured",
                )
            )
        elif item_type == "file_change":
            changes = item.get("changes") if isinstance(item.get("changes"), list) else []
            events.append(
                _action_event(
                    step=step,
                    event_index=len(events),
                    event_type="file_change",
                    source="native_runtime_log",
                    source_tool="native.file_change",
                    status=_optional_string(item.get("status")),
                    changes=changes,
                    raw_paths=_raw_paths_from_changes(changes),
                    observability="native_structured",
                )
            )
    return events


def _embedded_action_events(step: dict[str, Any], raw_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for raw in raw_tool_calls:
        if not isinstance(raw, dict):
            continue
        command = _optional_string(raw.get("command"))
        output = _optional_string(raw.get("output") or raw.get("result") or raw.get("stdout"))
        changes = raw.get("changes") if isinstance(raw.get("changes"), list) else None
        event_type = "tool_call"
        tool = str(raw.get("tool") or raw.get("name") or "embedded.tool_call")
        if "command" in raw or "shell" in tool:
            event_type = "shell_command"
        elif changes or "file" in tool:
            event_type = "file_change"
        events.append(
            _action_event(
                step=step,
                event_index=len(events),
                event_type=event_type,
                source=str(raw.get("source") or "trajectory"),
                source_tool=tool,
                status=_optional_string(raw.get("status")),
                command=command,
                exit_code=raw.get("exit_code"),
                output_preview=_preview(output or ""),
                changes=changes,
                raw_paths=_raw_paths_from_changes(changes or []),
                observability="embedded_tool_call",
            )
        )
    return events


def _runtime_metadata_event(step: dict[str, Any], native_log: str) -> dict[str, Any] | None:
    for record in _json_records_from_text(native_log):
        if not isinstance(record, dict):
            continue
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        payloads = record.get("payloads") if isinstance(record.get("payloads"), list) else None
        agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
        system_report = meta.get("systemPromptReport") if isinstance(meta.get("systemPromptReport"), dict) else {}
        if payloads is None and not agent_meta and not system_report:
            continue
        text = "\n".join(
            str(payload.get("text") or "")
            for payload in payloads or []
            if isinstance(payload, dict) and payload.get("text")
        )
        return _action_event(
            step=step,
            event_index=0,
            event_type="runtime_metadata",
            source="native_runtime_log",
            source_tool=f"native.{step.get('runtime') or 'unknown'}.metadata",
            status="completed" if text else "unknown",
            text_preview=_preview(text or native_log),
            observability="response_with_runtime_meta",
            metadata={
                "session_id": agent_meta.get("sessionId"),
                "agent_harness_id": agent_meta.get("agentHarnessId"),
                "provider": agent_meta.get("provider"),
                "model": agent_meta.get("model"),
                "workspace_dir": system_report.get("workspaceDir"),
            },
        )
    return None


def _json_records_from_text(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    records = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            records.append(value)
        cursor = start + end
    return records


def _action_event(
    *,
    step: dict[str, Any],
    event_index: int,
    event_type: str,
    source: str,
    source_tool: str,
    status: str | None,
    raw_paths: list[str] | None = None,
    observability: str,
    command: str | None = None,
    exit_code: Any = None,
    output_preview: str | None = None,
    text_preview: str | None = None,
    changes: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    community = _optional_string(step.get("community"))
    event: dict[str, Any] = {
        "schema": ACTION_EVENT_SCHEMA,
        "step": step.get("step") or step.get("round"),
        "event_index": event_index,
        "agent_runtime": step.get("runtime"),
        "model": step.get("model"),
        "community": community,
        "session_id": step.get("session_id"),
        "source": source,
        "source_tool": source_tool,
        "event_type": event_type,
        "status": status or "unknown",
        "observability": observability,
    }
    if raw_paths:
        event["raw_paths"] = raw_paths
    if command:
        event["command"] = command
    if exit_code is not None:
        event["exit_code"] = exit_code
    if output_preview:
        event["output_preview"] = output_preview
    if text_preview:
        event["text_preview"] = text_preview
    if changes is not None:
        event["changes"] = changes
    if metadata:
        event["metadata"] = {key: value for key, value in metadata.items() if value is not None}
    return event


def _observability(events: list[dict[str, Any]]) -> str:
    levels = [str(event.get("observability") or "") for event in events]
    if "native_structured" in levels:
        return "native_structured"
    if "embedded_tool_call" in levels:
        return "embedded_tool_call"
    if "response_with_runtime_meta" in levels:
        return "response_with_runtime_meta"
    if "response" in levels:
        return "response_only"
    return "none"


def _raw_paths_from_changes(changes: list[Any]) -> list[str]:
    paths = []
    seen = set()
    for change in changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        if not path:
            continue
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        paths.append(text)
    return paths


def _preview(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _native_tool_calls(native_log: str | None) -> list[dict[str, Any]]:
    if not native_log:
        return []
    records: list[dict[str, Any]] = []
    for event in _json_records_from_text(native_log):
        item = event.get("item") if isinstance(event, dict) else None
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "command_execution":
            records.append(
                {
                    "source": "native_runtime_log",
                    "tool": "native.command_execution",
                    "status": item.get("status"),
                    "command": item.get("command"),
                    "exit_code": item.get("exit_code"),
                    "output": item.get("aggregated_output"),
                }
            )
        elif item_type == "file_change":
            records.append(
                {
                    "source": "native_runtime_log",
                    "tool": "native.file_change",
                    "status": item.get("status"),
                    "changes": item.get("changes"),
                }
            )
    return records


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete DynaComSec trial data-flow report.")
    parser.add_argument("trial_dir", type=Path)
    args = parser.parse_args()
    json_path, md_path = write_reports(args.trial_dir)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
