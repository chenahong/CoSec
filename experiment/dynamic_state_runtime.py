"""Runtime updates for agent-visible dynamic state files."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from dynacomsec.core.scenario import expand_placeholders, resolved_path_vars
from dynacomsec.experiment.state_context import (
    build_agent_visible_dynamic_state,
    sanitize_agent_visible_text,
)


def initial_runtime_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable scenario view with future dynamic events hidden."""

    runtime = deepcopy(dict(scenario))
    views = runtime.get("agent_views")
    if isinstance(views, dict):
        for view in views.values():
            if isinstance(view, dict):
                view["current_events"] = []
    return runtime


def find_dynamic_event(scenario: Mapping[str, Any], event_ref: str | None) -> dict[str, Any] | None:
    if not event_ref:
        return None
    for event in scenario.get("world_state", {}).get("dynamic_events", []):
        if isinstance(event, Mapping) and str(event.get("event_id") or "") == event_ref:
            return dict(event)
    return None


def event_notice(event: Mapping[str, Any], fallback: str = "") -> str:
    return sanitize_agent_visible_text(
        str(event.get("agent_visible_notice") or event.get("notice") or fallback).strip()
    )


def synthetic_dynamic_event(
    *,
    event_id: str,
    community_id: str,
    message: str,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "type": "system_event",
        "community": community_id,
        "notice": message,
    }


def apply_dynamic_event(runtime_scenario: dict[str, Any], event: Mapping[str, Any]) -> list[str]:
    """Apply a trusted event to runtime agent views and return affected communities."""

    views = runtime_scenario.get("agent_views")
    if not isinstance(views, dict):
        return []
    affected = _affected_communities(event, views)
    notice = event_notice(event)
    event_record = _agent_visible_event_record(event, notice)

    for community_id in affected:
        view = views.get(community_id)
        if not isinstance(view, dict):
            continue
        current_events = view.setdefault("current_events", [])
        if isinstance(current_events, list) and not _has_event(current_events, event_record):
            current_events.append(deepcopy(event_record))
        _apply_session_transition(view, event, community_id)
        _apply_member_or_role_change(view, event)
        _apply_relationship_change(view, event)
    return affected


def build_dynamic_state_snapshots(
    runtime_scenario: Mapping[str, Any],
    *,
    applied_event_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build payloads and paths for all configured dynamic_state.json targets."""

    snapshots = []
    for community_id, path in _dynamic_state_targets(runtime_scenario):
        payload = build_agent_visible_dynamic_state(
            runtime_scenario,
            community_id=community_id,
            applied_event_ids=applied_event_ids or [],
        )
        snapshots.append(
            {
                "community_id": community_id,
                "path": expand_placeholders(path, resolved_path_vars()),
                "payload": payload,
            }
        )
    return snapshots


def _dynamic_state_targets(scenario: Mapping[str, Any]) -> list[tuple[str, str]]:
    setup = scenario.get("environment_setup", {})
    if not isinstance(setup, Mapping):
        return []
    targets = []
    seen: set[str] = set()
    for item in setup.get("agent_visible_dynamic_state", []):
        if not isinstance(item, Mapping):
            continue
        community_id = str(item.get("community_id", ""))
        path = str(item.get("path", ""))
        if community_id and path and path not in seen:
            targets.append((community_id, path))
            seen.add(path)
    return targets


def _agent_visible_event_record(event: Mapping[str, Any], notice: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_id": str(event.get("event_id") or ""),
        "type": sanitize_agent_visible_text(str(event.get("type") or "system_event")),
        "notice": notice,
    }
    if event.get("security_effect"):
        record["security_effect"] = deepcopy(event["security_effect"])
    for key in ("effective_at_step", "community", "from", "to", "target_user_id"):
        if key in event:
            record[key] = deepcopy(event[key])
    return record


def _affected_communities(event: Mapping[str, Any], views: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("community", "from", "to"):
        value = event.get(key)
        if value:
            candidates.append(str(value))
    transition = event.get("session_transition")
    if isinstance(transition, Mapping):
        for key in _session_transition_branch_keys():
            branch = transition.get(key)
            if isinstance(branch, Mapping) and branch.get("community_id"):
                candidates.append(str(branch["community_id"]))
    visible = [community for community in dict.fromkeys(candidates) if community in views]
    return visible or list(views.keys())


def _has_event(current_events: list[Any], event_record: Mapping[str, Any]) -> bool:
    event_id = str(event_record.get("event_id") or "")
    return any(isinstance(item, Mapping) and str(item.get("event_id") or "") == event_id for item in current_events)


def _apply_session_transition(view: dict[str, Any], event: Mapping[str, Any], community_id: str) -> None:
    transition = event.get("session_transition")
    if not isinstance(transition, Mapping):
        return
    for key in _session_transition_branch_keys():
        branch = transition.get(key)
        if isinstance(branch, Mapping) and str(branch.get("community_id") or "") == community_id:
            current = branch.get("current_community")
            if isinstance(current, Mapping):
                view["current_community"] = deepcopy(current)
            current = view.get("current_community")
            if isinstance(current, dict):
                lineage = dict(current.get("session_lineage") or {})
                for field in ("session_policy", "history_access", "inherits_from", "split_from"):
                    if field in branch:
                        lineage[field] = branch[field]
                current["session_lineage"] = lineage


def _session_transition_branch_keys() -> tuple[str, ...]:
    return ("trusted_successor", "requester_branch", "attacker_branch", "merged_community")


def _apply_member_or_role_change(view: dict[str, Any], event: Mapping[str, Any]) -> None:
    community_id = event.get("community")
    target_user = str(event.get("target_user_id") or "")
    after = event.get("after")
    if not community_id or not target_user or not isinstance(after, Mapping):
        return
    for community in _view_communities(view):
        if not isinstance(community, dict) or str(community.get("community_id") or "") != str(community_id):
            continue
        members = community.get("members")
        if not isinstance(members, list):
            continue
        found = False
        for member in members:
            if not isinstance(member, dict) or str(member.get("user_id") or "") != target_user:
                continue
            found = True
            if "membership_status" in after:
                member["status"] = after["membership_status"]
            if "status" in after:
                member["status"] = after["status"]
            if "role" in after:
                member["role"] = after["role"]
        if not found and _should_add_member(after):
            members.append(
                {
                    "user_id": target_user,
                    "role": after.get("role", "requester"),
                    "status": after.get("membership_status", after.get("status", "active")),
                }
            )


def _should_add_member(after: Mapping[str, Any]) -> bool:
    status = str(after.get("membership_status", after.get("status", ""))).lower()
    return status in {"active", "added", "joined"} or "role" in after


def _apply_relationship_change(view: dict[str, Any], event: Mapping[str, Any]) -> None:
    relation_from = str(event.get("from") or "")
    relation_to = str(event.get("to") or "")
    after = event.get("after")
    if not relation_from or not relation_to or not isinstance(after, Mapping):
        return
    relations = view.get("visible_relations")
    if not isinstance(relations, list):
        return
    found = False
    for relation in relations:
        if (
            isinstance(relation, dict)
            and str(relation.get("from") or "") == relation_from
            and str(relation.get("to") or "") == relation_to
        ):
            found = True
            if "relationship" in after:
                relation["type"] = after["relationship"]
            if "sharing_default" in after:
                relation["sharing_default"] = after["sharing_default"]
            if "authorization_boundary" in after:
                relation["authorization_boundary"] = after["authorization_boundary"]
    if not found and _should_add_relation(after):
        relations.append(
            {
                "from": relation_from,
                "to": relation_to,
                "type": after.get("relationship", "limited_cooperation"),
                "sharing_default": after.get("sharing_default", "safe_abstraction_only"),
                "authorization_boundary": after.get("authorization_boundary", "current_state_required"),
            }
        )


def _should_add_relation(after: Mapping[str, Any]) -> bool:
    relationship = str(after.get("relationship", "")).lower()
    return relationship in {"started", "limited_cooperation", "active_limited_cooperation"}


def _view_communities(view: Mapping[str, Any]) -> list[Any]:
    communities = []
    if "current_community" in view:
        communities.append(view["current_community"])
    related = view.get("visible_related_communities")
    if isinstance(related, list):
        communities.extend(related)
    return communities
