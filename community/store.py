"""Scenario-backed community world store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dynacomsec.core.scenario import communities, community_relations, info_items


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


class CommunityStore:
    """Read-only policy view plus workspace path helpers for one scenario."""

    def __init__(self, scenario: dict, workspace_root: str | Path | None = None) -> None:
        self.scenario = scenario
        self.workspace_root = Path(
            workspace_root
            or os.environ.get("DYNACOMSEC_WORKSPACE_ROOT")
            or (Path(os.environ.get("AGENT_HOME_DIR", "/workspace/")) / "workspace")
        )

    def list_communities(self) -> list[dict]:
        return [
            {
                "community_id": community.get("community_id"),
                "name": community.get("name"),
                "privacy_norms": community.get("privacy_norms", {}),
                "members": community.get("members", []),
            }
            for community in communities(self.scenario)
        ]

    def community(self, community_id: str) -> dict | None:
        return next(
            (
                community
                for community in communities(self.scenario)
                if community.get("community_id") == community_id
            ),
            None,
        )

    def list_members(self, community_id: str) -> list[dict]:
        community = self.community(community_id)
        if not community:
            return []
        return list(community.get("members", []))

    def is_member(self, user_id: str, community_id: str) -> bool:
        return any(
            member.get("user_id") == user_id and member.get("status", "active") == "active"
            for member in self.list_members(community_id)
        )

    def workspace_path(self, community_id: str, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute():
            raise ValueError("relative_path must be relative to the community workspace")
        candidate = (self.workspace_root / community_id / raw).resolve()
        community_root = (self.workspace_root / community_id).resolve()
        if community_root not in candidate.parents and candidate != community_root:
            raise ValueError("path escapes the requested community workspace")
        return candidate

    def check_authorization(
        self,
        *,
        user_id: str,
        source_community: str,
        destination_community: str | None = None,
        action: str = "read",
        info_id: str | None = None,
    ) -> AuthorizationDecision:
        destination_community = destination_community or source_community
        if not self.is_member(user_id, source_community):
            return AuthorizationDecision(False, f"{user_id} is not an active member of {source_community}")
        if destination_community != source_community and not self.is_member(user_id, destination_community):
            return AuthorizationDecision(False, f"{user_id} is not an active member of {destination_community}")

        if info_id:
            info = next(
                (item for item in info_items(self.scenario) if item.get("info_id") == info_id),
                None,
            )
            if not info:
                return AuthorizationDecision(False, f"unknown info_id: {info_id}")
            if source_community not in set(info.get("allowed_communities", [])):
                return AuthorizationDecision(False, f"{info_id} is not allowed in {source_community}")
            if destination_community in set(info.get("forbidden_communities", [])):
                return AuthorizationDecision(False, f"{info_id} is forbidden in {destination_community}")
            if user_id in set(info.get("forbidden_recipients", [])):
                return AuthorizationDecision(False, f"{user_id} is a forbidden recipient for {info_id}")
            if user_id not in set(info.get("allowed_recipients", [])) and action in {"read", "share"}:
                return AuthorizationDecision(False, f"{user_id} is not an allowed recipient for {info_id}")

        if destination_community != source_community:
            relation = next(
                (
                    item
                    for item in community_relations(self.scenario)
                    if item.get("from") == source_community and item.get("to") == destination_community
                ),
                {},
            )
            if relation.get("authorization_boundary") == "strict_isolation":
                return AuthorizationDecision(
                    False,
                    f"{source_community} -> {destination_community} has strict_isolation boundary",
                )
        return AuthorizationDecision(True, "authorized by active membership and scenario policy")
