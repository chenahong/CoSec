"""Community tool functions that can be exposed directly or over MCP."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dynacomsec.community.store import CommunityStore


class CommunityTools:
    """Tool facade over a DynaComSec scenario-backed community store."""

    def __init__(self, store: CommunityStore, audit_path: str | Path | None = None) -> None:
        self.store = store
        configured = audit_path or os.environ.get("DYNACOMSEC_TOOL_LOG")
        if configured is None and Path("/logs/artifacts").is_dir():
            configured = "/logs/artifacts/tool_calls.jsonl"
        self.audit_path = Path(configured) if configured else None
        self._audit_lock = threading.Lock()

    def list_communities(self) -> list[dict]:
        result = self.store.list_communities()
        self._audit("community.list_communities", {}, result)
        return result

    def list_members(self, community_id: str) -> list[dict]:
        result = self.store.list_members(community_id)
        self._audit("community.list_members", {"community_id": community_id}, result)
        return result

    def check_authorization(
        self,
        user_id: str,
        source_community: str,
        destination_community: str | None = None,
        action: str = "read",
        info_id: str | None = None,
    ) -> dict:
        decision = self.store.check_authorization(
            user_id=user_id,
            source_community=source_community,
            destination_community=destination_community,
            action=action,
            info_id=info_id,
        )
        result = {"allowed": decision.allowed, "reason": decision.reason}
        self._audit(
            "community.check_authorization",
            {
                "user_id": user_id,
                "source_community": source_community,
                "destination_community": destination_community,
                "action": action,
                "info_id": info_id,
            },
            result,
        )
        return result

    def read_file(self, community_id: str, relative_path: str) -> dict:
        path = self.store.workspace_path(community_id, relative_path)
        if not path.exists():
            result = {"ok": False, "error": f"file not found: {relative_path}"}
        elif not path.is_file():
            result = {"ok": False, "error": f"path is not a file: {relative_path}"}
        else:
            result = {"ok": True, "path": str(path), "content": path.read_text(encoding="utf-8")}
        self._audit(
            "community.read_file",
            {"community_id": community_id, "relative_path": relative_path},
            result,
        )
        return result

    def write_file(self, community_id: str, relative_path: str, content: str, append: bool = False) -> dict:
        path = self.store.workspace_path(community_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(content)
        result = {
            "ok": True,
            "path": str(path),
            "bytes": len(content.encode("utf-8")),
            "append": append,
        }
        self._audit(
            "community.write_file",
            {
                "community_id": community_id,
                "relative_path": relative_path,
                "content": content,
                "append": append,
            },
            result,
        )
        return result

    def workspace_tree(self, community_id: str) -> list[str]:
        root = self.store.workspace_path(community_id, ".")
        if not root.exists():
            return []
        paths = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                paths.append(str(path.relative_to(root)))
        self._audit("community.workspace_tree", {"community_id": community_id}, paths)
        return paths

    def _audit(self, tool: str, arguments: dict[str, Any], result: Any) -> None:
        if self.audit_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "ok": not (isinstance(result, dict) and result.get("ok") is False),
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            # Auditing must not make the benchmark tool itself unavailable.
            return


def build_tools(
    scenario: dict,
    workspace_root: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> CommunityTools:
    return CommunityTools(
        CommunityStore(scenario, workspace_root=workspace_root),
        audit_path=audit_path,
    )
