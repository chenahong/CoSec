#!/bin/sh
set -e

TASK_ID="${DYNACOMSEC_TASK_ID:-}"
TASKS_ROOT="${DYNACOMSEC_TASKS_ROOT:-/opt/dynacomsec/tasks}"
MCP_PORT="${DYNACOMSEC_MCP_PORT:-8120}"
ENABLE_MCP="${DYNACOMSEC_ENABLE_MCP:-0}"
CLEAN_AGENT_STATE="${DYNACOMSEC_CLEAN_AGENT_STATE:-0}"

case "$(printf '%s' "$ENABLE_MCP" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) MCP_ENABLED=0 ;;
  *) MCP_ENABLED=1 ;;
esac

if [ -z "$TASK_ID" ]; then
  echo "entrypoint: DYNACOMSEC_TASK_ID is not set" >&2
  exit 64
fi

SCENARIO="${DYNACOMSEC_SCENARIO:-${TASKS_ROOT}/${TASK_ID}/scenario.json}"
if [ ! -f "$SCENARIO" ]; then
  echo "entrypoint: scenario not found at $SCENARIO" >&2
  exit 65
fi
export DYNACOMSEC_SCENARIO="$SCENARIO"

case "${DYNACOMSEC_WORKSPACE_ROOT%/}" in
  ""|/|/workspace)
    echo "entrypoint: unsafe DYNACOMSEC_WORKSPACE_ROOT=$DYNACOMSEC_WORKSPACE_ROOT" >&2
    exit 67
    ;;
esac

mkdir -p /logs/agent /logs/verifier /logs/artifacts "$DYNACOMSEC_WORKSPACE_ROOT"

case "$(printf '%s' "$CLEAN_AGENT_STATE" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) CLEAN_ENABLED=0 ;;
  *) CLEAN_ENABLED=1 ;;
esac

if [ "$CLEAN_ENABLED" -eq 1 ]; then
  rm -rf "$DYNACOMSEC_WORKSPACE_ROOT"/* "$DYNACOMSEC_WORKSPACE_ROOT"/.[!.]* "$DYNACOMSEC_WORKSPACE_ROOT"/..?*
  if command -v openclaw >/dev/null 2>&1; then
    OPENCLAW_HOME="${OPENCLAW_HOME:-/root/.openclaw}"
    rm -rf "$OPENCLAW_HOME/workspace-attestations"
    find "$OPENCLAW_HOME/agents" -type d \( -name sessions -o -name memory \) -prune -exec rm -rf {} + 2>/dev/null || true
  fi
fi

python -m dynacomsec.experiment.runtime_setup --scenario "$SCENARIO" --output-dir /logs/artifacts \
  > /logs/artifacts/setup.log 2>&1 || {
    echo "entrypoint: setup failed" >&2
    tail -n 80 /logs/artifacts/setup.log >&2 || true
    exit 66
  }

if command -v openclaw >/dev/null 2>&1; then
  OPENCLAW_HOME="${OPENCLAW_HOME:-/root/.openclaw}"
  # OpenClaw 2026.8+ migrates legacy auth-profiles.json into the agent
  # SQLite credential store.  Model profiles are materialized with the
  # portable JSON form, so perform the vendor-recommended migration before
  # the agent command starts.  Without this, every trial exits before making
  # a model request with "auth profile store ... requires legacy credential
  # migration".
  if [ -f "$OPENCLAW_HOME/agents/main/agent/auth-profiles.json" ]; then
    openclaw doctor --fix --non-interactive --yes \
      > /logs/artifacts/openclaw-doctor.log 2>&1 || {
        echo "entrypoint: openclaw auth migration failed" >&2
        tail -n 80 /logs/artifacts/openclaw-doctor.log >&2 || true
        exit 69
      }
  fi
  rm -rf "$OPENCLAW_HOME/workspace-attestations"
  if [ "$OPENCLAW_HOME/workspace" != "$DYNACOMSEC_WORKSPACE_ROOT" ]; then
    rm -rf "$OPENCLAW_HOME/workspace"
    ln -s "$DYNACOMSEC_WORKSPACE_ROOT" "$OPENCLAW_HOME/workspace"
  else
    mkdir -p "$OPENCLAW_HOME/workspace"
  fi
  {
    echo "openclaw_home=$OPENCLAW_HOME"
    echo "openclaw_workspace=$OPENCLAW_HOME/workspace"
    echo "dynacomsec_workspace=$DYNACOMSEC_WORKSPACE_ROOT"
    ls -la "$OPENCLAW_HOME/workspace" || true
  } > /logs/artifacts/openclaw-workspace.log 2>&1
fi

if [ "$MCP_ENABLED" -eq 1 ]; then
  python -m dynacomsec.community.mcp_server \
    --scenario "$SCENARIO" \
    --workspace-root "$DYNACOMSEC_WORKSPACE_ROOT" \
    --port "$MCP_PORT" \
    > /logs/artifacts/community-mcp.log 2>&1 &
  MCP_PID=$!
else
  MCP_PID=""
fi

trap '[ -n "$MCP_PID" ] && kill "$MCP_PID" 2>/dev/null || true' INT TERM

if [ "$MCP_ENABLED" -eq 1 ]; then
  MCP_READY=0
  for _ in $(seq 1 60); do
    if python - <<PY >/dev/null 2>&1
import json, urllib.request
body = json.dumps({
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": "entrypoint-probe",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "dynacomsec-entrypoint", "version": "1"},
    },
}).encode()
req = urllib.request.Request(
    "http://localhost:${MCP_PORT}/mcp",
    data=body,
    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=3) as resp:
    assert 200 <= resp.status < 300
PY
    then
      MCP_READY=1
      break
    fi
    sleep 1
  done
  if [ "$MCP_READY" -ne 1 ]; then
    echo "entrypoint: community MCP did not become ready" >&2
    tail -n 80 /logs/artifacts/community-mcp.log >&2 || true
    exit 68
  fi
fi

touch /tmp/dynacomsec-ready

exec "$@"
