"""Optional MCP server for DynaComSec community tools."""

from __future__ import annotations

import argparse

from dynacomsec.community.tools import build_tools
from dynacomsec.core.scenario import expand_placeholders, load_scenario, resolved_path_vars


def build_community_mcp(
    *,
    scenario: dict,
    workspace_root: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8120,
):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The mcp package is required to run the DynaComSec community MCP server."
        ) from exc

    mcp = FastMCP("dynacomsec-community", host=host, port=port)
    tools = build_tools(scenario, workspace_root=workspace_root)

    mcp.tool(name="community.list_communities")(tools.list_communities)
    mcp.tool(name="community.list_members")(tools.list_members)
    mcp.tool(name="community.check_authorization")(tools.check_authorization)
    mcp.tool(name="community.read_file")(tools.read_file)
    mcp.tool(name="community.write_file")(tools.write_file)
    mcp.tool(name="community.workspace_tree")(tools.workspace_tree)
    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DynaComSec community MCP server.")
    parser.add_argument("--scenario", required=True, help="Path to a DynaComSec scenario JSON.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()

    scenario = expand_placeholders(load_scenario(args.scenario), resolved_path_vars())
    mcp = build_community_mcp(
        scenario=scenario,
        workspace_root=args.workspace_root,
        host=args.host,
        port=args.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
