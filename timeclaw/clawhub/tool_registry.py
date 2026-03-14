"""Automatic tool discovery, registration, and retrieval for ClawHub."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory

_TOOL_INSTANCES: dict[str, ClawTool] = {}


def register_tool_instance(tool: ClawTool) -> None:
    """Register an instantiated tool."""
    _TOOL_INSTANCES[tool.name] = tool


def get_tool_instance(name: str) -> ClawTool:
    """Retrieve a tool instance by name."""
    tool = _TOOL_INSTANCES.get(name)
    if tool is None:
        raise KeyError(f"Tool '{name}' not found. Available: {list(_TOOL_INSTANCES.keys())}")
    return tool


def list_tool_instances(category: ToolCategory | str | None = None) -> list[ClawTool]:
    """List all registered tool instances, optionally filtered by category."""
    if category is None:
        return list(_TOOL_INSTANCES.values())
    cat = category if isinstance(category, str) else category.value
    return [t for t in _TOOL_INSTANCES.values() if t.category.value == cat]


def get_all_function_schemas() -> list[dict[str, Any]]:
    """Get LLM function-calling schemas for all registered tools."""
    return [t.to_function_schema() for t in _TOOL_INSTANCES.values()]


def get_function_schemas_by_category(category: ToolCategory | str) -> list[dict[str, Any]]:
    """Get function schemas filtered by category."""
    return [t.to_function_schema() for t in list_tool_instances(category)]


def discover_and_register() -> int:
    """Auto-discover all tool modules under clawhub sub-packages and register them."""
    import timeclaw.clawhub as clawhub_pkg

    count = 0
    for sub_pkg in [
        "processing",
        "profiling",
        "detection",
        "decomposition",
        "relation",
        "prediction",
        "validation",
        "text",
        "image",
    ]:
        try:
            pkg = importlib.import_module(f"timeclaw.clawhub.{sub_pkg}")
            for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
                if modname.startswith("_"):
                    continue
                try:
                    mod = importlib.import_module(f"timeclaw.clawhub.{sub_pkg}.{modname}")
                    # Each module should define a `tool` attribute (instance of ClawTool)
                    tool_obj = getattr(mod, "tool", None)
                    if tool_obj is not None and isinstance(tool_obj, ClawTool):
                        register_tool_instance(tool_obj)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load tool module {sub_pkg}.{modname}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load sub-package {sub_pkg}: {e}")

    logger.info(f"ClawHub: discovered and registered {count} tools")
    return count


async def discover_and_register_mcp(server_configs: list[dict] | None = None) -> int:
    """Discover and register tools from external MCP servers.

    Args:
        server_configs: List of dicts, each with 'command' (list[str]) and optional 'env' (dict).

    Returns:
        Total number of MCP tools registered.
    """
    if not server_configs:
        return 0

    from timeclaw.mcp.client import register_mcp_tools

    total = 0
    for config in server_configs:
        count = await register_mcp_tools(config)
        total += count
    logger.info(f"ClawHub: registered {total} MCP tools from {len(server_configs)} servers")
    return total


def clear_registry() -> None:
    """Clear all registered tools (for testing)."""
    _TOOL_INSTANCES.clear()
