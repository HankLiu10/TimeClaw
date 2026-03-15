"""Factory for converting ClawTool instances into FastMCP tool registrations.

Usage in each MCP server script:

    from mcp.server.fastmcp import FastMCP
    from timeclaw.mcp.server_factory import register_category

    mcp = FastMCP("timeclaw-prediction")
    register_category(mcp, "prediction")

    if __name__ == "__main__":
        mcp.run(transport="stdio")
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _serialize(data: Any) -> str:
    """Serialize tool output to JSON text for MCP content."""
    return json.dumps(data, cls=_NumpyEncoder, ensure_ascii=False, default=str)


def register_tool(mcp: Any, tool: ClawTool) -> None:
    """Register a single ClawTool as an MCP tool on the given FastMCP server.

    The generated wrapper unpacks MCP arguments, routes ``ts_data`` / ``text``
    / ``images`` correctly, and serializes the ``ToolResult`` back to MCP
    text content.
    """
    schema = tool.parameters_schema or {"type": "object", "properties": {}}
    props = dict(schema.get("properties", {}))
    schema_required = set(schema.get("required", []))

    # Inject the implicit positional args that ClawTool.execute() expects
    # but that are NOT declared in parameters_schema.
    io = tool.io_type.value if hasattr(tool.io_type, "value") else str(tool.io_type)
    needs_ts = "X" in io or not props
    needs_text = "T" in io
    needs_images = "I" in io

    if needs_ts and "ts_data" not in props:
        props["ts_data"] = {"type": "array", "description": "Input time-series data (list of floats)"}
    if needs_text and "text" not in props:
        props["text"] = {"type": "string", "description": "Input text data"}
    if needs_images and "images" not in props:
        props["images"] = {"type": "array", "description": "Input image paths or base64 strings"}

    _TYPE_MAP = {
        "number": float, "integer": int, "string": str,
        "boolean": bool, "array": list, "object": dict,
    }

    # ts_data is required only when there are no other required params
    # (e.g. pure profiling/detection tools). For validation tools that
    # have their own required params (prediction, historical, etc.),
    # ts_data is optional — the tool uses it as a fallback.
    required = set(schema_required)
    if needs_ts and "ts_data" not in required:
        if not schema_required:
            required.add("ts_data")

    def _resolve_type(pspec: dict) -> type:
        """Resolve JSON Schema type to Python type, handling oneOf/anyOf."""
        if "type" in pspec:
            return _TYPE_MAP.get(pspec["type"], Any)
        if "oneOf" in pspec or "anyOf" in pspec:
            variants = pspec.get("oneOf") or pspec.get("anyOf", [])
            types = {_TYPE_MAP.get(v.get("type"), Any) for v in variants if "type" in v}
            if types == {int, str}:
                return Any  # accept both
            if len(types) == 1:
                return types.pop()
            return Any
        return Any

    sig_params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {"return": str}
    for pname, pspec in props.items():
        ptype = _resolve_type(pspec)
        annotations[pname] = ptype
        default = inspect.Parameter.empty if pname in required else pspec.get("default")
        sig_params.append(inspect.Parameter(
            pname,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=ptype,
        ))

    # Capture tool ref in closure
    _tool = tool

    async def _handler(**kwargs: Any) -> str:
        ts_data = kwargs.pop("ts_data", None)
        if ts_data is not None and isinstance(ts_data, list):
            ts_data = np.asarray(ts_data, dtype=float)
        text = kwargs.pop("text", None)
        images = kwargs.pop("images", None)
        # Remove None-valued optional params so tool defaults take effect
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = await _tool.execute(ts_data, text=text, images=images, **kwargs)
        if result.success:
            return _serialize(result.data)
        raise ValueError(result.error or "Tool execution failed")

    _handler.__name__ = tool.name
    _handler.__qualname__ = tool.name
    _handler.__doc__ = tool.description or f"ClawHub tool: {tool.name}"
    _handler.__annotations__ = annotations
    _handler.__signature__ = inspect.Signature(parameters=sig_params)

    mcp.tool(name=tool.name, description=tool.description)(_handler)


def register_category(mcp: Any, category: str) -> int:
    """Discover all ClawTool instances in a clawhub sub-package and register
    them on the given FastMCP server.

    Returns the number of tools registered.
    """
    import importlib
    import pkgutil

    count = 0
    try:
        pkg = importlib.import_module(f"timeclaw.clawhub.{category}")
    except ImportError as e:
        logger.warning(f"[MCP ServerFactory] Cannot import clawhub.{category}: {e}")
        return 0

    for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"timeclaw.clawhub.{category}.{modname}")
            for attr_name in dir(mod):
                if attr_name == "tool" or attr_name.startswith("tool_"):
                    tool_obj = getattr(mod, attr_name, None)
                    if tool_obj is not None and isinstance(tool_obj, ClawTool):
                        if tool_obj.is_stub:
                            continue
                        register_tool(mcp, tool_obj)
                        count += 1
        except Exception as e:
            logger.warning(
                f"[MCP ServerFactory] Failed to load {category}.{modname}: {e}"
            )

    logger.info(f"[MCP ServerFactory] Registered {count} tools for '{category}'")
    return count


def create_category_server(name: str, category: str) -> Any:
    """Create a ready-to-run FastMCP server for a given tool category.

    Returns the FastMCP instance (call ``mcp.run()`` to start).
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(name)
    register_category(mcp, category)
    return mcp
