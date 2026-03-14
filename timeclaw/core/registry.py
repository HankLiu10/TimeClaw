"""Global registry for tools, models, and checkpoints with decorator-based registration."""

from __future__ import annotations

from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")

_TOOL_REGISTRY: dict[str, Any] = {}
_MODEL_REGISTRY: dict[str, Any] = {}
_CHECKPOINT_REGISTRY: dict[str, Any] = {}


def register_tool(name: str | None = None, category: str = ""):
    """Decorator to register a ClawTool class into the global tool registry."""

    def decorator(cls: type[T]) -> type[T]:
        tool_name = name or getattr(cls, "name", cls.__name__)
        _TOOL_REGISTRY[tool_name] = {
            "cls": cls,
            "name": tool_name,
            "category": category,
        }
        logger.debug(f"Registered tool: {tool_name} [{category}]")
        return cls

    return decorator


def get_tool(name: str) -> Any:
    """Retrieve a registered tool class by name."""
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Tool '{name}' not found in registry. Available: {list_tools()}")
    return entry["cls"]


def list_tools(category: str | None = None) -> list[str]:
    """List all registered tool names, optionally filtered by category."""
    if category is None:
        return list(_TOOL_REGISTRY.keys())
    return [n for n, e in _TOOL_REGISTRY.items() if e["category"] == category]


def get_tool_metadata(name: str) -> dict[str, Any]:
    """Get metadata for a registered tool."""
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Tool '{name}' not found")
    return {"name": entry["name"], "category": entry["category"]}


def get_all_tool_entries() -> dict[str, Any]:
    """Return the full tool registry (for introspection)."""
    return dict(_TOOL_REGISTRY)


def register_model(name: str, model_info: dict[str, Any]) -> None:
    """Register a model/checkpoint into the model registry."""
    _MODEL_REGISTRY[name] = model_info
    logger.debug(f"Registered model: {name}")


def get_model(name: str) -> dict[str, Any]:
    entry = _MODEL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Model '{name}' not found")
    return entry


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


def register_checkpoint(name: str, checkpoint_info: dict[str, Any]) -> None:
    _CHECKPOINT_REGISTRY[name] = checkpoint_info


def get_checkpoint(name: str) -> dict[str, Any]:
    entry = _CHECKPOINT_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Checkpoint '{name}' not found")
    return entry


def list_checkpoints() -> list[str]:
    return list(_CHECKPOINT_REGISTRY.keys())


def clear_all() -> None:
    """Clear all registries (useful for testing)."""
    _TOOL_REGISTRY.clear()
    _MODEL_REGISTRY.clear()
    _CHECKPOINT_REGISTRY.clear()
