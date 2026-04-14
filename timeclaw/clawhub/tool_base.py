"""ClawTool base class and ToolResult for the ClawHub tool pool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    PROCESSING = "processing"
    PROFILING = "profiling"
    DETECTION = "detection"
    DECOMPOSITION = "decomposition"
    RELATION = "relation"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
    GENERIC_LLM = "generic_llm"


class ToolIOType(str, Enum):
    X_TO_X = "X->X"
    X_TO_R = "X->R"
    X_TO_C = "X->C"
    XK_TO_R = "Xk->R"
    XK_TO_C = "Xk->C"
    XP_TO_ACCEPT = "XP->Accept"
    T_TO_R = "T->R"
    I_TO_R = "I->R"
    TX_TO_R = "TX->R"
    IX_TO_R = "IX->R"
    TI_TO_R = "TI->R"
    TIX_TO_R = "TIX->R"


class ToolResult(BaseModel):
    """Standardized output from any ClawHub tool."""

    success: bool = True
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, data: Any, **metadata: Any) -> ToolResult:
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, error=error)


class ClawTool(ABC):
    """Abstract base class for all ClawHub atomic tools."""

    name: str = ""
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = ""
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    is_stub: bool = False

    async def execute(
        self, ts_data: Any = None, *, text: Any = None, images: Any = None, **params: Any
    ) -> ToolResult:
        """Execute the tool with optional multimodal inputs.

        Args:
            ts_data: Time-series data (numpy array, pandas DataFrame, etc.)
            text: Text data (str, list of str, or list of dicts with timestamps)
            images: Image data (file paths, PIL images, or base64 strings)
            **params: Additional tool-specific parameters
        """
        try:
            result = await self._run(ts_data, text=text, images=images, **params)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))

    @abstractmethod
    async def _run(self, ts_data: Any, **params: Any) -> Any:
        """Implement the actual tool logic. Return raw result data.

        Multimodal tools receive text= and images= via **params.
        """
        ...

    def to_function_schema(self) -> dict[str, Any]:
        """Convert to LLM function-calling schema (OpenAI format)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def required_modalities(self) -> set[str]:
        """Infer the minimum required modalities from ``io_type``.

        New definition:
            ``required_modalities`` is the canonical modality signature used
            for tool exposure. A tool is eligible only when its required
            modalities are all present in the current sample.

            ``GENERIC_LLM`` tools are treated as universally applicable and
            therefore require no minimum modality subset.
        """
        if self.category == ToolCategory.GENERIC_LLM:
            return set()

        io_value = self.io_type.value if hasattr(self.io_type, "value") else str(self.io_type)
        if io_value.startswith("TIX"):
            return {"timeseries", "text", "image"}
        if io_value.startswith("TX"):
            return {"timeseries", "text"}
        if io_value.startswith("IX"):
            return {"timeseries", "image"}
        if io_value.startswith("TI"):
            return {"text", "image"}
        if io_value.startswith("T"):
            return {"text"}
        if io_value.startswith("I"):
            return {"image"}
        return {"timeseries"}

    def input_contract(self) -> dict[str, Any]:
        """Return structured runtime input requirements for the current artifact.

        New definition:
            ``input_contract`` is a lightweight, machine-readable summary of the
            minimum artifact conditions a tool expects at call time. The
            executor can use it to hide tools that are incompatible with the
            current working artifact instead of relying only on free-text tool
            descriptions.

        Supported keys:
            - ``artifact_types``: allowed current artifact types, e.g. ``["series"]``
            - ``min_series_channels``: minimum number of channels/columns in the
              current time-series artifact
        """
        return {}

    def runtime_compatibility_error(
        self,
        ts_data: Any = None,
        *,
        current_series_state: dict[str, Any] | None = None,
    ) -> str | None:
        contract = self.input_contract()
        if not isinstance(contract, dict) or not contract:
            return None

        artifact_type = str((current_series_state or {}).get("artifact_type") or "").strip()
        allowed_artifact_types = [
            str(item).strip()
            for item in (contract.get("artifact_types") or [])
            if str(item).strip()
        ]
        if allowed_artifact_types and artifact_type and artifact_type not in allowed_artifact_types:
            allowed_text = ", ".join(allowed_artifact_types)
            return (
                f"`{self.name}` requires current artifact type in {{{allowed_text}}}, "
                f"but the working artifact is `{artifact_type}`."
            )

        min_series_channels = contract.get("min_series_channels")
        if min_series_channels is not None:
            channel_count = self._series_channel_count(ts_data)
            try:
                required_channels = int(min_series_channels)
            except (TypeError, ValueError):
                required_channels = 0
            if channel_count < required_channels:
                return (
                    f"`{self.name}` requires at least {required_channels} time-series channel(s), "
                    f"but the current working data has {channel_count}."
                )

        return None

    @staticmethod
    def _series_channel_count(ts_data: Any) -> int:
        if ts_data is None:
            return 0
        arr = np.asarray(ts_data)
        if arr.ndim <= 1:
            return 1 if arr.size > 0 else 0
        return int(arr.shape[1])

    @staticmethod
    def resolve_llm(llm: Any, model: str | None = None) -> Any | None:
        """Resolve the LLM backend from a single backend or {"light": ..., "heavy": ...} dict.

        When *llm* is a dict, *model* selects the key ("light" or "heavy").
        Falls back to "heavy" → "light" → first available when the key is missing.
        When *llm* is a single backend, returns it directly.
        """
        if llm is None:
            return None
        if not isinstance(llm, dict):
            return llm
        if model and model in llm:
            return llm[model]
        return llm.get("heavy") or llm.get("light") or next(iter(llm.values()), None)

    @staticmethod
    def extract_text_content(item: Any) -> str:
        """Extract text content from various dict formats (supports 'text', 'content' keys)."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("text", item.get("content", ""))
        return str(item)

    def __repr__(self) -> str:
        stub = " [STUB]" if self.is_stub else ""
        return f"<ClawTool {self.name} [{self.category.value}]{stub}>"
