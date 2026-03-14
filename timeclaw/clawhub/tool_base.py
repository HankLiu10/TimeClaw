"""ClawTool base class and ToolResult for the ClawHub tool pool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

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
