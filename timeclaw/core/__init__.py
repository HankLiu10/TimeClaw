"""Core abstractions: Agent, TEA protocol, Evidence, Task DAG, Registry."""

from timeclaw.core.agent_base import AgentBase
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.evidence import EvidenceEntry, EvidenceLog, QualityGate, QualityGateResult
from timeclaw.core.message import AgentRole, Message, MessageType
from timeclaw.core.registry import get_tool, list_tools, register_tool
from timeclaw.core.task_graph import TaskDAG, TaskNode, TaskStatus
from timeclaw.core.tea_protocol import (
    AgentState,
    EnvironmentSnapshot,
    TaskEnvelope,
    TaskSpec,
)

__all__ = [
    "AgentBase",
    "AgentRole",
    "AgentState",
    "EnvironmentSnapshot",
    "EvidenceEntry",
    "EvidenceLog",
    "Message",
    "MessageType",
    "QualityGate",
    "QualityGateResult",
    "TaskDAG",
    "TaskEnvelope",
    "TaskNode",
    "TaskSpec",
    "TaskStatus",
    "TimeClawConfig",
    "get_tool",
    "list_tools",
    "register_tool",
]
