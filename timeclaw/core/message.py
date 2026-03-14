"""Agent inter-communication message types."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TASK = "task"
    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    PLAN = "plan"
    ACTION = "action"
    FEEDBACK = "feedback"
    RESULT = "result"
    ERROR = "error"


class AgentRole(str, Enum):
    MANAGER = "manager"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    REASONER = "reasoner"
    SYSTEM = "system"


class Message(BaseModel):
    """Base message exchanged between agents."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    type: MessageType
    sender: AgentRole
    receiver: AgentRole | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    parent_id: str | None = None


class TaskMessage(Message):
    """Task assignment from Manager to other agents."""

    type: MessageType = MessageType.TASK
    sender: AgentRole = AgentRole.MANAGER
    task_type: str = ""
    sub_goal: str = ""
    priority: int = 0


class ObservationMessage(Message):
    """Structured observation from tool execution."""

    type: MessageType = MessageType.OBSERVATION
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    error_msg: str | None = None


class EvidenceMessage(Message):
    """Evidence entry for the reasoning trace."""

    type: MessageType = MessageType.EVIDENCE
    step: int = 0
    thought: str = ""
    tool_name: str = ""
    observation: Any = None
    critic_feedback: str = ""


class PlanMessage(Message):
    """Task DAG plan from Manager."""

    type: MessageType = MessageType.PLAN
    sender: AgentRole = AgentRole.MANAGER
    dag_nodes: list[dict[str, Any]] = Field(default_factory=list)
    dag_edges: list[tuple[str, str]] = Field(default_factory=list)


class ResultMessage(Message):
    """Final result delivered to the user."""

    type: MessageType = MessageType.RESULT
    prediction: Any = None
    evidence_summary: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
