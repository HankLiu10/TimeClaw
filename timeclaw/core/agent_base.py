"""Abstract base class for all TimeClaw agents with async lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from timeclaw.core import trace_logger as trace
from timeclaw.core.evidence import EvidenceLog
from timeclaw.core.message import AgentRole, Message
from timeclaw.core.tea_protocol import TaskEnvelope
from timeclaw.memory.domain_knowledge import DomainKnowledge

if TYPE_CHECKING:
    from timeclaw.llm.base import LLMBackend

_shared_domain_knowledge: DomainKnowledge | None = None


def get_domain_knowledge() -> DomainKnowledge:
    """Get or create the shared DomainKnowledge instance."""
    global _shared_domain_knowledge
    if _shared_domain_knowledge is None:
        _shared_domain_knowledge = DomainKnowledge()
        _shared_domain_knowledge.load_all()
    return _shared_domain_knowledge


class AgentBase(ABC):
    """Base class implementing the perceive -> plan -> act -> reflect lifecycle."""

    role: AgentRole

    def __init__(self, llm: LLMBackend, config: dict[str, Any] | None = None):
        self.llm = llm
        self.config = config or {}
        self.evidence_log = EvidenceLog()
        self._step_count = 0
        self.domain_knowledge = get_domain_knowledge()

    def get_knowledge_context(self, task_type: str) -> str:
        """Get relevant domain knowledge for the current task."""
        ctx = self.domain_knowledge.get_context_for_task(task_type)
        return f"\n\n## Domain Knowledge\n{ctx}" if ctx else ""

    @abstractmethod
    async def perceive(self, envelope: TaskEnvelope) -> dict[str, Any]:
        """Observe the environment and extract relevant information."""
        ...

    @abstractmethod
    async def plan(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Generate an action plan based on perception."""
        ...

    @abstractmethod
    async def act(self, plan: dict[str, Any], envelope: TaskEnvelope) -> dict[str, Any]:
        """Execute the plan using tools and models."""
        ...

    @abstractmethod
    async def reflect(self, action_result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the action result and provide feedback."""
        ...

    async def run(self, envelope: TaskEnvelope) -> dict[str, Any]:
        """Execute the full agent lifecycle."""
        logger.info(f"[{self.role.value}] Starting agent run")
        self._step_count += 1

        perception = await self.perceive(envelope)
        plan = await self.plan(perception)
        result = await self.act(plan, envelope)
        feedback = await self.reflect(result)

        logger.info(f"[{self.role.value}] Agent run complete (step {self._step_count})")
        trace.agent_step(self.role.value, "reflect", feedback)
        return {
            "perception": perception,
            "plan": plan,
            "result": result,
            "feedback": feedback,
        }

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.evidence_log = EvidenceLog()
        self._step_count = 0

    async def send_message(self, msg: Message) -> None:
        """Send a message (placeholder for future message bus)."""
        logger.debug(f"[{self.role.value}] -> [{msg.receiver}]: {msg.type.value}")
