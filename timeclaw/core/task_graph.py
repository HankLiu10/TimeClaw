"""Task topology DAG definition and execution engine."""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """A single node in the task DAG."""

    id: str
    name: str
    agent_role: str
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def mark_completed(self, result: Any = None) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error


class TaskDAG(BaseModel):
    """Directed acyclic graph of task nodes."""

    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    edges: list[tuple[str, str]] = Field(default_factory=list)

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        self.edges.append((from_id, to_id))

    @property
    def adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = defaultdict(list)
        for src, dst in self.edges:
            adj[src].append(dst)
        return dict(adj)

    @property
    def in_degree(self) -> dict[str, int]:
        deg: dict[str, int] = {nid: 0 for nid in self.nodes}
        for _, dst in self.edges:
            deg[dst] = deg.get(dst, 0) + 1
        return deg

    def topological_order(self) -> list[str]:
        """Return nodes in topological order via Kahn's algorithm."""
        in_deg = self.in_degree
        queue: deque[str] = deque(nid for nid, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in self.adjacency.get(nid, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            raise ValueError("Task graph contains a cycle")
        return order

    def parallel_groups(self) -> list[list[str]]:
        """Return groups of nodes that can execute in parallel (same depth level)."""
        in_deg = self.in_degree
        queue: deque[str] = deque(nid for nid, d in in_deg.items() if d == 0)
        groups: list[list[str]] = []
        while queue:
            group = list(queue)
            groups.append(group)
            next_queue: deque[str] = deque()
            for nid in group:
                for neighbor in self.adjacency.get(nid, []):
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
        return groups

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all completed."""
        completed = {nid for nid, n in self.nodes.items() if n.status == TaskStatus.COMPLETED}
        ready = []
        for nid, node in self.nodes.items():
            if node.status != TaskStatus.PENDING:
                continue
            deps = [src for src, dst in self.edges if dst == nid]
            if all(d in completed for d in deps):
                ready.append(node)
        return ready

    @property
    def is_complete(self) -> bool:
        return all(
            n.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for n in self.nodes.values()
        )

    @property
    def has_failures(self) -> bool:
        return any(n.status == TaskStatus.FAILED for n in self.nodes.values())
