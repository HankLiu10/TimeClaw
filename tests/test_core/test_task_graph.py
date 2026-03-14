"""Tests for the task DAG."""

from __future__ import annotations

import pytest

from timeclaw.core.task_graph import TaskDAG, TaskNode, TaskStatus


class TestTaskDAG:
    def _make_linear_dag(self) -> TaskDAG:
        dag = TaskDAG()
        dag.add_node(TaskNode(id="a", name="A", agent_role="analyzer"))
        dag.add_node(TaskNode(id="b", name="B", agent_role="executor"))
        dag.add_node(TaskNode(id="c", name="C", agent_role="reasoner"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        return dag

    def test_topological_order(self):
        dag = self._make_linear_dag()
        order = dag.topological_order()
        assert order == ["a", "b", "c"]

    def test_parallel_groups(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(id="a", name="A", agent_role="analyzer"))
        dag.add_node(TaskNode(id="b", name="B", agent_role="analyzer"))
        dag.add_node(TaskNode(id="c", name="C", agent_role="executor"))
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")
        groups = dag.parallel_groups()
        assert len(groups) == 2
        assert set(groups[0]) == {"a", "b"}
        assert groups[1] == ["c"]

    def test_get_ready_nodes(self):
        dag = self._make_linear_dag()
        ready = dag.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "a"

        dag.nodes["a"].mark_completed()
        ready = dag.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "b"

    def test_is_complete(self):
        dag = self._make_linear_dag()
        assert not dag.is_complete
        for node in dag.nodes.values():
            node.mark_completed()
        assert dag.is_complete

    def test_cycle_detection(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(id="a", name="A", agent_role="x"))
        dag.add_node(TaskNode(id="b", name="B", agent_role="x"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            dag.topological_order()
