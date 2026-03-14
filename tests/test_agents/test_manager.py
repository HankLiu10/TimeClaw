"""Tests for the Manager Agent."""

from __future__ import annotations

import numpy as np
import pytest

from timeclaw.agents.manager import ManagerAgent
from timeclaw.core.tea_protocol import (
    AgentState,
    EnvironmentSnapshot,
    TaskEnvelope,
    TaskSpec,
)


@pytest.mark.asyncio
async def test_manager_builds_dag(mock_llm_backend):
    manager = ManagerAgent(mock_llm_backend)

    env = EnvironmentSnapshot()
    env.set_data(np.random.randn(100))

    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="forecast", description="Forecast next 10 steps"),
        environment=env,
        agent_state=AgentState(agent_role="system"),
    )

    result = await manager.run(envelope)
    dag = result["result"]["dag"]

    assert len(dag.nodes) == 3
    assert "analyze" in dag.nodes
    assert "execute_ts" in dag.nodes
    assert "fuse_verify" in dag.nodes

    order = dag.topological_order()
    assert order.index("analyze") < order.index("execute_ts")
    assert order.index("execute_ts") < order.index("fuse_verify")
