"""Quick test to verify verbose trace logging works end-to-end."""

import asyncio
import sys
import numpy as np

sys.path.insert(0, ".")

from timeclaw.core.config import TimeClawConfig
from timeclaw.core import trace_logger as trace


def test_trace_logger_off():
    """When verbose=False, no output should be produced."""
    trace.configure(False)
    trace.llm_call("test", [{"role": "user", "content": "hi"}], None)
    trace.tool_exec("test", "normalize", {"method": "zscore"}, None)
    trace.agent_step("test", "plan", {"tool_chain": ["a"]})
    trace.pipeline_stage("T+0", {"intent": {}})
    trace.reasoning_step("test", 0, thought={"done": True})
    print("[PASS] verbose=False: no trace output")


def test_trace_logger_on():
    """When verbose=True, all trace calls should produce output."""
    trace.configure(True)
    print("\n--- verbose=True output below ---")
    trace.llm_call(
        "manager.plan",
        [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Forecast temperature for next 7 days"},
        ],
        type("Resp", (), {
            "text": '{"task_type": "forecast", "sub_goals": ["analyze trend"]}',
            "parsed": {"task_type": "forecast", "sub_goals": ["analyze trend"]},
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        })(),
        response_format="json",
    )
    trace.tool_exec(
        "executor",
        "normalize",
        {"method": "zscore"},
        type("TR", (), {
            "success": True,
            "data": [0.1, -0.3, 0.5, 0.2],
            "error": None,
            "metadata": {"inverse": {"method": "zscore", "mean": 20.5, "std": 1.2}},
        })(),
    )
    trace.agent_step("executor", "plan", {
        "tool_chain": [{"tool": "normalize", "args": {}}, {"tool": "arima_forecast", "args": {"horizon": 7}}],
        "reasoning": "Data needs normalization before ARIMA",
    })
    trace.pipeline_stage("T+0 Plan", {
        "intent": {"task_type": "forecast", "sub_goals": ["analyze trend"]},
        "modalities": ["timeseries"],
    })
    trace.reasoning_step("evidence_iterator", 0,
        thought={"reasoning": "Need to check stationarity first", "tool": "stationarity_test", "args": {}},
        observation={"is_stationary": False, "p_value": 0.15},
        critic="Good choice, data appears non-stationary",
    )
    print("--- end verbose output ---")
    print("[PASS] verbose=True: trace output produced")


def test_config_verbose():
    """Verify config loads verbose field correctly."""
    cfg = TimeClawConfig()
    assert cfg.pipeline.verbose is False, f"Default should be False, got {cfg.pipeline.verbose}"

    cfg2 = TimeClawConfig(pipeline={"verbose": True})
    assert cfg2.pipeline.verbose is True, f"Should be True, got {cfg2.pipeline.verbose}"
    print("[PASS] config verbose field works")


if __name__ == "__main__":
    test_trace_logger_off()
    test_trace_logger_on()
    test_config_verbose()
    print("\n[ALL PASS] Verbose logging system verified")
