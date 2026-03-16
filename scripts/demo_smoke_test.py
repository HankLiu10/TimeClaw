"""Smoke test: run 1 sample through the full pipeline to verify everything works."""

import asyncio
import json
import sys
import traceback

from loguru import logger


async def main():
    logger.info("=" * 60)
    logger.info("TimeClaw Smoke Test — Post-Cleanup Verification")
    logger.info("=" * 60)

    errors: list[str] = []

    # ── 1. Tool registry (in-process) ─────────────────────────────
    logger.info("[1/5] Testing tool discovery & registration...")
    try:
        from timeclaw.clawhub.tool_registry import discover_and_register, list_tool_instances
        count = discover_and_register()
        tools = list_tool_instances()
        tool_names = sorted(t.name for t in tools)
        logger.info(f"  Registered {count} tools: {tool_names}")

        must_have = ["llm_reasoning", "arima_forecast", "prophet_forecast"]
        for name in must_have:
            if name not in tool_names:
                errors.append(f"Missing expected tool: {name}")
                logger.error(f"  FAIL: {name} not registered")
            else:
                logger.info(f"  OK: {name} found")

        categories = sorted(set(t.category.value for t in tools))
        logger.info(f"  Categories: {categories}")
        if "generic_llm" not in categories:
            errors.append("generic_llm category missing from in-process registry")
            logger.error("  FAIL: generic_llm category not found")
        else:
            logger.info("  OK: generic_llm category present")
    except Exception as e:
        errors.append(f"Tool registration failed: {e}")
        logger.error(f"  FAIL: {traceback.format_exc()}")

    # ── 2. Hierarchical memory ────────────────────────────────────
    logger.info("[2/5] Testing hierarchical memory...")
    try:
        from timeclaw.memory.hierarchical import HierarchicalMemory
        mem = HierarchicalMemory(enabled=True)
        summary = mem.summary()
        logger.info(f"  Memory summary: {json.dumps(summary, default=str)}")
        logger.info("  OK: HierarchicalMemory initialised")
    except Exception as e:
        errors.append(f"HierarchicalMemory failed: {e}")
        logger.error(f"  FAIL: {traceback.format_exc()}")

    # ── 3. Dataset loading ────────────────────────────────────────
    logger.info("[3/5] Testing dataset loading...")
    try:
        from timeclaw.benchmark.mtbench_adapter import load_dataset
        samples = load_dataset("weather_aligned_short")
        logger.info(f"  Loaded {len(samples)} samples from weather_aligned_short")
        sample = samples[0]
        logger.info(f"  Sample 0: domain={sample.domain}, tasks={sample.task_types}")
        logger.info(f"  input_window shape: {sample.input_window.shape}")
        logger.info(f"  output_window shape: {sample.output_window.shape if sample.output_window is not None else 'None'}")
        logger.info(f"  text: {sample.text[:100]}..." if sample.text else "  text: (empty)")
        logger.info("  OK: Dataset loaded")
    except Exception as e:
        errors.append(f"Dataset loading failed: {e}")
        logger.error(f"  FAIL: {traceback.format_exc()}")

    # ── 4. PipelineRunner single sample (test mode) ───────────────
    logger.info("[4/5] Testing PipelineRunner (test mode, 1 sample)...")
    try:
        from timeclaw.pipeline.runner import PipelineRunner
        from timeclaw.core.config import TimeClawConfig

        cfg = TimeClawConfig.load()
        cfg.mcp.enabled = False
        pipeline = PipelineRunner(cfg, disable_memory=True)

        sample = samples[0]
        horizon = len(sample.output_window) if sample.output_window is not None else 24
        text_data = [{"text": sample.text}] if sample.text else None
        result = await pipeline.run(
            data=sample.input_window,
            task_type="forecast",
            description="Predict the next temperature values based on the historical hourly observations.",
            domain=sample.domain,
            horizon=horizon,
            text_data=text_data,
        )

        logger.info(f"  Result keys: {list(result.keys())}")
        answer_type = result.get("answer_type", "")
        prediction = result.get("prediction")
        reasoning = result.get("reasoning", "")[:200]
        exec_log = result.get("execution_log", [])
        tools_used = [e.get("tool", "") for e in exec_log if e.get("tool")]

        logger.info(f"  answer_type: {answer_type}")
        logger.info(f"  tools_used: {tools_used}")
        logger.info(f"  reasoning: {reasoning}...")

        if prediction is not None:
            if hasattr(prediction, '__len__'):
                logger.info(f"  prediction length: {len(prediction)}")
            else:
                logger.info(f"  prediction: {prediction}")
            logger.info("  OK: Pipeline produced prediction")
        else:
            errors.append("Pipeline returned no prediction")
            logger.error("  FAIL: No prediction returned")

        await pipeline.shutdown()
    except Exception as e:
        errors.append(f"PipelineRunner failed: {e}")
        logger.error(f"  FAIL: {traceback.format_exc()}")

    # ── 5. Training mode (1 sample with GT) ───────────────────────
    logger.info("[5/5] Testing PipelineRunner (train mode, 1 sample)...")
    try:
        from timeclaw.benchmark.train_loop import _extract_ground_truth
        cfg2 = TimeClawConfig.load()
        cfg2.mcp.enabled = False
        pipeline2 = PipelineRunner(cfg2, disable_memory=False)

        gt = _extract_ground_truth(sample, "forecast")
        logger.info(f"  Ground truth extracted, type={type(gt).__name__}, len={len(gt) if hasattr(gt, '__len__') else 'N/A'}")

        result2 = await pipeline2.run(
            data=sample.input_window,
            task_type="forecast",
            description="Predict the next temperature values based on the historical hourly observations.",
            domain=sample.domain,
            horizon=horizon,
            text_data=text_data,
            mode="train",
            ground_truth=gt,
        )

        answer_type2 = result2.get("answer_type", "")
        exec_log2 = result2.get("execution_log", [])
        tools2 = [e.get("tool", "") for e in exec_log2 if e.get("tool")]

        logger.info(f"  answer_type: {answer_type2}")
        logger.info(f"  tools_used: {tools2}")

        eval_used = any(e.get("tool") == "evaluate_against_gt" for e in exec_log2)
        if eval_used:
            logger.info("  OK: evaluate_against_gt was called during training")
        else:
            logger.warning("  WARN: evaluate_against_gt not called (agent chose not to)")

        mem_summary = pipeline2.memory_summary()
        logger.info(f"  Memory after training: {json.dumps(mem_summary, default=str)}")

        dn = mem_summary.get("daily_notes", 0)
        dn_count = dn.get("total", 0) if isinstance(dn, dict) else dn
        if dn_count > 0:
            logger.info("  OK: Learning outcome saved to Daily Notes")
        else:
            errors.append("Training did not save to Daily Notes")
            logger.error("  FAIL: No Daily Notes recorded")

        await pipeline2.shutdown()
    except Exception as e:
        errors.append(f"Training mode failed: {e}")
        logger.error(f"  FAIL: {traceback.format_exc()}")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=" * 60)
    if errors:
        logger.error(f"SMOKE TEST FAILED — {len(errors)} error(s):")
        for e in errors:
            logger.error(f"  ✗ {e}")
        sys.exit(1)
    else:
        logger.info("SMOKE TEST PASSED — all components working")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
