#!/usr/bin/env python3
"""Distill TimeClaw full-dump logs into concise, readable analysis reports.

For each task in the log, produces a structured report:
  1. Original problem & inputs (text, timeseries, images, etc.)
  2. Clear execution path (pipeline stages as a flow)
  3. Summarised tool calls & LLM reasoning (input→output digest)
  4. Final output with full justification
  5. Comparison against ground truth
  6. Reflection on errors (if prediction != ground truth)

Design principles:
  - Generic parser: no hard-coded task types, modalities, or tool names
  - LLM-powered summarisation: each task's raw log is sent to an LLM
    that produces the structured report
  - Extensible: new tasks, datasets, modalities are handled automatically
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AZURE_KEY = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
AZURE_BASE = "https://asure-test-openai-004.openai.azure.com/"
AZURE_VER = "2024-12-01-preview"


# ─── Data structures ──────────────────────────────────────────────────

@dataclass
class TaskBlock:
    """One task's complete log, extracted from the full dump."""
    task_type: str          # e.g. FORECAST, TREND, INDICATOR, MCQA, CORRELATION
    dataset: str            # e.g. weather_aligned_short
    sample_file: str        # e.g. USW00003017_0.json
    raw_lines: list[str]    # all log lines for this task
    input_shape: str = ""
    output_shape: str = ""
    question: str = ""
    labels: dict = field(default_factory=dict)
    prediction: str = ""
    gate_passed: bool = False
    elapsed_ms: int = 0


# ─── Step 1: Parse log into per-task blocks ───────────────────────────

def parse_task_blocks(log_path: Path) -> list[TaskBlock]:
    """Split the full-dump log into individual task blocks.

    Detection is generic: looks for the task header pattern
    '任务: XXX | 数据集: YYY' surrounded by '===' separators.
    """
    lines = log_path.read_text(encoding="utf-8").split("\n")
    blocks: list[TaskBlock] = []
    task_header_re = re.compile(r"任务:\s*(\S+)\s*\|\s*数据集:\s*(\S+)")

    i = 0
    while i < len(lines):
        m = task_header_re.search(lines[i])
        if m:
            task_type = m.group(1)
            dataset = m.group(2)

            # Find the start (go back to the '===' line)
            start = max(0, i - 1)
            while start > 0 and "===" not in lines[start]:
                start -= 1

            # Find the end: next task header or EOF
            j = i + 1
            while j < len(lines):
                if task_header_re.search(lines[j]):
                    # Go back to the '===' before next header
                    end = j - 1
                    while end > i and "===" not in lines[end]:
                        end -= 1
                    break
                j += 1
            else:
                end = len(lines)

            raw = lines[start:end]
            block = TaskBlock(task_type=task_type, dataset=dataset, sample_file="", raw_lines=raw)
            _extract_metadata(block)
            blocks.append(block)
            i = j if j < len(lines) else len(lines)
        else:
            i += 1

    return blocks


def _extract_metadata(block: TaskBlock):
    """Extract structured metadata from raw log lines."""
    for line in block.raw_lines:
        if "样本:" in line:
            block.sample_file = line.split("样本:")[-1].strip()
        elif "输入形状:" in line:
            block.input_shape = line.split("输入形状:")[-1].strip()
        elif "输出形状:" in line:
            block.output_shape = line.split("输出形状:")[-1].strip()
        elif "问题:" in line:
            block.question = line.split("问题:")[-1].strip()
        elif "标签:" in line:
            try:
                block.labels = eval(line.split("标签:")[-1].strip())
            except Exception:
                block.labels = {"raw": line.split("标签:")[-1].strip()}
        elif "质量门控:" in line:
            block.gate_passed = "通过" in line and "未通过" not in line
        elif "总耗时:" in line and "ms" in line:
            try:
                block.elapsed_ms = int(re.search(r"(\d+)ms", line).group(1))
            except Exception:
                pass
        elif re.search(r"\[(FORECAST|TREND|INDICATOR|MCQA|CORRELATION|[A-Z_]+)\]\s*完成", line):
            # Extract final prediction from the summary line
            pass

    # Extract prediction from the final summary block
    for line in reversed(block.raw_lines):
        if "预测:" in line and "[Pipeline]" not in line:
            block.prediction = line.split("预测:")[-1].strip()
            break


# ─── Step 2: Load ground truth ───────────────────────────────────────

def load_ground_truth(block: TaskBlock) -> dict[str, Any]:
    """Load ground truth for a task block from the dataset.

    Returns a dict with keys depending on task type.
    Generic: tries to load the sample and extract whatever GT is available.
    """
    try:
        from timeclaw.benchmark.mtbench_adapter import load_dataset
    except ImportError:
        return {"error": "cannot import mtbench_adapter"}

    samples = load_dataset(block.dataset)
    if not samples:
        return {"error": f"dataset {block.dataset} empty"}

    sample = None
    for s in samples:
        if s.file == block.sample_file:
            sample = s
            break
    if sample is None:
        sample = samples[0]

    gt: dict[str, Any] = {"sample_file": sample.file}

    if sample.output_window is not None:
        gt["output_window"] = sample.output_window.tolist()
        gt["output_shape"] = str(sample.output_window.shape)

    if sample.answer:
        gt["answer"] = sample.answer

    if sample.question:
        gt["question"] = sample.question

    if sample.labels:
        gt["labels"] = {k: v for k, v in sample.labels.items()
                        if not isinstance(v, (np.ndarray,))}

    return gt


# ─── Step 3: LLM-powered distillation ────────────────────────────────

DISTILL_SYSTEM_PROMPT = """\
你是一个AI系统分析专家。你的任务是将一段时间序列自治Agent平台（TimeClaw）的运行日志提炼成结构化的分析报告。

## 报告格式（严格按以下结构输出，使用 Markdown）

### 1. 任务概述
- 任务类型、数据集、样本
- 原始问题/描述
- 输入数据概况（各模态：时序形状、文本摘要、图像描述等）

### 2. 执行路径
用流程图的方式展示 Pipeline 的执行阶段：
```
T+0 规划 → T+1 诊断 → T+2 路由 → T+3 执行 → T+4 验证 → T+5 交付
```
标注每个阶段的关键决策和耗时。

### 3. 各阶段详细分析

对每个阶段，按以下格式总结：

#### T+X [阶段名]
- **Agent**: 哪个Agent在工作
- **输入**: 该阶段收到了什么
- **工具调用**（如有）:
  - 工具名(参数) → 关键结果（一句话）
- **LLM 思考**: 
  - 输入摘要（一句话概括发给LLM的核心内容）
  - 输出摘要（一句话概括LLM的决策/结论）
- **关键决策**: 这一步做出了什么决定

### 4. 最终输出
- 预测/回答内容
- 置信度
- 推理链（LLM给出的完整理由，简明扼要地转述）
- 质量门控结果（通过/未通过，各检查项）

### 5. Ground Truth 对比
- Ground Truth 是什么
- 预测是否正确
- 如果是数值预测，给出误差指标（MAE等）
- 如果是分类/选择题，给出是否匹配

### 6. 反思与诊断（仅在预测不正确时）
- 错误出在哪个阶段
- 根本原因分析（是工具选择错误？LLM推理错误？信息丢失？模态融合问题？）
- 改进建议

## 重要规则
1. 简洁为王：每个工具调用和LLM思考只用1-2句话总结，不要复制原始日志
2. 保留关键数字：统计量、预测值、置信度等数字要保留
3. 突出决策链：让读者一眼看清"为什么做了这个决定"
4. 中文输出，专业术语可保留英文
5. 如果日志中有多个模态分支（timeseries/text/image等），分别总结每个分支
6. 不要遗漏任何阶段，但也不要展开无关细节
"""


async def distill_one_task(block: TaskBlock, ground_truth: dict, sem: asyncio.Semaphore) -> str:
    """Send one task's log + ground truth to LLM for distillation."""
    import litellm

    raw_log = "\n".join(block.raw_lines)

    # Truncate if too long (keep first and last parts)
    max_chars = 60000
    if len(raw_log) > max_chars:
        half = max_chars // 2
        raw_log = raw_log[:half] + "\n\n... [日志中间部分省略] ...\n\n" + raw_log[-half:]

    gt_str = json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str)

    user_msg = f"""请分析以下 TimeClaw 运行日志，生成结构化分析报告。

## 任务信息
- 任务类型: {block.task_type}
- 数据集: {block.dataset}
- 样本: {block.sample_file}
- 输入形状: {block.input_shape}
- 输出形状: {block.output_shape}
- 质量门控: {"通过" if block.gate_passed else "未通过"}
- 耗时: {block.elapsed_ms}ms

## Ground Truth
```json
{gt_str}
```

## 完整运行日志
```
{raw_log}
```

请按照系统提示中的格式，生成完整的分析报告。"""

    async with sem:
        try:
            resp = await litellm.acompletion(
                model="azure/gpt-5.1",
                api_base=AZURE_BASE,
                api_key=AZURE_KEY,
                api_version=AZURE_VER,
                messages=[
                    {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=8192,
                timeout=300.0,
            )
            return resp.choices[0].message.content or "[LLM 返回为空]"
        except Exception as e:
            return f"[提炼失败: {e}]"


# ─── Step 4: Assemble final report ───────────────────────────────────

def build_report_header() -> str:
    from datetime import datetime
    return f"""# TimeClaw 运行日志分析报告

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 日志来源: timeclaw_full_dump.log

---

"""


def build_task_section(idx: int, block: TaskBlock, report: str) -> str:
    status = "✅ 通过" if block.gate_passed else "❌ 未通过"
    return f"""## 任务 {idx}: {block.task_type} ({block.dataset})

| 属性 | 值 |
|------|------|
| 样本 | `{block.sample_file}` |
| 输入形状 | `{block.input_shape}` |
| 输出形状 | `{block.output_shape}` |
| 质量门控 | {status} |
| 耗时 | {block.elapsed_ms}ms |

{report}

---

"""


# ─── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Distill TimeClaw full-dump logs into structured analysis reports"
    )
    parser.add_argument("input", type=str, nargs="?",
                        default="timeclaw_full_dump.log",
                        help="Input log file (default: timeclaw_full_dump.log)")
    parser.add_argument("-o", "--output", type=str,
                        default="timeclaw_analysis_report.md",
                        help="Output report file (default: timeclaw_analysis_report.md)")
    parser.add_argument("--task", type=str, default=None,
                        help="Only distill this task type (e.g. FORECAST, TREND)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent LLM calls (default: 5)")
    parser.add_argument("--lang", type=str, default="zh",
                        choices=["zh", "en"],
                        help="Report language (default: zh)")
    args = parser.parse_args()

    log_path = Path(args.input)
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📖 解析日志: {log_path}")
    blocks = parse_task_blocks(log_path)
    print(f"   找到 {len(blocks)} 个任务")

    if args.task:
        blocks = [b for b in blocks if b.task_type.upper() == args.task.upper()]
        print(f"   筛选后: {len(blocks)} 个任务 (type={args.task})")

    if not blocks:
        print("❌ 没有找到任务", file=sys.stderr)
        sys.exit(1)

    for b in blocks:
        print(f"   - {b.task_type} | {b.dataset} | {b.sample_file} | "
              f"gate={'PASS' if b.gate_passed else 'FAIL'} | {b.elapsed_ms}ms")

    print(f"\n🔍 加载 Ground Truth...")
    ground_truths = []
    for b in blocks:
        gt = load_ground_truth(b)
        ground_truths.append(gt)
        gt_preview = {k: (str(v)[:80] + "..." if len(str(v)) > 80 else v)
                      for k, v in gt.items() if k != "output_window"}
        print(f"   - {b.task_type}: {gt_preview}")

    print(f"\n🌐 开始 LLM 提炼 (并发数: {args.concurrency})...")
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        distill_one_task(block, gt, sem)
        for block, gt in zip(blocks, ground_truths)
    ]
    reports = await asyncio.gather(*tasks)

    for i, (block, report) in enumerate(zip(blocks, reports)):
        status = "✅" if "[提炼失败" not in report else "❌"
        print(f"   {status} {block.task_type} 提炼完成")

    print(f"\n📝 生成报告...")
    output = build_report_header()

    for i, (block, report) in enumerate(zip(blocks, reports), 1):
        output += build_task_section(i, block, report)

    # Add summary section
    output += "## 总体总结\n\n"
    output += f"| 任务 | 数据集 | 质量门控 | 耗时 |\n"
    output += f"|------|--------|----------|------|\n"
    for b in blocks:
        status = "✅ 通过" if b.gate_passed else "❌ 未通过"
        output += f"| {b.task_type} | {b.dataset} | {status} | {b.elapsed_ms}ms |\n"

    out_path = Path(args.output)
    out_path.write_text(output, encoding="utf-8")
    print(f"\n✅ 报告已保存到: {out_path} ({len(output)} 字符)")


if __name__ == "__main__":
    os.environ.setdefault("AZURE_API_KEY", AZURE_KEY)
    os.environ.setdefault("AZURE_API_BASE", AZURE_BASE)
    os.environ.setdefault("AZURE_API_VERSION", AZURE_VER)
    asyncio.run(main())
