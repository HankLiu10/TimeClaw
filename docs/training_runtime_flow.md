# TimeClaw 训练阶段完整运行流程

这份文档只描述当前仓库里的真实实现，不描述理想设计，不补脑补流程。

我重点整理了 4 件事：

1. 训练入口到底在哪里。
2. 当前训练到底会跑多少任务、多少样本。
3. 一条真实 sample 在训练时每一步拿到了什么、做了什么、生成了什么。
4. Notes / MEMORY / TOOLS / SKILLS / trace / train_results 各自长什么样。

本文基于以下真实代码和本地真实产物整理：

- `timeclaw/benchmark/run_learning.py`
- `timeclaw/benchmark/train_loop.py`
- `timeclaw/pipeline/runner.py`
- `timeclaw/agents/executor.py`
- `timeclaw/memory/hierarchical.py`
- `timeclaw/benchmark/mtbench_support/learning_analysis.py`
- `.timeclaw/train_results/`
- `.timeclaw/traces/`
- `.timeclaw/hierarchical_memory/`

---

## 1. 先定义几个词

下面这些词在代码里反复出现。如果不先定义，后面很容易绕。

### 1.1 原始样本

原始样本指 MTBench 数据集里的一条样本，统一被加载成 `MTBenchSample`。

它通常包含：

- `input_window`
- `output_window`
- `text`
- `timestamps`
- 可能还有 `question` / `answer` / `labels`

### 1.2 训练任务

训练任务指“某个样本在某个 task 上的一次训练运行”。

例子：

- 同一条 `weather_aligned_short` 样本
- 会被扩成 `forecast`、`trend`、`trend_past`

所以：

- 1 条原始样本
- 不一定只对应 1 次训练

### 1.3 `pipeline_task`

`pipeline_task` 是真正传进 `PipelineRunner.run()` 的任务类型。

它和 benchmark task 不完全一样，会先做一次映射：

- `mcqa -> qa`
- `trend -> classification`
- `trend_past -> classification`
- `indicator_macd -> indicator`
- `indicator_upper_bb -> indicator`

例子：

- benchmark task 是 `trend`
- 进入 pipeline 时变成 `classification`

### 1.4 `scope_key`

`scope_key` 是 memory 系统最重要的路由键。

它决定：

- note 写到哪个 scope 文件
- MEMORY 从哪个 scope 读取
- TOOLS.md 用哪个 scope 更新
- SKILLS 用哪个 scope 更新
- cross-domain 按哪个 task bucket 聚合

例子：

- `weather_aligned_short + forecast`
- 对应 `scope_key = weather_aligned_short_forecast`

金融 aligned 的 `indicator` 会先展开，所以 scope 会变成：

- `finance_aligned_short_indicator_macd`
- `finance_aligned_short_indicator_upper_bb`

### 1.5 `sample commit`

这是 memory 侧的原子提交边界。

一条 sample 跑完以后，会发生：

1. 先把这次 sample 的 note 写入。
2. 然后在同一次 commit 里检查有没有跨过蒸馏阈值。
3. 如果跨过，就立刻做 Notes -> MEMORY，以及后面的 fan-out。

所以这里不是“先记下来，以后再慢慢蒸馏”，而是：

- 当前 sample commit
- 就可能同步触发 distillation

### 1.6 `slot_id`

这是训练时主 agent 给 sub-agent 分配的候选分支槽位。

新定义说明：

- `slot_id` 不是 memory id，也不是规则 id。
- `slot_id` 只是一次训练运行里，sub-agent 候选分支的编号。
- 每个 `slot_id` 对应一组“可见工具子集”和对应的 keep probability。

### 1.7 `candidate trace`

这是训练结束后，从 `execution_log` 里整理出的候选分支轨迹。

它不是单独落盘的独立文件，而是运行中间结构，后面会被用于：

- learning analysis
- Notes 的 `Candidates` 段
- MEMORY 蒸馏

---

## 2. 训练入口到底在哪里

训练 CLI 入口是：

```bash
python -m timeclaw.benchmark.run_learning --train
```

调用链是：

1. `timeclaw/benchmark/run_learning.py:cmd_train()`
2. `timeclaw/benchmark/train_loop.py:run_train_loop()`
3. 每个 work item 调 `PipelineRunner.run(..., mode="train")`

如果你从代码里往里追，真正的训练核心不是 `benchmark` 评测入口，而是：

- `run_learning.py` 负责 CLI
- `train_loop.py` 负责并发调度
- `pipeline/runner.py` 负责单条运行
- `memory/hierarchical.py` 负责 notes/memory/tools/skills 落盘和蒸馏

训练完成后还会额外写：

- `.timeclaw/train_results/train_summary.json`

但更重要的是边跑边写的 per-task checkpoint：

- `.timeclaw/train_results/train_<dataset>__<task>.json`

---

## 3. 当前训练到底会跑多少任务

### 3.1 当前真正参与 train/eval 的数据集

当前 registry 里有 8 个数据集，但真正参与 train/eval 的，是 `tasks` 非空的 6 个：

- `weather_aligned_short`
- `weather_aligned_long`
- `finance_aligned_short`
- `finance_aligned_long`
- `finance_qa_short`
- `finance_qa_long`

`weather_qa_short` 和 `weather_qa_long` 目前 registry 里 task 是空的，所以不会进当前训练主线。

### 3.2 当前展开后的 scope task 数

当前代码期望的展开后总 task 数是 18。

原因如下：

| dataset | registry task | 展开后 task 数 |
| --- | --- | --- |
| `weather_aligned_short` | `forecast`, `trend`, `trend_past` | 3 |
| `weather_aligned_long` | `forecast`, `trend`, `trend_past` | 3 |
| `finance_aligned_short` | `forecast`, `trend`, `indicator` | 4 |
| `finance_aligned_long` | `forecast`, `trend`, `indicator` | 4 |
| `finance_qa_short` | `mcqa`, `correlation` | 2 |
| `finance_qa_long` | `mcqa`, `correlation` | 2 |

合计：

- `3 + 3 + 4 + 4 + 2 + 2 = 18`

其中最容易忽略的一点是：

- finance aligned 的 `indicator`
- 会先展开成 `indicator_macd` 和 `indicator_upper_bb`

### 3.3 当前官方 train split 的样本数

我用本地 `load_split(ds, "train")` 实际读了一次，当前 train split 样本数是：

| dataset | train split 样本数 | 每条样本会扩多少 task | 理论 work item 数 |
| --- | ---: | ---: | ---: |
| `weather_aligned_short` | 1445 | 3 | 4335 |
| `weather_aligned_long` | 1448 | 3 | 4344 |
| `finance_aligned_short` | 529 | 4 | 2116 |
| `finance_aligned_long` | 384 | 4 | 1536 |
| `finance_qa_short` | 341 | 2 | 682 |
| `finance_qa_long` | 368 | 2 | 736 |

总理论 work item 数：

- `13749`

注意这只是“理论上会排到的 `(sample, task)` 数”。

实际还会再扣掉：

- 没有 GT 的 task
- 被 checkpoint 跳过的 `(idx, task)`
- 被 `max_samples` / `sample_indices` 限制掉的样本

### 3.4 当前 benchmark 训练有没有图片任务

没有。

这是一个很重要的事实：

- 当前 `MTBenchSample` loader 只加载时序、文本、问答、labels
- 没有任何一个 benchmark loader 会给 `images`

所以当前官方 benchmark 训练主线实际上是：

- `timeseries`
- 或 `timeseries + text`

不是：

- `timeseries + image`
- 也不是 `timeseries + text + image`

图片链路是代码支持的能力，不是当前 benchmark 训练的真实主线。后面我会单独讲。

---

## 4. `run_train_loop()` 在做什么

`run_train_loop()` 是训练阶段真正的调度核心。

### 4.1 它先做的准备

1. 读取配置：`TimeClawConfig.load()`
2. 强制关闭 MCP：`cfg.mcp.enabled = False`
3. 校验展开后的 benchmark task 数是不是 18
4. 创建共享的 `PipelineRunner`
5. 读取 sample index 过滤器
6. 确定要跑哪些 dataset
7. 读取已有 checkpoint，跳过已完成 `(idx, task)`

### 4.2 它的 work item 粒度

work item 通常是：

- 1 个 dataset
- 1 个 sample index
- 1 个 task

也就是：

```python
pipeline.run(..., mode="train")
```

对应一次训练运行。

### 4.3 它怎么并发

这里有两层并发控制：

1. 全局并发：`asyncio.Semaphore(concurrency)`
2. 每个 `(dataset, task)` 的局部并发：`task_sems_global[(ds_name, task)]`

其中：

- 普通 task 用 `per_task_concurrency`
- indicator task 可以单独用 `indicator_concurrency`

所以当前训练入口确实支持并发，而且不是临时脚本加的，而是正式写在 `train_loop.py` 里的。

### 4.4 单个 work item 的硬超时

当前代码里，单个 `(sample, task)` 的训练硬超时是：

```python
asyncio.wait_for(pipeline.run(...), timeout=1200)
```

也就是：

- 1200 秒
- 20 分钟

注意这 20 分钟包住的是整次 `pipeline.run()`，不是只包 executor。

也就是说下面这些时间都算在 20 分钟里：

- preprocess
- 主 agent / sub-agent 执行
- learning analysis
- note 写入
- 可能被同步触发的 MEMORY / TOOLS / SKILLS / cross-domain distillation

---

## 5. 一条真实 sample 在训练前会准备什么

这里我选一条真实跑过的 sample，后面所有例子尽量都用它。

### 5.1 这条 sample 的身份

- dataset: `weather_aligned_short`
- file: `USW00003017_0.json`
- sample index: `0`
- task: `forecast`
- `pipeline_task`: `forecast`
- `scope_key`: `weather_aligned_short_forecast`
- `trace_id`: `de00e62dc456`
- `sample_ref`: `weather_aligned_short:USW00003017_0.json:0:forecast`

对应文件：

- trace: `.timeclaw/traces/de00e62dc456.jsonl`
- note: `.timeclaw/hierarchical_memory/notes/weather_aligned_short_forecast.md`

### 5.2 训练前准备的输入

在 `_process_one_task_sample()` 里，这条 sample 会先准备：

#### A. Ground truth

`forecast` 任务的 GT 是：

- `sample.output_window.tolist()`

#### B. `description`

由：

```python
build_task_description(sample, task)
```

生成。

这一步会把 benchmark 的 system prompt 和 user prompt 拼成一段文本。

这段文本后面会同时影响：

- 主 agent 的任务说明
- sub-agent 的任务说明
- learning analysis 的任务上下文
- Notes 里的 `Task Prompt`

#### C. `text_data`

如果 sample 自带文本，就包装成：

```json
[
  {
    "content": "<sample.text>",
    "source": "weather",
    "timestamp": ""
  }
]
```

#### D. `horizon`

如果是 `forecast`：

- `horizon = len(sample.output_window)`

这条样本是 24 步预测，所以：

- `horizon = 24`

#### E. `scope_keys`

通过：

```python
scope_keys_for_dataset_task(ds_name, task)
```

生成。

这条样本结果是：

```json
["weather_aligned_short_forecast"]
```

#### F. `source_metadata`

这是后面 note 持久化“来源”的基础。

这条 sample 的来源信息是：

```json
{
  "dataset_name": "weather_aligned_short",
  "sample_ref": "weather_aligned_short:USW00003017_0.json:0:forecast",
  "source_file": "USW00003017_0.json",
  "sample_idx": 0,
  "task_name": "forecast",
  "pipeline_task": "forecast",
  "domain": "weather"
}
```

---

## 6. `PipelineRunner.run(mode="train")` 里到底发生了什么

这一节只看 1 次单条训练运行。

### 6.1 Step 0: 建 trace 与 envelope

一进入 `run()`，先做这些事：

1. `ensure_tools()`
2. 生成 `trace_id`
3. `trace.start_run(...)`
4. 构建 `TaskEnvelope`

trace 一开始就会落盘：

- `.timeclaw/traces/<trace_id>.jsonl`

这条样本的 run_start 记录里已经带上了：

- `task_type=forecast`
- `domain=weather`
- `horizon=24`
- `mode=train`
- `scope_keys=["weather_aligned_short_forecast"]`

### 6.2 Step 1: Preprocess

这是纯代码阶段，没有 LLM。

当前固定跑的预处理工具是：

- `basic_stats`
- `autocorrelation`
- `stationarity_test`
- `detect_trend`
- `detect_anomaly`

如果输入含图片，还会额外跑：

- `image_load`
- `image_feature`

如果输入含文本，目前不会额外跑 text profiling，因为：

- `_TEXT_PROFILING_TOOLS = []`

这一阶段产出 3 类东西：

1. `fingerprint`
2. `modalities`
3. `profiling_results`

这条真实 sample 的模态检测结果是：

- `["timeseries", "text"]`

### 6.3 Step 2: Memory Injection

如果 memory 没禁用，会在 executor 开始前注入：

1. `soul_prompt`
2. `memory_l2`
3. `tool_notes`
4. `skill_prompt`

同时还会写：

- `memory_l1 = ""`

这说明当前实现里：

- Notes 不再直接注入 prompt
- 真正注入的是 `SOUL + MEMORY + TOOLS + SKILLS`

### 6.4 Step 3: Training mode 注入

如果 `mode == "train"`，还会额外注入：

- `_mode = "train"`
- `_ground_truth`
- `_tool_dropout_scope`
- `_tool_dropout_counts`

这一步的作用主要有 2 个：

1. 允许 agent 在训练时使用 `evaluate_against_gt`
2. 让 sub-agent exploration 可以读取当前 scope 的 tool usage 统计，做 tool dropout

### 6.5 Step 4: 主 Executor 执行

当前架构名义上是单 Executor，但训练时它可以：

- `spawn_subagent`

所以真实运行形态更准确地说是：

- 1 个主 executor
- 0 到多条 sub-agent candidate 分支

主执行阶段最终会产出：

- `raw_answer`
- `answer_type`
- `confidence`
- `reasoning`
- `execution_log`

训练时如果顶层输出是 `learning_summary`，后面还会继续：

- 抽 candidate trace
- 做 learning analysis
- 写 note
- 触发蒸馏

---

## 7. 主 agent 实际拿到的 prompt 是怎么拼的

这一节是你特别关心的部分。

### 7.1 主 agent 的 system prompt 怎么拼

主 agent 的 system prompt 是：

```text
REACT_SYSTEM_PROMPT
+ SOUL.md
+ MEMORY（如果有）
+ visible tool guidance（训练模式会包含 spawn / evaluate 规则）
```

这条真实 sample 的主 agent system prompt 已经持久化在 trace 里：

- trace: `.timeclaw/traces/de00e62dc456.jsonl`
- content block ref: `prompt_message/system:41ee49c6f3`
- 长度：5461 chars

它的开头就是：

```text
You are TimeClaw, a multimodal time-series analysis system.
You solve tasks step-by-step by selecting and orchestrating tools.

## How you work
Each turn, you see the results of all previous steps. You then decide:
- Run ONE more tool ...
- Declare "done" ...
...

# SOUL — TimeClaw Agent Identity
...

## Spawn Guidance
- If you think this step would benefit from exploring multiple follow-up branches for comparison, prefer using `spawn_subagent`.
...

## Evaluate Guidance
- `evaluate_*` is for post-hoc validation ...
...
```

### 7.2 主 agent 的 user prompt 怎么拼

主 agent 的 first-turn user prompt 是：

```text
Task Prompt
+ Execution Context
+ Output Contract
+ Sample Fingerprint
+ Profiling Results
+ Support
+ Available Tools
```

对应代码模板是：

- `timeclaw/agents/prompts/executor_prompt.py:REACT_FIRST_TURN`

这条真实 sample 的完整 user prompt 也已经持久化在 trace 里：

- trace: `.timeclaw/traces/de00e62dc456.jsonl`
- content block ref: `prompt_message/user:a4af500b4c`
- 长度：70731 chars

这里很重要：

- 这是拼接完后的真实 prompt
- 不是模板
- 里面已经塞进了 benchmark 任务原文、完整天气新闻、fingerprint、profiling、tool list

由于这个 user prompt 太长，我不在这里机械贴 7 万字；但它的结构就是：

```text
## Task Prompt
<benchmark 拼出来的完整任务文本，含整段天气新闻>

## Execution Context
<训练时的 slot 信息 / 分支执行上下文>

## Output Contract
<训练模式要求最终 answer_type 必须是 learning_summary 等>

## Sample Fingerprint
<fingerprint JSON>

## Profiling Results
<basic_stats / autocorrelation / stationarity_test / detect_trend / detect_anomaly>

## Support
<tool notes + skills + reanalysis 信息；如果有的话>

## Available Tools
<当前候选工具完整列表和参数约束>
```

也就是说：

- 主 agent 真正看到的是一个非常长的完整 benchmark 任务说明
- 不是压缩版 note
- 不是只看最近样本摘要

### 7.3 这条 sample 的主 agent 第一步怎么想

这条 sample 的主 agent 第一步没有直接 forecast，而是先判断：

- 训练模式要求至少比较 3 条 candidate path
- 所以应该先 `spawn_subagent`

trace 里保存下来的真实 JSON 如下：

```json
{
  "thought": "Reason: Training-mode instructions require comparing at least 3 candidate paths via spawn_subagent before finishing, and the task is multimodal with plausible alternative forecasting strategies (seasonal ETS, ARIMA, and direct LLM multimodal reasoning using flood narrative). Action: spawn three distinct sub-agents on different slots to produce candidate 24-step forecasts using diverse methods.",
  "done": false,
  "tool": "spawn_subagent",
  "args": {
    "tasks": [
      {
        "task": "Produce a 24-step hourly temperature forecast candidate from the given 168-point series and flood-related news, emphasizing seasonal structure and recent cooling; prefer detect_seasonality, stl_decompose, ets_forecast, then sanity-check.",
        "slot_id": 0,
        "tools_hint": ["detect_seasonality", "stl_decompose", "ets_forecast", "consistency_check"]
      },
      {
        "task": "Produce a 24-step hourly temperature forecast candidate from the given 168-point series and news, emphasizing trend-stationary behavior and autoregression; prefer detect_changepoint or window_stats, then arima_forecast, then sanity-check.",
        "slot_id": 1,
        "tools_hint": ["window_stats", "detect_changepoint", "arima_forecast", "consistency_check"]
      },
      {
        "task": "Produce a 24-step hourly temperature forecast candidate using multimodal reasoning over the hourly series plus flood/heavy-rain narrative, emphasizing storm-induced cooling and diurnal cycle persistence; prefer summarize or text_to_event plus llm_reasoning_forecast.",
        "slot_id": 2,
        "tools_hint": ["summarize", "text_to_event", "llm_reasoning_forecast", "bound_check"]
      }
    ]
  }
}
```

这说明主 agent 在训练时的第一层“思考”非常明确：

1. 先满足训练合约。
2. 先拉出 3 条候选分支。
3. 再用 GT 评估来选 winner。

---

## 8. tool-dropout 到底送了什么进 sub-agent

这部分也很关键，因为“主 agent 提交的任务”和“sub-agent 最终看到的 hint”不是一回事。

### 8.1 代码里的真实流程

主 agent 调 `spawn_subagent` 后，框架会做 3 件事：

1. 先按当前 scope 预计算 `subagent_slots`
2. 每个 `slot_id` 对应一组 `visible_candidates` 和 `keep_probabilities`
3. 再根据 slot 的可见工具，把原始 `tools_hint` 缩成一个更短、更硬的 advisory path

这里的新定义说明：

- `raw_tools_hint` 是主 agent 原始提交的 hint。
- `tools_hint` 是框架在 `_normalize_spawn_tasks()` 之后真正送进 sub-agent 的 hint。
- 也就是说，主 agent 说“prefer 一组工具”，不代表 sub-agent 最终仍会拿到整组。

### 8.2 这条真实 sample 的 tool-dropout 有没有真的 drop 工具

就这条 sample 来说：

- 没有真正 drop 掉工具

trace 里 `subagent_tool_dropout` 记录显示：

- `visible_candidates` 长度是 61
- `dropped_candidates` 长度是 0
- `keep_probabilities` 基本都是 1.0

也就是说，这次训练里：

- tool-dropout 的机制确实跑了
- 但这条 sample 的当前统计状态下，没有实际裁掉工具

这是一个非常重要的事实：

- 当前这条真实样本里的 sub-agent 差异
- 主要不是来自“真正删掉了不同工具”
- 而是来自后面的 `tools_hint` 归一化和 branch goal 差异

### 8.3 这条 sample 最终送进 3 个 sub-agent 的 hint

主 agent 原始提交的是多工具 hint，但框架归一化后，trace 里 `subagent_start` 显示真正送进去的是：

- sub-agent 0: `tools_hint = ["slice"]`
- sub-agent 1: `tools_hint = ["value_at"]`
- sub-agent 2: `tools_hint = ["basic_stats"]`

注意这里最反直觉的一点：

- 第 3 条 branch 原始想走的是 `summarize / text_to_event / llm_reasoning_forecast`
- 但最后归一化后，真正被强约束成了 `basic_stats`

也就是说：

- 训练里“多分支探索”的真实差异
- 不是简单照抄主 agent 的原始 hint
- 而是框架基于 slot 可见工具，再次收窄出来的

### 8.4 sub-agent 实际拿到的 first-turn prompt 长什么样

sub-agent 不会重新拿一个完全不同的 benchmark 任务。

它拿到的是：

- 同一个大任务背景
- 再加一个更强的 `Execution Context`

这条真实 sample 里，sub-agent 1 的 Execution Context 片段如下：

```text
## Execution Context
Branch Goal:
Produce a 24-step hourly temperature forecast candidate from the given 168-point series and news, emphasizing trend-stationary behavior and autoregression; prefer detect_changepoint or window_stats, then arima_forecast, then sanity-check.

Assigned Exploration Path:
- Prioritize this path: value_at.
- Before returning `done=true`, you must have actually called at least one tool from this path.

## Output Contract
- Return ONLY a JSON numeric array with at least 24 numbers.
- Example: [1.234, 1.567, 1.890]
- Do NOT return prose like `Predicted Temperatures: ...`.
```

这说明 sub-agent 真正被约束的东西有 3 层：

1. 同一个原始任务背景
2. 一个 branch-specific goal
3. 一个更硬的 `Assigned Exploration Path`

---

## 9. 这 3 个 sub-agent 实际做了什么

这条真实 sample 最终 3 个 sub-agent 都成功了。

### 9.1 sub-agent 0

工具链：

- `slice -> detect_seasonality -> ets_forecast`

它做的事：

1. 先切最近 72 小时
2. 检查季节性
3. 再用 ETS 预测

最后产出一个 24 步 forecast 候选。

### 9.2 sub-agent 1

工具链：

- `value_at -> window_stats -> arima_forecast`

它做的事：

1. 读取最后一个观测值
2. 统计最近 24 小时窗口
3. 再用 ARIMA 做自回归预测

### 9.3 sub-agent 2

工具链：

- `basic_stats`

它做的事：

1. 只跑了 `basic_stats`
2. 然后结合任务背景和天气文本，直接给出一个 storm-aware 的 diurnal forecast 候选

这一点非常关键：

- sub-agent 不一定要跑很多工具
- 只要满足自己的 path 约束、产出 task-valid answer，就可以结束

### 9.4 三条分支的真实结果概览

主 agent 后面收到的 sub-agent 汇总，trace 里长这样：

```text
Sub-agent results (3 tasks):

Sub-agent 0 (slot_id=0): SUCCESS — tool_chain=slice -> detect_seasonality -> ets_forecast
Sub-agent 1 (slot_id=1): SUCCESS — tool_chain=value_at -> window_stats -> arima_forecast
Sub-agent 2 (slot_id=2): SUCCESS — tool_chain=basic_stats
```

所以这条 sample 的 candidate family 很清楚：

1. recent-window seasonal path
2. endpoint + ARIMA path
3. stats-anchored storm-aware path

---

## 10. 主 agent 后面怎么做比较和收尾

### 10.1 先跑 GT 评估

主 agent 在拿到 3 条 candidate 后，没有直接总结，而是调用：

- `evaluate_batch_against_gt`

这是训练模式的强约束之一：

- 至少要有一轮 GT 评估

这条 sample 的评估结果是：

```json
{
  "results": [
    {
      "name": "slot_0",
      "metrics": {"mae": 2.644, "rmse": 3.767, "mape": 15.069, "mse": 14.193}
    },
    {
      "name": "slot_1",
      "metrics": {"mae": 3.887, "rmse": 4.260, "mape": 28.222, "mse": 18.149}
    },
    {
      "name": "slot_2",
      "metrics": {"mae": 1.977, "rmse": 2.627, "mape": 12.194, "mse": 6.900}
    }
  ]
}
```

可以直接看出：

- `slot_2` 全指标最好

### 10.2 再输出 `learning_summary`

主 agent 收到 GT 评估后，会用训练模式专用的 answer type 收尾：

- `answer_type = "learning_summary"`

这条 sample 最终学习结论的核心是：

- winner 是 `slot_2`
- 因为它最符合“早段持续偏冷、后段有恢复、但不完全回到常态”的真实路径

### 10.3 顶层 reasoning 还会被自我纠正

这是训练 trace 里一个很重要、但平时很容易漏掉的细节。

主 agent 第一次收尾后，框架又发了一个 self-correction：

```text
Self-correction: your final `reasoning` must explain substantive candidate approaches, not orchestration tools. Rewrite it as numbered steps comparing these candidate paths where relevant: slice, value_at, basic_stats. State which candidate patterns were considered, what decisive evidence separated them, and why the winner was selected. Do not mention spawn_subagent or evaluate_* in the reasoning.
```

这说明训练阶段的主 agent 最终 reasoning 不是“模型第一次写什么就是什么”，而是会被框架再修一次格式和内容边界。

---

## 11. learning outcome 保存前，还会做一次 single-sample learning analysis

主 agent 结束后，`PipelineRunner._save_learning_outcome()` 不会立刻写 note。

它会先做这些事：

1. 从 `execution_log` 提取 `candidate_traces`
2. 收集所有 GT evaluation metrics
3. 选出 `best_metrics`
4. 调 `generate_learning_analysis(...)`

这一步会再发一次单独的 LLM 调用。

### 11.1 这一步 LLM 要产出什么

固定是 5 组字段：

1. `text_signals`
2. `sample_conditions`
3. `winner_evidence`
4. `failure_signals`
5. `reusable_rule`

### 11.2 这一步的输入是什么

它看到的内容包括：

- `scope`
- `task_type`
- `domain`
- `task_description`
- `Text/Context Source`
- `sample_fingerprint`
- `candidate_traces`
- `best_metrics`
- `fallback_summary`

这条 sample 的 learning analysis prompt 也被持久化到了 trace 里：

- trace: `.timeclaw/traces/de00e62dc456.jsonl`
- system ref: `prompt_message/system:27b3e96bc5`
- user ref: `prompt_message/user:a46d893cb6`

### 11.3 这一步的输出怎么用

它会产出两份东西：

1. `structured_analysis`
2. `rendered summary`

其中：

- `structured_analysis` 用于 note 结构化存储
- `rendered summary` 会写到 note 的 `Analysis` 段

---

## 12. 训练阶段最终会落什么产物

这一节是“产物地图”。

### 12.1 `train_results`

路径：

- `.timeclaw/train_results/train_<dataset>__<task>.json`

这是训练过程中的 per-task checkpoint。

这条 sample 在 `train_weather_aligned_short__forecast.json` 里的结果行，核心字段像这样：

```json
{
  "idx": 0,
  "dataset_name": "weather_aligned_short",
  "sample_ref": "weather_aligned_short:USW00003017_0.json:0:forecast",
  "task": "forecast",
  "pipeline_task": "forecast",
  "scope_keys": ["weather_aligned_short_forecast"],
  "status": "completed",
  "answer_type": "learning_summary",
  "has_prediction": true,
  "trace_id": "de00e62dc456",
  "gate_accepted": true,
  "tool_chain": ["spawn_subagent", "evaluate_batch_against_gt"],
  "total_ms": 605516
}
```

这里要特别注意：

- 顶层 `tool_chain` 只有主 agent 层的框架动作
- sub-agent 真正跑过的候选工具链，要去 trace 或 note 的 `Candidates` 看

### 12.2 trace

路径：

- `.timeclaw/traces/de00e62dc456.jsonl`

trace 会记录：

- run_start / run_end
- preprocess tool_exec
- prompt content blocks
- llm_call
- agent_step
- tool_exec
- decision
- stage

所以如果你要还原一条训练的真实过程，最有用的通常就是：

1. `train_results` 先定位 `trace_id`
2. 再看 `.timeclaw/traces/<trace_id>.jsonl`

### 12.3 Notes

路径：

- `.timeclaw/hierarchical_memory/notes/weather_aligned_short_forecast.md`

当前 note 是按 scope shard 的 append-only 文件。

一条 note 的结构现在是：

```text
### [timestamp] <scope> — <status>
[<scope>] <status>

Scope:
...

Source:
...

Task Prompt:
...

Numeric Time Series:
...

Text Signals:
...

Winner Tools:
...

Prediction:
...

Metrics:
...

Fingerprint:
...

Approaches:
...

Candidates:
...

Analysis:
...
```

这条真实 note 的 `Source` 段如下：

```text
Source:
- Dataset: weather_aligned_short
- Sample Ref: weather_aligned_short:USW00003017_0.json:0:forecast
- Source File: USW00003017_0.json
- Sample Index: 0
- Task: forecast
- Trace ID: de00e62dc456
- Trace Path: .timeclaw/traces/de00e62dc456.jsonl
```

所以现在 note 的来源是清楚的，而且已经持久化了。

### 12.4 MEMORY

路径：

- `.timeclaw/hierarchical_memory/memory/weather_aligned_short_forecast.json`

当前 MEMORY 不是一整篇 markdown，而是：

- 每个 scope 一个 JSON 数组
- 每条规则一个结构化 rule object

一条真实 rule 长这样：

```json
{
  "scope": "weather_aligned_short_forecast",
  "rule_id": "weather_aligned_short_forecast::004",
  "kind": "tool_failure",
  "summary": "Do not rely on basic statistics alone for next-day hourly temperature paths when diurnal shape matters.",
  "conditions": [
    "24-step hourly temperature forecast",
    "visible intraday cycle",
    "recent weather event affects local level"
  ],
  "preferred_tools": [],
  "avoid_tools": ["basic_stats"],
  "rationale": "Across several runs, summary-statistics paths were consistently worse than methods that preserved hourly phase and recent shape. They tended to oversmooth or miss the local weather-driven trajectory.",
  "evidence": {
    "support_count": 15,
    "sample_refs": ["00e0f3ec5972ee7c", "..."]
  },
  "confidence": 0.9499999993048331,
  "injectable": true,
  "status": "active"
}
```

当前允许的 `kind` 只有：

- `tool_selection`
- `tool_failure`
- `domain_heuristic`
- `result_constraint`

### 12.5 TOOLS.md

路径：

- `.timeclaw/hierarchical_memory/TOOLS.md`

它不是传统 markdown，而是一个 JSON 文件，顶层主要有：

- `notes_by_scope`
- `_scope_counts`

这条 scope 的真实 tool card 片段是：

```json
{
  "basic_stats": {
    "when_it_helps": [
      "Use when the latest day is unusually temperature-suppressed after a storm, so recent level matters more than a normal diurnal pattern."
    ],
    "when_to_avoid": [
      "Avoid when a visible intraday day-night cycle is present in the recent history."
    ],
    "output_risks": [
      "Can flatten or miss day-night structure and degrade a 24-hour temperature path when intraday cycling is visible."
    ]
  },
  "value_at": {
    "when_it_helps": [
      "Use when a severe storm or hail event lands very near the history endpoint, making the endpoint level especially informative."
    ],
    "when_to_avoid": [
      "Avoid when the history shows a clear day-night cycle rather than a one-off endpoint effect.",
      "Avoid when text implies a storm or rain-to-snow transition into a colder regime."
    ],
    "output_risks": []
  }
}
```

### 12.6 SKILLS

路径：

- `.timeclaw/hierarchical_memory/skills/weather_aligned_short_forecast.md`

当前 skill 是 per-scope markdown SOP。

真实内容长这样：

```md
### Skill 1
Condition:
Recent history keeps a clear day/night cycle, and storm/rain/hail/cold-front context suggests the latest 24–48 hours are more representative than older history.

Why:
The strongest rules favor preserving hourly diurnal shape and adjusting level or amplitude for weather effects.

Workflow:
1. Inspect the last 24–48 hours ...
2. Build the next path from that recent daily pattern ...
3. Apply a restrained weather adjustment ...
4. Check continuity ...

Guardrails:
- Avoid `basic_stats` when intraday cycle matters ...
- Return exactly 24 temperatures ...
```

所以你可以把当前 memory 链理解成：

- Notes 负责保存单样本经验
- MEMORY 负责存规则
- TOOLS 负责存单工具边界
- SKILLS 负责存“怎么做”的 SOP

---

## 13. Notes -> MEMORY -> TOOLS / SKILLS / cross_domain 是怎么触发的

### 13.1 当前阈值

当前阈值写死在 `timeclaw/memory/hierarchical.py` 里：

- Notes -> MEMORY：每个 scope 满 5 条 note
- Notes -> SKILLS：每个 scope 满 10 条 note
- Notes -> TOOLS：每个 scope 满 10 条 note
- MEMORY -> cross_domain：每个 task bucket 满 20 条 note

### 13.2 当前 commit 顺序

当前 `HierarchicalMemory.record()` 的 commit 顺序是：

1. 对每个 scope 先写 Notes
2. 然后 `distiller.on_commit(committed_scopes)`

`on_commit()` 内部顺序又是：

1. 先串行做当前 scope 的 Notes -> MEMORY
2. 再并行 fan-out：
   - MEMORY -> SKILLS
   - Notes -> TOOLS
   - MEMORY -> cross_domain

所以现在的蒸馏不是后台异步慢慢跑，而是：

- 当前 sample commit
- 就可能同步触发

### 13.3 Notes -> MEMORY 的 prompt 重点

当前 Notes -> MEMORY 的 LLM prompt 明确要求：

1. 优先支持已有 rule，不要乱造新 rule
2. 先比对同 kind 的现有 rule 再决定是不是新 rule
3. 对 `result_constraint` / `domain_heuristic` 做更强的近义去重
4. 只允许 4 种 kind
5. 不允许写 training / orchestration / evaluate_* / spawn_subagent 规则

另外，代码里还加了本地近义去重保险：

- `_is_near_duplicate_rule()`
- `_find_near_duplicate_rule_id()`

也就是说，现在不是只靠 LLM 自觉 dedupe。

如果 LLM 还想新加一条非常近义的 `result_constraint` / `domain_heuristic`，本地逻辑会再尝试把它并回已有 rule。

---

## 14. 图片在训练阶段是怎么处理的

这一节专门回答你要的“简单图片怎么处理”。

但先说结论：

- 当前 benchmark 训练主线没有图片
- 下面这节讲的是代码支持的图片路径，不是当前官方 train split 真实主线

### 14.1 为什么说当前训练主线没有图片

因为：

1. `MTBenchSample` loader 只加载时序 / 文本 / labels / QA 字段
2. 当前 benchmark loader 没有任何 `images` 字段注入
3. 所以 `train_loop -> pipeline.run(mode="train")` 目前不会从 benchmark 样本里带图片进去

### 14.2 但代码确实支持图片

如果你手动给 `PipelineRunner.run(..., images=[...])` 传图片，流程是：

1. `ModalityDetector` 检测到 `image`
2. preprocess 阶段额外跑：
   - `image_load`
   - `image_feature`
3. executor 在候选工具里只暴露“所需模态已满足”的 image tool
4. 后续工具执行时，`images` 会被直接传给工具

当前可用的典型 image tool 包括：

- `image_load`
- `image_feature`
- `image_classify`
- `chart_to_data`
- `image_ocr`
- `temporal_align_image`

### 14.3 一张真实本地图片的简单例子

我直接用仓库里的这张图做了一个真实例子：

- `data/time_mmd/DescriptionOfOT.png`

文件信息：

```text
PNG image data, 816 x 322, 8-bit/color RGBA, non-interlaced
```

#### A. 先做 modality detect

输入：

```json
[
  {
    "path": "data/time_mmd/DescriptionOfOT.png",
    "timestamp": "demo"
  }
]
```

输出：

```json
{
  "modalities": ["image"],
  "n_modalities": 1,
  "is_multimodal": false,
  "details": {
    "image": {
      "n_images": 1,
      "paths": ["data/time_mmd/DescriptionOfOT.png"],
      "has_timestamps": true
    }
  }
}
```

#### B. 再跑 `image_load`

真实输出：

```json
{
  "width": 816,
  "height": 322,
  "mode": "RGBA",
  "channels": 4,
  "dtype": "uint8",
  "pixel_mean": 236.71289181433443,
  "pixel_std": 54.57925941775903,
  "pixel_min": 0,
  "pixel_max": 255
}
```

### 14.4 图片信息会怎么进入 agent

这里有个容易误解的地方：

- raw image 像素不会直接塞进 prompt

当前真实代码路径是：

1. 图片本体在 `EnvironmentSnapshot.images`
2. 预处理得到的 `image_load` / `image_feature` 结果写进 `profiling_results`
3. executor prompt 里看到的是：
   - 模态报告
   - profiling 结果摘要
4. 真正需要看图时，agent 通过 image tool 调 multimodal LLM

也就是说：

- 图片不是“先 OCR 成大段文本再全塞 prompt”
- 而是“prompt 只知道这里有图，以及图的 profiling 概况；需要深入看图时再调工具”

---

## 15. 这一整条训练链路，按时间顺序重新串一遍

如果你只想记住主线，可以记下面这个版本。

### 15.1 入口

```text
python -m timeclaw.benchmark.run_learning --train
  -> run_train_loop()
```

### 15.2 调度

```text
load_split(train)
  -> 样本按 dataset/task 展开
  -> 并发调度每个 (sample, task)
```

### 15.3 单条运行

```text
pipeline.run(mode="train")
  -> Preprocess
  -> Memory Injection
  -> Training-mode 注入 GT / dropout scope
  -> 主 executor
  -> 可能 spawn_subagent
  -> GT evaluation
  -> learning_summary
  -> extract candidate traces
  -> generate learning analysis
  -> Notes.add()
  -> Distiller.on_commit()
  -> Post-process
  -> 返回结果
```

### 15.4 产物

```text
.timeclaw/train_results/train_<dataset>__<task>.json
.timeclaw/traces/<trace_id>.jsonl
.timeclaw/hierarchical_memory/notes/<scope>.md
.timeclaw/hierarchical_memory/memory/<scope>.json
.timeclaw/hierarchical_memory/TOOLS.md
.timeclaw/hierarchical_memory/skills/<scope>.md
```

---

## 16. 这份文档里最值得记住的几个结论

### 16.1 训练的“顶层 tool_chain”不等于真正候选方法

顶层 `train_result.tool_chain` 经常只有：

- `spawn_subagent`
- `evaluate_batch_against_gt`

真正候选方法要看：

- sub-agent execution log
- Notes 的 `Candidates`
- trace 的 `Sub-agent results`

### 16.2 当前 Notes 已经持久化了来源

现在 note 里会明确写：

- dataset
- sample_ref
- source_file
- sample_idx
- task
- trace_id
- trace_path

所以现在可以从：

- MEMORY rule
- 追到 note
- 再追到具体 trace

### 16.3 当前 benchmark 训练主线没有图片

图片代码是支持的，但当前官方 train split 的真实训练主线不是图像训练。

所以如果以后你看到 image tool 的逻辑，不要误以为它已经在当前 MTBench train loop 里大规模生效了。

### 16.4 这条真实 sample 里 tool-dropout 机制“跑了，但没真正 drop”

这是很重要的真实结论。

就 `de00e62dc456` 这条训练 trace 来说：

- subagent slot 机制存在
- keep probability 也算了
- 但是实际 `dropped_candidates = 0`

所以这次多分支的差异主要来自：

- branch goal
- `tools_hint` 归一化

不是来自真正的工具删减。

### 16.5 learning analysis 是单独的一次 LLM 调用

它不是主 agent 的最终回答直接原样塞进 note。

现在的流程是：

- 主 agent 先生成 `learning_summary`
- 框架再单独调用 `generate_learning_analysis`
- 最后把结构化分析写进 Notes

所以如果你要 debug note 质量，不能只看主 agent 最终 summary，还得看：

- learning analysis 这一步有没有跑好

---

## 17. 如果你要自己复盘一条训练样本，最推荐的顺序

建议按这个顺序看：

1. 先看 `train_results` 里的那一行，拿到 `trace_id`
2. 打开 `.timeclaw/traces/<trace_id>.jsonl`
3. 先看主 agent prompt 和 `spawn_subagent` payload
4. 再看 sub-agent start / tool result / candidate outputs
5. 再看 `evaluate_batch_against_gt`
6. 再看最终 `learning_summary`
7. 再去 note / memory / tools / skills 看它最后沉淀成了什么

对本文这条样本，就是：

- train result 行：`.timeclaw/train_results/train_weather_aligned_short__forecast.json`
- trace：`.timeclaw/traces/de00e62dc456.jsonl`
- note：`.timeclaw/hierarchical_memory/notes/weather_aligned_short_forecast.md`
- memory：`.timeclaw/hierarchical_memory/memory/weather_aligned_short_forecast.json`
- tools：`.timeclaw/hierarchical_memory/TOOLS.md`
- skill：`.timeclaw/hierarchical_memory/skills/weather_aligned_short_forecast.md`

这条链已经足够把一条训练样本从输入追到最终蒸馏产物。
