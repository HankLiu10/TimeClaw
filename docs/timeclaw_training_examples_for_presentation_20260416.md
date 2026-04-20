# TimeClaw 展示用实例文档

更新时间：2026-04-16

说明：

- 这份文档用于汇报展示，不是从某一条真实 trace 文件原样拷贝，而是**严格按照当前仓库协议组织的中文展示版代表性实例**。
- 我保留了真实系统里的关键字段名、`tool name`、`answer_type`、`learning_summary` 结构、`MEMORY` 字段结构、`SKILLS` 模板结构。
- 为了便于展示，我把自然语言内容改成中文，但没有改动关键协议形状。
- 你可以把它理解为：**一组高度贴近当前 TimeClaw 实现的“论文/汇报级别原始样例”**。

---

## 1. 主 Agent 的完整交互记录实例

场景设定：

- 任务类型：`forecast`
- scope：`finance_short_horizon_forecast`
- 目标：根据给定股票价格序列，预测未来 `6` 个时间步
- 当前阶段：训练阶段
- 因此主 agent 不能直接输出 forecast 数组，必须先：
  1. `spawn_subagent`
  2. 得到至少 2 条 task-valid candidate
  3. `evaluate_batch_against_gt`
  4. 最后输出 `learning_summary`

下面从 `spawn_subagent` 开始展示。

---

### 1.1 主 Agent：当前任务背景

```text
任务描述：
给定某支金融时间序列最近 48 个观测点，预测未来 6 个时间步的数值。
输出必须是一个 JSON 数值数组。

样本指纹：
- 序列长度：48
- 缺失率：0.0
- 最近局部趋势：平稳后小幅上行
- 短窗口波动：中等
- 周期性：弱
- 异常尖峰：无明显尖峰

当前可用训练期机制：
- 可以使用 spawn_subagent
- 可以使用 evaluate_batch_against_gt
- 最终必须输出 answer_type="learning_summary"
```

---

### 1.2 主 Agent，第 1 个动作：发起 `spawn_subagent`

#### assistant -> tool call

```json
{
  "thought": "Reason: 当前样本适合显式比较不同 forecasting 路径，且训练阶段需要至少两条真实 task-valid candidate。Action: 调用 spawn_subagent，分别探索 Chronos-2、ETS 和 ARIMA 三条路径。",
  "tool": "spawn_subagent",
  "args": {
    "tasks": [
      {
        "slot_id": 0,
        "task": "使用 Chronos-2 路径生成未来 6 步预测，重点检查短期延续性是否足够。",
        "tools_hint": [
          "chronos2_forecast"
        ],
        "branch_metadata": {
          "branch_name": "子agent1",
          "branch_family": "foundation_forecast"
        }
      },
      {
        "slot_id": 1,
        "task": "使用 ETS 路径生成未来 6 步预测，重点利用局部平稳和短期平滑趋势。",
        "tools_hint": [
          "ets_forecast"
        ],
        "branch_metadata": {
          "branch_name": "子agent2",
          "branch_family": "statistical_forecast"
        }
      },
      {
        "slot_id": 2,
        "task": "使用 ARIMA 路径生成未来 6 步预测，检查该路径是否会过度平滑。",
        "tools_hint": [
          "arima_forecast"
        ],
        "branch_metadata": {
          "branch_name": "子agent3",
          "branch_family": "autoregressive_forecast"
        }
      }
    ]
  }
}
```

---

### 1.3 `spawn_subagent` 返回结果

这里按你的要求，子 agent 的部分先简写为 `子agent1/2/3`。

#### tool -> assistant

```json
{
  "n_tasks": 3,
  "n_success": 3,
  "results": [
    {
      "sub_task": 0,
      "slot_id": 0,
      "branch_metadata": {
        "branch_name": "子agent1",
        "branch_family": "foundation_forecast"
      },
      "success": true,
      "result": {
        "slot_id": 0,
        "candidate_name": "chronos2_forecast",
        "sub_task_description": "使用 Chronos-2 路径生成未来 6 步预测，重点检查短期延续性是否足够。",
        "tools_hint": [
          "chronos2_forecast"
        ],
        "tool_chain": [
          "chronos2_forecast"
        ],
        "answer": [
          102.4,
          103.1,
          103.7,
          104.2,
          104.6,
          105.0
        ],
        "answer_type": "forecast",
        "reasoning": "1. 使用 chronos2_forecast 直接生成未来 6 步序列。2. 该路径保持了近期上行趋势，但斜率略高。",
        "execution_context": {
          "tool_chain": [
            "chronos2_forecast"
          ],
          "public_reasoning": [
            "chronos2_forecast 生成了一条延续近期上行的未来路径。",
            "该预测相对更激进，后半段抬升较快。"
          ],
          "evidence_summary": "Chronos-2 给出了平滑上行预测，但后半段斜率偏高。"
        }
      }
    },
    {
      "sub_task": 1,
      "slot_id": 1,
      "branch_metadata": {
        "branch_name": "子agent2",
        "branch_family": "statistical_forecast"
      },
      "success": true,
      "result": {
        "slot_id": 1,
        "candidate_name": "ets_forecast",
        "sub_task_description": "使用 ETS 路径生成未来 6 步预测，重点利用局部平稳和短期平滑趋势。",
        "tools_hint": [
          "ets_forecast"
        ],
        "tool_chain": [
          "ets_forecast"
        ],
        "answer": [
          101.8,
          102.0,
          102.3,
          102.5,
          102.8,
          103.0
        ],
        "answer_type": "forecast",
        "reasoning": "1. 使用 ets_forecast 建模近期局部水平与缓慢上行趋势。2. 输出路径更保守，也更贴近当前样本的短期平稳结构。",
        "execution_context": {
          "tool_chain": [
            "ets_forecast"
          ],
          "public_reasoning": [
            "ets_forecast 保留了近期的平稳后缓升形态。",
            "该路径没有把尾部抬升得过快，更接近短期局部结构。"
          ],
          "evidence_summary": "ETS 路径较保守，和当前样本的局部平稳后缓升模式更一致。"
        }
      }
    },
    {
      "sub_task": 2,
      "slot_id": 2,
      "branch_metadata": {
        "branch_name": "子agent3",
        "branch_family": "autoregressive_forecast"
      },
      "success": true,
      "result": {
        "slot_id": 2,
        "candidate_name": "arima_forecast",
        "sub_task_description": "使用 ARIMA 路径生成未来 6 步预测，检查该路径是否会过度平滑。",
        "tools_hint": [
          "arima_forecast"
        ],
        "tool_chain": [
          "arima_forecast"
        ],
        "answer": [
          101.6,
          101.7,
          101.8,
          101.8,
          101.9,
          101.9
        ],
        "answer_type": "forecast",
        "reasoning": "1. 使用 arima_forecast 生成未来 6 步预测。2. 该路径输出较平，可能低估近期缓升趋势。",
        "execution_context": {
          "tool_chain": [
            "arima_forecast"
          ],
          "public_reasoning": [
            "arima_forecast 给出的未来路径接近平坦。",
            "该路径可能把近期的缓慢上行压得过低。"
          ],
          "evidence_summary": "ARIMA 路径较平，可能低估短期缓升。"
        }
      }
    }
  ],
  "observation": "Sub-agent results (3 tasks):\n  子agent1: SUCCESS — tool_chain=chronos2_forecast; answer=[102.4,103.1,103.7,104.2,104.6,105.0]\n  子agent2: SUCCESS — tool_chain=ets_forecast; answer=[101.8,102.0,102.3,102.5,102.8,103.0]\n  子agent3: SUCCESS — tool_chain=arima_forecast; answer=[101.6,101.7,101.8,101.8,101.9,101.9]"
}
```

---

### 1.4 主 Agent，第 2 个动作：调用 `evaluate_batch_against_gt`

#### assistant -> tool call

```json
{
  "thought": "Reason: 已有三条真实 task-valid candidate，需要在同一任务上做 GT 对比，判断哪条路径在连续数值指标上更优。Action: 调用 evaluate_batch_against_gt，对三条 forecast 路径进行批量评估。",
  "tool": "evaluate_batch_against_gt",
  "args": {
    "task_type": "forecast",
    "predictions": [
      [
        102.4,
        103.1,
        103.7,
        104.2,
        104.6,
        105.0
      ],
      [
        101.8,
        102.0,
        102.3,
        102.5,
        102.8,
        103.0
      ],
      [
        101.6,
        101.7,
        101.8,
        101.8,
        101.9,
        101.9
      ]
    ]
  }
}
```

---

### 1.5 `evaluate_batch_against_gt` 返回结果

#### tool -> assistant

```json
{
  "task_type": "forecast",
  "results": [
    {
      "candidate_index": 0,
      "metrics": {
        "mae": 2.410,
        "rmse": 2.882,
        "mse": 8.305
      },
      "correct": null
    },
    {
      "candidate_index": 1,
      "metrics": {
        "mae": 1.120,
        "rmse": 1.392,
        "mse": 1.938
      },
      "correct": null
    },
    {
      "candidate_index": 2,
      "metrics": {
        "mae": 2.980,
        "rmse": 3.344,
        "mse": 11.184
      },
      "correct": null
    }
  ],
  "winner_index": 1,
  "winner_metrics": {
    "mae": 1.120,
    "rmse": 1.392,
    "mse": 1.938
  },
  "summary": "第 2 条候选路径（ETS）在 MAE / RMSE 上最优。Chronos-2 路径有过冲倾向，ARIMA 路径则明显偏平。"
}
```

---

### 1.6 主 Agent，最终汇总：输出 `learning_summary`

#### assistant -> final answer

```json
{
  "thought": "Reason: 已完成多路径探索与 GT 对比，ETS 路径在当前样本上以更低 MAE 和 RMSE 明确胜出。Action: 输出 learning_summary，总结胜出路径和可复用经验。",
  "done": true,
  "answer": {
    "insight": "本样本中 ets_forecast 路径胜出，因为它保留了局部平稳后缓升结构，在 MAE=1.120、RMSE=1.392 上优于 chronos2_forecast 的过冲路径和 arima_forecast 的偏平路径。",
    "recommendation": "当短期序列呈现局部平稳后缓慢上行、且没有明显尖峰时，优先尝试 ets_forecast 这类保守平滑路径，并警惕过度抬升或过度压平。"
  },
  "answer_type": "learning_summary",
  "confidence": 0.93,
  "reasoning": "1. 对同一样本分别执行 chronos2_forecast、ets_forecast、arima_forecast 三条候选路径。2. 三条路径都产生了 task-valid 的未来 6 步数值数组，因此比较重点不再是能否完成任务，而是哪条路径更优。3. evaluate_batch_against_gt 显示 ets_forecast 的 MAE=1.120、RMSE=1.392，为三条路径中最优。4. chronos2_forecast 虽然保持上行，但尾部抬升偏快，形成过冲；arima_forecast 输出较平，低估了短期缓升。5. 因此应选择 ets_forecast 作为当前样本的优胜路径，并把“局部平稳后缓升时优先保守平滑路径”的经验写入后续外部产物。",
  "execution_context": {
    "tool_chain": [
      "chronos2_forecast",
      "ets_forecast",
      "arima_forecast"
    ],
    "public_reasoning": [
      "chronos2_forecast 生成了延续上行的路径，但尾部抬升偏快。",
      "ets_forecast 保留了局部平稳后缓慢上行的短期结构。",
      "arima_forecast 输出较平，可能压低近期缓升幅度。",
      "在批量 GT 评估中，ets_forecast 的 MAE 和 RMSE 最低。"
    ],
    "evidence_summary": "ETS 路径在连续误差指标上最优，且其保守平滑的形态最贴近当前样本的局部结构。"
  },
  "control_context": {
    "completion_notes": ""
  }
}
```

---

## 2. 子 Agent 的完整交互记录实例

这里展示 `子agent2` 的完整上下文。  
这个例子对应上面主 agent 中最终胜出的 `ets_forecast` 分支。

说明：

- 子 agent 不能再 `spawn_subagent`
- 子 agent 不能调用 `evaluate_*`
- 子 agent 必须先至少调用一次被提示路径里的工具
- 最终答案必须是**某个真实工具已经产出的 task-valid 输出**

---

### 2.1 子agent2：system message

```text
你是一个 TimeClaw 子代理。

要求：
- 你最多有 4 个反应步。
- 你必须基于真实工具结果完成任务。
- 你不能使用 spawn_subagent。
- 你不能使用 evaluate_against_gt / evaluate_batch_against_gt / batch_evaluate_against_gt。
- 你的最终答案必须与本分支中某个真实工具已经产生的 task-valid 输出完全一致。
- 如果被分配了探索路径提示，则在 done=true 之前必须至少调用一次该路径中的工具。

当前可见工具：
- ets_forecast
- arima_forecast
- basic_stats
```

---

### 2.2 子agent2：first user message（完整上下文）

```text
Task Prompt:
- 你将看到一段金融时间序列。
- 你的任务是预测未来 6 个时间步的数值。
- 只允许输出 JSON 数值数组。

Task Boundary:
- 核心任务：根据提供的序列预测未来数值序列。
- task-valid candidate output：一个覆盖未来 6 步的 JSON 数值数组。
- 不合法输出：trend label、indicator JSON、QA 风格文本。

Fingerprint:
{
  "length": 48,
  "missing_ratio": 0.0,
  "recent_trend": "flat_to_slight_up",
  "volatility": "medium",
  "seasonality": "weak",
  "spike": "none"
}

Profiling:
- stationarity: 局部更接近平稳
- local_pattern: 近期先平后缓慢上行
- anomaly_check: 无明显异常尖峰

Execution Context:

Control Context:
Assigned Exploration Path:
- Prioritize this path: ets_forecast.
- Before returning done=true, you must have actually called at least one tool from this path.

Branch Goal:
使用 ETS 路径生成未来 6 步预测，重点利用局部平稳和短期平滑趋势。

Output Contract:
- Do not invent new task content in the final step.
- The final task answer must exactly match a task-valid output already produced by a real tool in this branch.
- If no tool has produced a task-valid answer yet, call another tool instead of finishing.
- Return ONLY a JSON numeric array with at least 6 numbers.
- Example: [1.234, 1.567, 1.890]
- Do NOT return prose like `Predicted Temperatures: ...`.

Support:
Tool Notes:
- ets_forecast:
  - when_it_helps: 近期序列较平稳，只有缓慢趋势变化时更稳健
  - when_to_avoid: 短期内存在明显尖峰或突变时可能过度平滑
  - output_risks: 容易压低突然上冲的尾部

## MEMORY — Learned Insights
### finance_short_horizon_forecast Specific
- Rule: 当短期序列局部平稳、仅有轻微上行时，保守平滑路径通常优于更激进的 foundation forecast。 | Prefer: ets_forecast | Avoid: chronos2_forecast | Why: 激进路径在短窗口里更容易过冲。 | Evidence: 4 note(s)
- Rule: 当未来变化幅度不大时，过于平坦的 ARIMA 路径可能低估缓慢上行。 | Prefer: ets_forecast | Avoid: arima_forecast | Why: ARIMA 在该类样本上容易压平尾部。 | Evidence: 3 note(s)

### Skill 1
Condition:
近期窗口整体平稳，仅在尾部出现轻微抬升，没有明显尖峰。

Why:
这类样本更适合保守平滑路径；过激预测更容易把短期尾部抬得过高。

Workflow:
1. 先调用 ets_forecast 形成基线预测。
2. 检查输出是否保留“平稳后缓升”的形态，而不是突然陡升。
3. 若输出形态合理，则直接采用该工具结果作为候选答案。

Guardrails:
- 如果结果尾部突然陡升，说明该路径不稳定，应谨慎接受。

Decision:
本轮请输出一个动作：要么调用一个工具，要么在已有 task-valid 输出后直接完成。

Visible Tools:
ets_forecast, arima_forecast, basic_stats
```

---

### 2.3 子agent2，第 1 轮：调用工具

#### assistant -> tool call

```json
{
  "thought": "Reason: 当前样本的局部平稳后缓升结构和 ets_forecast 的适用边界一致，而且分配路径要求优先执行 ets_forecast。Action: 调用 ets_forecast 生成未来 6 步预测。",
  "tool": "ets_forecast",
  "args": {
    "horizon": 6
  }
}
```

#### tool -> assistant

```json
{
  "forecast": [
    101.8,
    102.0,
    102.3,
    102.5,
    102.8,
    103.0
  ],
  "model_summary": {
    "level": "stable",
    "trend": "slight_up"
  }
}
```

---

### 2.4 子agent2，第 2 轮：输出最终答案

#### user -> assistant（follow-up）

```text
Decision State:
- 当前分支已经有真实工具输出。
- 如果该输出是 task-valid candidate，可以直接完成。
- 最终答案必须与某个真实工具输出完全一致。
```

#### assistant -> final answer

```json
{
  "thought": "Reason: ets_forecast 已经生成了合法的未来 6 步数值数组，而且该路径与当前样本的局部结构一致。Action: 直接返回该工具输出作为本分支最终答案。",
  "done": true,
  "answer": [
    101.8,
    102.0,
    102.3,
    102.5,
    102.8,
    103.0
  ],
  "answer_type": "forecast",
  "confidence": 0.86,
  "reasoning": "1. 先执行 ets_forecast 得到未来 6 步数值预测。2. 该结果保留了局部平稳后缓升的形态，没有出现突兀过冲。3. 因为输出已经满足 forecast 任务的合法格式，所以直接返回该工具结果。",
  "execution_context": {
    "tool_chain": [
      "ets_forecast"
    ],
    "public_reasoning": [
      "ets_forecast 产出了合法的未来 6 步数值数组。",
      "该输出保持了平稳后缓升的局部形态。"
    ],
    "evidence_summary": "ETS 输出既合法又与当前样本结构一致。"
  },
  "control_context": {
    "completion_notes": ""
  }
}
```

---

## 3. 学习到的 Notes、Memory、Skills 实例

这里给出几条可以直接展示的原始实例。

---

### 3.1 Notes 实例 1

说明：

- `Notes` 是追加写入的原始经验块
- 它是 distillation source
- 它不是最终推理时直接注入的主 memory

```md
### [finance_short_horizon_forecast] SCORED

Scope:
finance_short_horizon_forecast

Source:
- Dataset: TimeClaw-Train-Finance
- Sample Ref: fin_forecast_00417
- Source File: data/train/finance_short_horizon_forecast.jsonl
- Sample Index: 417
- Task: forecast
- Train Run ID: train_20260416_a
- Trace ID: 8b72f3d1a9c24ef0
- Trace Path: .timeclaw/traces/8b72f3d1a9c24ef0.jsonl

Task Prompt:
[TASK_PROMPT_BEGIN]
- 给定最近 48 个观测点，预测未来 6 个时间步。
- 输出必须是 JSON 数值数组。
[TASK_PROMPT_END]

Numeric Time Series:
[NUMERIC_TIME_SERIES_BEGIN]
Numeric time series:
[98.2, 98.4, 98.1, 98.5, 98.7, 98.8, 99.0, 98.9, 99.1, 99.0, 99.2, 99.3, 99.4, 99.3, 99.5, 99.7, 99.8, 99.9, 100.0, 100.1, 100.0, 100.2, 100.3, 100.2, 100.4, 100.5, 100.6, 100.5, 100.7, 100.8, 100.9, 100.9, 101.0, 101.1, 101.1, 101.2, 101.3, 101.4, 101.4, 101.5, 101.5, 101.6, 101.6, 101.7, 101.7, 101.8, 101.8, 101.9]
[NUMERIC_TIME_SERIES_END]

Winner Tools:
- ets_forecast

Prediction:
[101.8, 102.0, 102.3, 102.5, 102.8, 103.0]

Metrics:
- mae: 1.120
- rmse: 1.392
- mse: 1.938

Fingerprint:
{"length":48,"missing_ratio":0.0,"recent_trend":"flat_to_slight_up","volatility":"medium","seasonality":"weak","spike":"none"}

Approaches:
- ets_forecast(MAE=1.120, RMSE=1.392)
- chronos2_forecast(MAE=2.410, RMSE=2.882)
- arima_forecast(MAE=2.980, RMSE=3.344)

Candidates:
[CANDIDATES_BEGIN]
- Candidate: ets_forecast (MAE=1.120, RMSE=1.392)
  Tools: ets_forecast
  Reasoning: Ets_forecast 保留了局部平稳后缓升的形态。该路径没有把尾部抬得过快。

- Candidate: chronos2_forecast (MAE=2.410, RMSE=2.882)
  Tools: chronos2_forecast
  Reasoning: Chronos2_forecast 生成了延续上行的路径。该预测后半段抬升较快。
  Weakness: This path recorded MAE=2.410, RMSE=2.882 on the sample.

- Candidate: arima_forecast (MAE=2.980, RMSE=3.344)
  Tools: arima_forecast
  Reasoning: Arima_forecast 给出的未来路径接近平坦。该路径可能把近期缓慢上行压得过低。
  Weakness: This path recorded MAE=2.980, RMSE=3.344 on the sample.
[CANDIDATES_END]

Analysis:
[ANALYSIS_BEGIN]
ETS 路径在当前样本上胜出。关键区别不在于三条路径能否产生合法 forecast，而在于哪条路径更贴近“局部平稳后缓升”的短期结构。Chronos-2 过冲，ARIMA 偏平，ETS 最稳。
[ANALYSIS_END]
```

---

### 3.2 Notes 实例 2

```md
### [finance_short_horizon_forecast] SCORED

Scope:
finance_short_horizon_forecast

Source:
- Dataset: TimeClaw-Train-Finance
- Sample Ref: fin_forecast_00461
- Source File: data/train/finance_short_horizon_forecast.jsonl
- Sample Index: 461
- Task: forecast
- Train Run ID: train_20260416_a
- Trace ID: c14d22a1bb1f4c63
- Trace Path: .timeclaw/traces/c14d22a1bb1f4c63.jsonl

Task Prompt:
[TASK_PROMPT_BEGIN]
- 给定最近 48 个观测点，预测未来 6 个时间步。
- 输出必须是 JSON 数值数组。
[TASK_PROMPT_END]

Winner Tools:
- ets_forecast

Prediction:
[88.6, 88.7, 88.9, 89.0, 89.2, 89.4]

Metrics:
- mae: 0.940
- rmse: 1.210
- mse: 1.464

Fingerprint:
{"length":48,"missing_ratio":0.0,"recent_trend":"stable_then_slight_up","volatility":"low","seasonality":"weak","spike":"none"}

Approaches:
- ets_forecast(MAE=0.940, RMSE=1.210)
- arima_forecast(MAE=1.870, RMSE=2.144)
- chronos2_forecast(MAE=2.230, RMSE=2.591)

Analysis:
[ANALYSIS_BEGIN]
当序列低波动且尾部只有轻微抬升时，ETS 继续优于更激进或更平坦的路径。这个样本进一步支持“保守平滑优先”的局部规律。
[ANALYSIS_END]
```

---

### 3.3 Memory 实例

说明：

- 这里给的是 `MEMORY` 的原始结构化条目实例
- 这些条目是从多条 notes 蒸馏而来
- 与 notes 不同，`MEMORY` 是规则对象，不是自然语言流水账

```json
[
  {
    "scope": "finance_short_horizon_forecast",
    "rule_id": "finance_short_horizon_forecast::007",
    "kind": "tool_selection",
    "summary": "当短期序列局部平稳、仅有轻微上行时，优先使用 ets_forecast 这类保守平滑路径。",
    "conditions": [
      "recent_trend=flat_to_slight_up",
      "volatility=low_or_medium",
      "no_large_spike=true"
    ],
    "preferred_tools": [
      "ets_forecast"
    ],
    "avoid_tools": [
      "chronos2_forecast"
    ],
    "rationale": "在该类样本中，更激进的 foundation forecast 容易把尾部抬得过高，导致短期过冲。",
    "evidence": {
      "support_count": 4,
      "sample_refs": [
        "0f91c8e2b1d4497a",
        "8b72f3d1a9c24ef0",
        "c14d22a1bb1f4c63",
        "d3a2b7e8f9214a10"
      ]
    },
    "source_rule_ids": [],
    "confidence": 0.86,
    "injectable": true,
    "status": "active",
    "timestamp": 1776297601.0
  },
  {
    "scope": "finance_short_horizon_forecast",
    "rule_id": "finance_short_horizon_forecast::008",
    "kind": "tool_failure",
    "summary": "当样本只有缓慢上行而没有明显跳变时，arima_forecast 容易把尾部压得过平。",
    "conditions": [
      "recent_trend=slight_up",
      "local_shape=smooth",
      "volatility=low_or_medium"
    ],
    "preferred_tools": [
      "ets_forecast"
    ],
    "avoid_tools": [
      "arima_forecast"
    ],
    "rationale": "ARIMA 路径在这类短期样本上经常低估缓慢持续的抬升。",
    "evidence": {
      "support_count": 3,
      "sample_refs": [
        "8b72f3d1a9c24ef0",
        "c14d22a1bb1f4c63",
        "61ab2f39c4427a18"
      ]
    },
    "source_rule_ids": [],
    "confidence": 0.79,
    "injectable": true,
    "status": "active",
    "timestamp": 1776297602.0
  },
  {
    "scope": "finance_short_horizon_forecast",
    "rule_id": "finance_short_horizon_forecast::009",
    "kind": "result_constraint",
    "summary": "若预测尾部相对最近真实窗口突然变得明显更陡，应谨慎接受该 forecast，即使整体方向正确。",
    "conditions": [
      "forecast_tail_slope >> recent_local_slope",
      "short_horizon=true"
    ],
    "preferred_tools": [],
    "avoid_tools": [
      "chronos2_forecast"
    ],
    "rationale": "短窗口 forecasting 中，尾部斜率异常增大经常对应过冲，而不是更优的趋势捕捉。",
    "evidence": {
      "support_count": 2,
      "sample_refs": [
        "8b72f3d1a9c24ef0",
        "52f20e1a6d3a4b7c"
      ]
    },
    "source_rule_ids": [],
    "confidence": 0.68,
    "injectable": true,
    "status": "active",
    "timestamp": 1776297603.0
  },
  {
    "scope": "finance_short_horizon_forecast",
    "rule_id": "finance_short_horizon_forecast::010",
    "kind": "domain_heuristic",
    "summary": "当局部证据不足、样本纹理不清晰时，不要仅凭全局均值或极简统计直接决定未来路径。",
    "conditions": [
      "weak_local_signal=true"
    ],
    "preferred_tools": [
      "ets_forecast"
    ],
    "avoid_tools": [
      "basic_stats"
    ],
    "rationale": "basic_stats 只能提供全局摘要，难以表达短期 future-path estimate。",
    "evidence": {
      "support_count": 2,
      "sample_refs": [
        "92ae18b74c13019d",
        "71cb25e0af913c21"
      ]
    },
    "source_rule_ids": [],
    "confidence": 0.61,
    "injectable": false,
    "status": "active",
    "timestamp": 1776297604.0
  }
]
```

注意：

- 最后一条 `rule_id=010` 是 `injectable=false`
- 它表示这条规则会被保存在 memory 里，但当前不一定进入 runtime prompt
- 这正好可以用来展示你们“存储”和“注入”分离的机制

---

### 3.4 Skills 实例

说明：

- `SKILLS` 来自 `MEMORY -> SKILLS`
- 当前模板要求使用：
  - `### Skill N`
  - `Condition`
  - `Why`
  - `Workflow`
  - 可选 `Guardrails`

下面给出两条真实风格的展示版实例。

```md
### Skill 1
Condition:
近期窗口整体平稳，仅在尾部出现轻微抬升；波动不大，也没有明显尖峰。

Why:
这类样本通常不需要激进路径。更保守的平滑 forecasting 更容易贴住局部结构，而不会把短期尾部抬得过高。

Workflow:
1. 先调用 ets_forecast 生成未来 6 步基线预测。
2. 检查预测是否保持“平稳后缓升”的局部形态，而不是突然陡升。
3. 如果输出形态稳定且没有明显过冲，优先采用该结果作为候选答案。

Guardrails:
- 如果预测尾部相对最近真实窗口突然明显变陡，不要直接接受。
- 如果输出近乎完全平坦，也要警惕低估缓慢上行。

### Skill 2
Condition:
样本短期内存在缓慢上行，但上行幅度不大，局部纹理比全局趋势更重要。

Why:
在这种短窗口场景里，ARIMA 常把未来压得过平，而更激进的 foundation path 又可能过冲；需要优先保留局部平滑变化。

Workflow:
1. 优先选择能表达局部平滑变化的 forecasting 路径，不要先用 basic_stats 直接下结论。
2. 若可选路径中存在 ets_forecast，先执行它形成结构化候选。
3. 对比输出尾部斜率与最近观测窗口的局部斜率，确认没有出现明显过冲或压平。

Guardrails:
- 不要仅凭全局均值、最值或简单摘要直接给出最终 forecast。
```

---

## 4. 最适合你展示时强调的几点

如果你要在汇报里展示这份文档，我建议你重点强调下面 4 件事：

1. 主 agent 学到的不是“哪个答案能做出来”，而是“哪条路径更优”。
2. 子 agent 的探索不是随便试，而是有 `tools_hint`、可见工具约束、task-valid output contract 约束。
3. `Notes` 是原始经验层，`MEMORY` 是结构化规则层，`SKILLS` 是可执行工作流层。
4. `injectable=false` 这种条目很适合展示你们的 memory 不是 append-only，而是“保留 + 筛选 + 注入分离”。

