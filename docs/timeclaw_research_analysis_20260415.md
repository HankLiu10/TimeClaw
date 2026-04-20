# TimeClaw 研究分析与 Related Works 梳理

更新时间：2026-04-15

这份文档只基于两类信息：

1. 当前仓库里的真实实现与真实产物。
2. `related works/` 中的论文，以及我额外核对到的 2026-04-15 之前相关论文。

`related works/TimeClaw.pdf` 只作为旧思路参考，不作为本次结论依据。当前代码与真实产物优先。

---

## 0. 先给结论

### 0.1 我对 TimeClaw 当前学术价值的判断

如果要一句话概括 TimeClaw，现在最强、最稳的定位不是：

- 一个“更大的多模态时序 agent”
- 一个“又一个 tool-augmented TSQA 系统”
- 一个“原生训练出更强 TimeART 风格模型的方法”

而是：

**一个面向 time-series agent 的外部化学习框架：它把“训练期强制探索 + 候选比较 + 学习结论蒸馏 + 测试期回注”做成了闭环。**

更具体一点：

- 它不是只让 agent 会用工具。
- 它要让 agent **从做题过程中学会如何选工具、何时不该信某种工具、什么数据条件对应什么路径**。
- 这个“学会”的内容不主要写进模型权重，而是外部化为 `MEMORY / TOOLS / SKILLS / history / state` 等产物，再在测试期重新注入。

所以，TimeClaw 的核心价值更接近：

- **time-series agent learning**
- **externalized continual learning for numeric reasoning**
- **experience distillation for tool-using time-series agents**

而不是单纯的：

- forecasting system
- benchmark
- 多 agent 推理框架

### 0.2 与现有工作的最关键区别

和现有工作相比，TimeClaw 最像谁？答案不是 MTBench，而是 **TimeART + 一类带反思/协作的 agent 系统 + 外部 skill distillation** 的交叉体。

但它和这些工作的真正区别在于：

1. **MTBench** 主要是 benchmark，本身不提供 agent learning 闭环。
2. **TimeART** 重点是 tool-augmented TSRM 和 `TimeToolBench` 轨迹训练，学习更偏“教模型会用工具”。
3. **TimeClaw** 重点是：
   - 训练期强制分叉探索
   - 用 GT 比较候选路径
   - 把 winner/loser 的差异蒸馏成可复用知识
   - 测试期不改权重，直接把知识重新注入执行 prompt

所以，如果你论文要立核心贡献，我建议主轴放在：

- **agent 学习范式**
- **外部化经验蒸馏**
- **反工具先验的探索机制**

不要把主轴放在“多模态 benchmark SOTA”。

---

## 1. 当前项目的真实定位

### 1.1 新定义：什么叫“外部化学习”

这里的“外部化学习”指：

- agent 在训练期做探索；
- 学到的不是隐式藏在参数里的习惯；
- 而是被写成文件化知识产物；
- 测试期再把这些知识重新拼回 prompt。

在 TimeClaw 里，这条链路是：

`训练样本 -> 候选路径 -> GT 比较 -> learning analysis -> Notes -> MEMORY / TOOLS / SKILLS -> 测试期注入`

这和普通的：

- 监督微调
- trajectory imitation
- inference-time 多 agent 协作

都不是一回事。

### 1.2 当前真实 benchmark 范围

当前代码的 benchmark 主线来自 MTBench，但不是“全量 MTBench 原样直跑”。

真实实现可从 `timeclaw/benchmark/mtbench_adapter.py:28-100` 看到：

- 当前 registry 里真正参与 train/eval 的是 6 个数据集
- 展开后的总 task 数目标是 `18`
- 这条主线对齐的是 **TimeART 风格的 18-scope 子集**

其中一个很重要的事实是：

- 当前 benchmark 主线是 `timeseries`
- 或 `timeseries + text`
- **不是当前论文可以强主张的 image 主线**

这点也和你后面 claim 是否写“multimodal”直接相关。

### 1.3 当前真实运行架构

当前主线不是旧文档里常见的“四 agent 固定流水线”，而是：

- `PipelineRunner` 做 deterministic preprocess
- 一个 `ExecutorAgent` 做主执行
- 训练期可 `spawn_subagent`
- 结束后做 deterministic post-check

入口在 `timeclaw/pipeline/runner.py:1-24, 196-336`。

所以现在最准确的描述应是：

**single-agent mainline with train-time parallel candidate exploration**

而不是：

**always-on multi-agent pipeline**

---

## 2. 与现有工作的对比

## 2.1 先看最关键的两个：MTBench 和 TimeART

| 维度 | MTBench | TimeART | TimeClaw |
| --- | --- | --- | --- |
| 角色 | benchmark | agentic TSQA framework + tool-use training | 外部化学习框架 |
| 核心问题 | 怎么评估 TS + text reasoning | 怎么让 TSRM 学会 strategic tool-use | 怎么让 time-series agent 从执行中积累可复用经验 |
| 学习单位 | benchmark sample/task | trajectory/tool-use corpus (`TimeToolBench`) | candidate-path comparison + distilled knowledge artifact |
| 学习是否主要写进权重 | 是 benchmark，不涉及 | 是，偏训练 TSRM | 否，主要写入外部 memory artifacts |
| 是否显式比较多个候选路径 | benchmark 本身不做 | 可有 tool-use 轨迹，但重点不在外部 memory distill | 是，训练时强制分叉并比较 winner/loser |
| 测试期是否依赖历史经验回注 | 否 | 重点不在这里 | 是，`SOUL + MEMORY + TOOLS + SKILLS` 回注 |
| 当前与你项目的关系 | 数据与 prompt 来源 | 最接近的直接 related work | 你自己的方法主轴 |

### 2.1.1 TimeClaw vs MTBench

`MTBench` 是你现在最重要的数据与评测来源，但它不是你的方法学对手，而是你的**底座 benchmark**。

TimeClaw 对 MTBench 做了三层“再利用”：

1. 用官方 prompt 作为任务描述来源。
2. 用时间切分的官方 processed pool 生成 train/test sample list。
3. 把 MTBench 的 train split 从“训练集”改造成“agent 学习材料来源”。

关键代码：

- 官方 prompt bridge：`timeclaw/benchmark/mtbench_prompts.py:1-49`
- prompt 拼装入口：`timeclaw/benchmark/mtbench_support/task_descriptions.py:24-29`
- split 使用官方 processed data：`timeclaw/benchmark/dataset_split.py:1-12`

所以和 MTBench 的关系最好写成：

**TimeClaw is built on top of MTBench, but turns MTBench from a pure evaluation benchmark into an agent-learning environment.**

但有一个必须诚实写出来的点：

- 当前 `pipeline_eval.py` 对 forecast task 直接用 pipeline 输出；
- 对 non-forecast task，会再做一次 final reasoning call，把 tool analysis 追加回原始 prompt。

代码在 `timeclaw/benchmark/pipeline_eval.py:1-13, 551-760`。

因此：

**TimeClaw 当前测试协议不是“原封不动的 raw MTBench official evaluation”。**

这不是坏事，但论文里必须写清楚，否则 reviewer 很容易卡你。

### 2.1.2 TimeClaw vs TimeART

`TimeART` 是当前最接近、也是最需要认真区分的对比对象。

从 `related works/Wu 等 - 2026 - TimeART...pdf` 的摘要看，TimeART 主要强调：

- tool-augmentation
- fully agentic TSQA
- `TimeToolBench` 100k expert trajectories
- 四阶段训练去提升 TSRM 的 tool use 与泛化

TimeClaw 和它的关键差异有四个：

1. **学习载体不同**
   - TimeART：重点是轨迹数据和模型训练。
   - TimeClaw：重点是运行中产生的外部知识产物。

2. **学习闭环不同**
   - TimeART：更像“教会模型如何调用工具”。
   - TimeClaw：更像“让 agent 在做题中形成经验，并把经验回写为可检索规则”。

3. **探索机制不同**
   - TimeART 没有把“强制候选比较 + tool-dropout”做成核心学习机制。
   - TimeClaw 训练期明确要求至少多个 candidate path，并用 GT 比较。

4. **测试适配方式不同**
   - TimeART 更靠训练后的模型能力。
   - TimeClaw 更靠测试期 memory reinjection。

所以论文里最稳的表述是：

**TimeART teaches a time-series reasoning model to use tools; TimeClaw teaches an executing agent to learn reusable path-level experience from tool use.**

这句话基本抓住了两者最重要的分界。

---

## 2.2 其他 related works：按类别看

### 2.2.1 Benchmark / Evaluation 类

| 工作 | 它主要做什么 | 与 TimeClaw 的关系 |
| --- | --- | --- |
| [MTBench](https://arxiv.org/abs/2503.16858) | TS + text 的多模态时序 benchmark | TimeClaw 的主要 benchmark 底座 |
| [TemporalBench](https://arxiv.org/abs/2602.13272) | 面向 contextual / event-informed time-series agent 的 benchmark | 和 TimeClaw 目标接近，但更偏评测，而不是学习闭环 |
| [MMTS-BENCH](https://arxiv.org/abs/2602.08588) | 更广义的 time-series understanding/reasoning benchmark | 说明 benchmark 赛道在扩张，TimeClaw 不宜把“benchmark 新颖性”放太重 |
| [TFRBench](https://arxiv.org/abs/2604.05364) | 专门评估 forecasting system reasoning 的 benchmark | 它强调 reasoning evaluation；TimeClaw 更强调 agent learning |
| [TimeSeriesGym](https://arxiv.org/abs/2505.13291) | time-series ML engineering agents 的 benchmark | 更偏工程 agent benchmark，不是 TSQA/agent-learning distillation |
| [When LLM Meets Time Series / TSAIA Benchmark](https://arxiv.org/abs/2509.01822) | 建 benchmark 检验多步 time-series reasoning | 和 TimeClaw 一样关心 reasoning，但不等于外部化学习 |

### 2.2.2 Tool-Augmented / General TS Agent 类

| 工作 | 它主要做什么 | 与 TimeClaw 的区别 |
| --- | --- | --- |
| [TimeART](https://arxiv.org/abs/2601.13653) | tool-augmented TSRM + trajectory training | 最接近，但更偏模型训练，不是 memory-driven learning loop |
| [TS-Agent](https://arxiv.org/abs/2510.07432) | iterative statistical insight gathering | 也是逐步收集统计证据，但没有把经验蒸馏成长期 memory 主轴 |
| [TimeSeriesScientist](https://arxiv.org/abs/2510.01538) | general-purpose time-series AI agent | 更偏自动化数据科学家，不强在“从 benchmark train split 学习” |
| [Domain-Oriented Time Series Inference Agents](https://arxiv.org/abs/2410.04047) | 面向不同 domain 的时序推理 agent | 更强调 domain-oriented inference，不强调 learning artifacts |
| [TimeCopilot](https://arxiv.org/abs/2509.00616) | forecasting pipeline automation + TSFM/LLM 统一接口 | 更偏 AutoML/forecasting performance，而不是 agent learning |
| [Empowering Time Series Forecasting with LLM-Agents](https://arxiv.org/abs/2508.04231) | 用 agent 改善 forecasting workflow | 与 TimeClaw 共享“forecasting agent”方向，但学习闭环更弱 |

### 2.2.3 Multi-Agent Debate / Competition / Collaboration 类

| 工作 | 它主要做什么 | 与 TimeClaw 的区别 |
| --- | --- | --- |
| [TS-Debate](https://arxiv.org/abs/2601.19151) | 模态专家 debate + reviewer 校验 | 更强在 inference-time 协作，不是跨样本记忆学习 |
| [MAS4TS](https://arxiv.org/abs/2602.03026) | visual reasoning + multi-agent + tool routing | 更强在视觉与多 agent 组织，不强在 memory distillation |
| [Can Competition Enhance...](https://arxiv.org/abs/2504.10210) | 用 competition 提升 news-driven forecasting | 有“比较多个 agent/path”的味道，但没有形成可回注的长期 artifact |
| [FETA](https://arxiv.org/abs/2510.05950) | 训练自由的 TSC 多 agent in-context reasoning | 更偏 classification 与 exemplar retrieval，不是持续学习 |

### 2.2.4 Event / Retrieval / Multimodal Forecasting 类

| 工作 | 它主要做什么 | 与 TimeClaw 的区别 |
| --- | --- | --- |
| [From News to Forecast](https://arxiv.org/abs/2409.17515) | 新闻过滤、事件分析、reflection 增强 forecasting | 更偏“把事件接入预测”；TimeClaw 更偏“让 agent 学会怎么用这些线索” |
| [FLAIRR-TS](https://arxiv.org/abs/2508.19279) | iterative refinement + retrieval 的 test-time prompt optimization | 更像 test-time prompt search，不是长期 memory learning |
| [TimeCAP](https://arxiv.org/abs/2502.11418) | LLM 作为 contextualizer，建事件预测框架 | 强在 context augmentation，不强在 agent 学习闭环 |
| [TimeXL](https://arxiv.org/abs/2503.01013) | multimodal prediction + explainability + LLM-in-the-loop | 更偏模型/解释性系统，而不是 benchmark-driven agent learning |
| [Can Multimodal LLMs Perform Time Series Anomaly Detection?](https://arxiv.org/abs/2502.17812) | 关注多模态 LLM 做时序异常检测 | 属于任务扩展，不是通用 agent learning 框架 |

### 2.2.5 Annotation / Data Analysis / Distillation / Survey 类

| 工作 | 它主要做什么 | 与 TimeClaw 的关系 |
| --- | --- | --- |
| [TESSA / Decoding Time Series with LLMs](https://arxiv.org/abs/2410.17462) | 多 agent 生成通用与领域注释 | 和 TimeClaw 一样会产出中间知识，但目标是 annotation，不是执行经验 |
| [ASDA](https://arxiv.org/abs/2603.16112) | 自动 skill distillation and adaptation | 这是 TimeClaw 在“外部 skill distillation”方向最值得对照的非时序工作 |
| [LLM/Agent-as-Data-Analyst Survey](https://arxiv.org/abs/2509.23988) | 数据分析 agent 总综述 | 给 TimeClaw 的高层定位提供背景，但不是直接方法对手 |
| [Position: What Can LLMs Tell Us about Time Series Analysis](https://arxiv.org/abs/2402.02713) | 位置论文，讨论 LLM 与时间序列分析能力边界 | 更像问题定义层面的背景 |
| [DART](https://arxiv.org/abs/2506.11752) | 把显式 autoregressive reasoning 蒸馏到 silent thought | 不是时序 agent 工作，但对 TimeClaw 后续压缩 reasoning 成本很有启发 |

### 2.2.6 领域特化工作

| 工作 | 它主要做什么 | 与 TimeClaw 的区别 |
| --- | --- | --- |
| [ElliottAgents](https://arxiv.org/abs/2506.16813) | 金融技术分析的多 agent 股票预测 | 领域更窄，方法更像 domain recipe，而不是通用 learning framework |

---

## 2.3 对所有 `related works/` 文件夹内论文的总判断

如果把整个 `related works/` 文件夹看成一张图谱，TimeClaw 正好落在三条线的交叉处：

1. **MTBench / TemporalBench / TSAIA 这一类 benchmark 线**
2. **TimeART / TS-Agent / TimeSeriesScientist 这一类 tool-using time-series agent 线**
3. **ASDA 这一类 external skill distillation 线**

而你最该强调的是第三条线和前两条线的结合。

因为单独看：

- benchmark，别人也在做；
- tool-augmented TS agent，别人也在做；
- multi-agent debate / competition，别人也在做；

但把这些东西做成：

- **训练期强制探索**
- **有 GT 的候选比较**
- **scope 级外部蒸馏**
- **测试期回注**

这条闭环，在 time-series agent 方向上仍然很有新意。

---

## 3. TimeClaw 当前真实流程

## 3.1 总流程

当前主线入口是 `python -m timeclaw.benchmark.run_learning --all`，对应：

1. `split`
2. `train`
3. `distill`
4. `test`

代码在 `timeclaw/benchmark/run_learning.py:156-180`。

### 3.1.1 训练前：split

`dataset_split.py` 会对 **official MTBench processed data** 做时间切分，生成：

- `data/splits/<dataset>/train_files.json`
- `data/splits/<dataset>/test_files.json`

这一步默认不是独立语料，而是 official processed pool 的时间分割，见：

- `timeclaw/benchmark/dataset_split.py:1-12`

### 3.1.2 训练时：train

训练入口在：

- `timeclaw/benchmark/train_loop.py:1-18`

训练 loop 的真实含义是：

1. 取 MTBench train split 样本。
2. 构造 task description。
3. 调 `PipelineRunner.run(..., mode="train", ground_truth=gt)`。
4. 训练期允许 `spawn_subagent` 与 `evaluate_*`。
5. agent 产出 `learning_summary`。
6. 系统把 outcome 记入 hierarchical memory。

### 3.1.3 蒸馏时：distill

蒸馏由 `HierarchicalMemory` 负责。它的链路是：

`Notes -> MEMORY -> TOOLS / SKILLS / cross_domain -> history snapshot`

定义和链路在：

- `timeclaw/memory/hierarchical.py:1-45`
- `timeclaw/memory/hierarchical.py:2693-2816`

### 3.1.4 测试时：test

测试入口在：

- `timeclaw/benchmark/pipeline_eval.py`

关键区别：

- forecast 任务：直接用 pipeline prediction
- non-forecast 任务：先跑 pipeline 收集 tool analysis，再做 final reasoning LLM call

见：

- `timeclaw/benchmark/pipeline_eval.py:1-13`
- `timeclaw/benchmark/pipeline_eval.py:475-548`
- `timeclaw/benchmark/pipeline_eval.py:551-760`

---

## 3.2 单条样本在训练期的真实执行流

### 3.2.1 第一步：官方 prompt 进入系统

`build_task_description(sample, task)` 会把官方 prompt bridge 生成的 system/user prompt 合成一段 description：

- `timeclaw/benchmark/mtbench_support/task_descriptions.py:24-29`

而官方 bridge 本身来自：

- `timeclaw/benchmark/mtbench_prompts.py:24-49`

### 3.2.2 第二步：Pipeline preprocess

`PipelineRunner.run()` 先做：

- modality detection
- fingerprint extraction
- profiling tools

见：

- `timeclaw/pipeline/runner.py:257-275`

### 3.2.3 第三步：memory 注入

如果 memory 开启，会把这些内容写入 `envelope.task.metadata`：

- `soul_prompt`
- `memory_l2`
- `tool_notes`
- `skill_prompt`

见：

- `timeclaw/pipeline/runner.py:277-299`

### 3.2.4 第四步：训练期特殊注入

训练模式还会额外注入：

- `_mode = train`
- `_ground_truth`
- `_tool_dropout_scope`
- `_tool_dropout_counts`

见：

- `timeclaw/pipeline/runner.py:313-326`

### 3.2.5 第五步：Executor 执行

`ExecutorAgent.plan()` 会做三件关键事：

1. 构造可见工具列表。
2. 训练期加上 `spawn_subagent` 和 `evaluate_*`。
3. 拼系统 memory、tool notes、skills、profiling、subagent slots。

见：

- `timeclaw/agents/executor.py:648-795`

### 3.2.6 第六步：训练期分支探索

训练时，主 agent 不只是走单路径，而是会：

- 强制产生多个 candidate path
- 借助 `spawn_subagent`
- 用 `evaluate_*` 看 GT
- 最终输出 `answer_type="learning_summary"`

这是当前方法最关键的“学习相”。

### 3.2.7 第七步：learning summary 重写与标准化

如果顶层返回的是 `learning_summary`，runner 还会做一层后处理：

- 规范化 winner
- 用 candidate traces 重写 summary
- 对齐 execution-side evidence

见：

- `timeclaw/pipeline/runner.py:356-381`
- `timeclaw/benchmark/mtbench_support/learning_analysis.py:462-515`

### 3.2.8 第八步：记录 outcome 并触发蒸馏

最终 outcome 通过：

- `PipelineRunner.record_outcome()`

进入 hierarchical memory：

- `timeclaw/pipeline/runner.py:2663-2671`

随后 scope worker 会根据 watermark 触发：

- Notes -> MEMORY
- MEMORY -> SKILLS
- MEMORY -> TOOLS
- snapshot 写入 history

见：

- `timeclaw/memory/hierarchical.py:2693-2816`

---

## 4. 训练产物与目录结构

用户要求不展示：

- `notes`
- `memory`
- `skills`
- `TOOLS.md`

所以这里我只展示以下真实产物：

1. `train_run_manifest.json`
2. `train_<dataset>__<task>.json`
3. `traces/<trace_id>.jsonl`
4. `history/<scope>/<seq>.json`
5. `state/distill_state.json`

---

## 4.1 `.timeclaw/train_results/train_run_manifest.json`

### 4.1.1 新定义

`train_run_manifest.json` 是一次 train run 的总账本。

它记录：

- 本次 train run 的 id 和时长
- 每个 dataset/task 跑了多少条
- 成功/失败状态分布
- 对应结果文件在哪里

### 4.1.2 真实渲染结果

下面是从真实文件裁出来的结构：

```json
{
  "train_run_id": "20260415_183034_cfd23fa5",
  "elapsed_s": 3514.484,
  "dataset_summary": {
    "weather_aligned_short": {
      "total": 15,
      "summary": {
        "forecast": { "count": 5, "learning_summaries": 2 },
        "trend": { "count": 5, "learning_summaries": 2 },
        "trend_past": { "count": 5, "learning_summaries": 2 }
      }
    }
  },
  "tasks": {
    "weather_aligned_short::forecast": {
      "rows": 5,
      "status_counts": { "completed": 2, "timeout": 3 },
      "note_view_path": ".timeclaw/train_results/train_notes_weather_aligned_short__forecast.md",
      "result_path": ".timeclaw/train_results/train_weather_aligned_short__forecast.json"
    }
  }
}
```

它的用途很直接：

- 看这次 train run 覆盖了哪些 task
- 看哪些 task 卡在 timeout / runtime error
- 快速跳到 per-task 结果文件

---

## 4.2 `.timeclaw/train_results/train_<dataset>__<task>.json`

### 4.2.1 新定义

这个文件里的每一行代表：

**一条样本在某个 task 上的一次训练运行结果。**

它不是纯 summary，而是：

- 可以扫描的 preview
- 也保留了 debug 所需的 exact executor outputs

这点可以从：

- `timeclaw/benchmark/train_loop.py:1423-1441`
- `timeclaw/benchmark/train_loop.py:1201-1276`

看到。

### 4.2.2 真实渲染结果

下面是 `.timeclaw/train_results/train_weather_aligned_short__forecast.json` 第一行的精简版：

```json
{
  "idx": 0,
  "dataset": "weather_aligned_short",
  "sample_ref": "weather_aligned_short:USW00003017_0.json:0:forecast",
  "task": "forecast",
  "pipeline_task": "forecast",
  "scope_keys": ["weather_aligned_short_forecast"],
  "status": "completed",
  "answer_type": "learning_summary",
  "trace_id": "8d4cf69b69bd",
  "gate": "PASS",
  "tool_chain": ["timesfm2_forecast"],
  "winner_tool_chain": ["timesfm2_forecast"],
  "winner_candidate": "timesfm2_forecast",
  "winner_metrics": {
    "mae": 2.221,
    "rmse": 2.541,
    "mape": 14.77,
    "mse": 6.458
  },
  "execution_context": {
    "tool_chain": ["timesfm2_forecast", "aurora_forecast"],
    "public_reasoning": [
      "timesfm2_forecast produced a 24-step forecast...",
      "aurora_forecast produced a valid 24-step forecast...",
      "The trace reports lower error for timesfm2_forecast..."
    ]
  },
  "trace": {
    "trace_id": "8d4cf69b69bd",
    "trace_path": ".timeclaw/traces/8d4cf69b69bd.jsonl",
    "candidate_traces": [
      { "name": "timesfm2_forecast", "tool_chain": ["timesfm2_forecast"] },
      { "name": "aurora_forecast", "tool_chain": ["aurora_forecast"] }
    ],
    "winner_trace": {
      "name": "timesfm2_forecast",
      "tool_chain": ["timesfm2_forecast"]
    }
  }
}
```

这个文件非常重要，因为它已经把“训练学到了什么”压缩成了可分析的一行：

- 哪个 scope
- 哪个 winner
- winner 和 loser 分别是谁
- GT 指标差多少
- trace 在哪

这对论文画 case study 特别有用。

---

## 4.3 `.timeclaw/traces/<trace_id>.jsonl`

### 4.3.1 新定义

`trace.jsonl` 是完整事件流。

它记录的不是单个 summary，而是：

- prompt 内容块
- llm 调用
- agent step
- tool result
- run_end

### 4.3.2 真实渲染结果

下面是 `8d4cf69b69bd.jsonl` 的头部结构缩写：

```json
[
  {
    "event": "content_block",
    "payload": {
      "kind": "run_description",
      "preview": "User prompt: You are a weather forecasting AI...",
      "content": "User prompt: ... official MTBench weather forecast prompt ..."
    }
  },
  {
    "event": "agent_step",
    "payload": {
      "phase": "react_step_1",
      "summary": {
        "tool": "spawn_subagent"
      }
    }
  },
  {
    "event": "agent_step",
    "payload": {
      "phase": "subagent_tool_result",
      "summary": {
        "sub_agent_idx": 0,
        "tool": "timesfm2_forecast"
      }
    }
  }
]
```

它的作用是：

- `train_results` 不够时，去 trace 里还原完整过程
- 看 agent 实际见过什么 prompt
- 看 subagent 分支是怎么展开的

---

## 4.4 `.timeclaw/hierarchical_memory/history/<scope>/<seq>.json`

### 4.4.1 新定义

`history snapshot` 不是纯 MEMORY 文件。

它表示：

**某个 scope 在某次蒸馏完成后的“可复用学习状态快照”。**

当前定义见：

- `timeclaw/memory/hierarchical.py:2526-2616`

快照里包含：

- 这次覆盖了哪些 notes
- 当时的 entries / tool notes / skill / cross-domain entries
- 对应的 state snapshot

### 4.4.2 真实渲染结果

下面是 `.timeclaw/hierarchical_memory/history/weather_aligned_short_forecast/000001.json` 的精简版：

```json
{
  "scope": "weather_aligned_short_forecast",
  "snapshot_seq": 1,
  "source_stage": "notes_to_memory",
  "notes_from_seq": 1,
  "notes_to_seq": 41,
  "notes_covered_count": 41,
  "source_note_count": 41,
  "memory_fingerprint": "6437a360bb9c884f",
  "rule_count": 11,
  "tool_note_count": 6,
  "has_skill": true,
  "cross_domain_scope": "cross_domain__aligned_short_forecast",
  "cross_domain_rule_count": 6,
  "state_snapshot": {
    "notes_total_seq": 41,
    "notes_to_memory_done_seq": 41,
    "memory_to_skills_done_fingerprint": "6437a360bb9c884f",
    "memory_to_tools_done_fingerprint": "6437a360bb9c884f"
  }
}
```

这个文件特别适合做论文里的“learning over time”图，因为它天然就是快照序列。

---

## 4.5 `.timeclaw/hierarchical_memory/state/distill_state.json`

### 4.5.1 新定义

`distill_state.json` 是蒸馏系统的 watermark 和一致性状态表。

它不是知识本身，而是知识生产流水线的状态：

- notes 到哪里已经蒸馏过
- 当前 memory 指纹是什么
- skills/tools 是否和当前 memory 同步
- bucket 级 cross-domain 是否同步

定义见：

- `timeclaw/memory/hierarchical.py:2255-2367`

### 4.5.2 真实渲染结果

```json
{
  "scopes": {
    "weather_aligned_short_forecast": {
      "notes_total_seq": 161,
      "notes_to_memory_done_seq": 159,
      "memory_fingerprint": "10d98a32e8cb6869",
      "memory_history_seq": 37,
      "memory_to_skills_done_fingerprint": "10d98a32e8cb6869",
      "memory_to_tools_done_fingerprint": "10d98a32e8cb6869"
    }
  },
  "buckets": {
    "aligned_short_forecast": {
      "memory_fingerprint": "62281a96c92ed8a4",
      "memory_to_cross_done_fingerprint": "62281a96c92ed8a4"
    }
  }
}
```

如果你要向 reviewer 解释“这个系统不是随便写点 note，而是真的有状态管理和增量蒸馏”，这个文件就是直接证据。

---

## 5. Prompt 组装流程与信息来源

## 5.1 先定义：TimeClaw prompt 不是一段 prompt，而是分层拼出来的

当前系统真正送给 Executor 的 prompt，不是单模板，而是五类信息拼起来的：

1. benchmark task prompt
2. sample-derived information
3. learned memory
4. training/test mode metadata
5. runtime tool catalog

所以如果你要在论文里写 prompt 设计，最好写成 **assembly pipeline**，不要写成一个静态 prompt 模板。

---

## 5.2 第一层：benchmark task prompt

来源：

- `timeclaw/benchmark/mtbench_prompts.py`
- `timeclaw/benchmark/mtbench_support/task_descriptions.py`

流程：

1. `prompts.build_prompt(sample, task)` 调官方 bridge。
2. `build_task_description()` 把 system/user prompt 合成一段文字。

结果：

- Executor 看到的 `task_description`
- 本质上仍然来自 official MTBench prompt

这意味着你的 benchmark prompt 不是自己重写的，这对“可比性”是加分项。

---

## 5.3 第二层：sample-derived information

来源主要是 `PipelineRunner._preprocess()` 和 sample 本身。

进入 prompt 的信息包括：

- 原始 task description
- fingerprint
- profiling results
- modality report
- horizon / task boundary
- indicator output keys

入口可看：

- `timeclaw/pipeline/runner.py:257-275`
- `timeclaw/agents/executor.py:616-646`

这些信息的作用是：

- 不是告诉模型答案；
- 而是把样本的“结构先验”显式给出来。

例如：

- 有没有趋势
- 是否平稳
- 是否缺失
- 有没有 text / image

---

## 5.4 第三层：learned memory

这一层来自：

- `soul_prompt`
- `memory_l2`
- `tool_notes`
- `skill_prompt`

注入点在：

- `timeclaw/pipeline/runner.py:277-299`

然后在 executor 中被分成两类：

### 5.4.1 注入 system prompt 的内容

- `SOUL`
- `MEMORY`

对应：

- `timeclaw/agents/executor.py:709-720`
- `timeclaw/agents/executor.py:820-833`

### 5.4.2 注入 first-turn user message 的内容

- `SKILLS`
- `Tool Notes`
- profiling section
- subagent slot section

其中 `tool_notes` 的拼接逻辑在：

- `timeclaw/agents/executor.py:9457-9476`

这其实体现了一个很好的设计：

- **稳定身份与长期规则** 放系统层；
- **场景 SOP 与工具经验** 放用户层。

---

## 5.5 第四层：training / test mode metadata

这层对方法学最关键。

### 5.5.1 训练模式

训练模式会加入：

- `spawn_subagent`
- `evaluate_against_gt`
- `evaluate_batch_against_gt`
- tool-dropout scope/counts

见：

- `timeclaw/agents/executor.py:672-681`
- `timeclaw/pipeline/runner.py:313-326`

这意味着训练 prompt 和测试 prompt 的核心差别不是文案，而是：

**训练期允许“学”，测试期只允许“用”。**

### 5.5.2 测试模式

测试模式会移除：

- `spawn_subagent`
- `evaluate_*`

只保留正常执行工具。

所以测试期拿到的是：

- 已蒸馏知识
- 但没有 GT 泄漏
- 也没有训练期的探索接口

---

## 5.6 第五层：Executor 最终组装

`Executor.act()` 里，最终 prompt 有两部分：

### 5.6.1 system prompt

由以下内容组成：

1. `REACT_SYSTEM_PROMPT`
2. `system_memory`
3. `visible tool guidance`

见：

- `timeclaw/agents/prompts/executor_prompt.py:214-300`
- `timeclaw/agents/executor.py:820-833`

### 5.6.2 first user prompt

由 `REACT_FIRST_TURN` 模板拼接：

- Task Prompt
- Task Boundary
- Execution Context
- Control Context
- Output Contract
- Sample Fingerprint
- Profiling Results
- Support
- Available Tools

见：

- `timeclaw/agents/prompts/executor_prompt.py:302-321`
- `timeclaw/agents/executor.py:840-860`

### 5.6.3 简化后的组装图

```text
official MTBench prompt
  + sample fingerprint/profiling
  + SOUL/MEMORY
  + SKILLS
  + Tool Notes
  + mode-specific tool visibility
  + subagent slot / dropout info
=> Executor first-turn prompt
```

---

## 5.7 非 forecast 任务的额外 prompt 组装

这一点很重要，容易被忽略。

在 `pipeline_eval.py` 中，non-forecast 任务不是直接相信 pipeline 的最终分类/QA 输出，而是：

1. 先跑 pipeline 收集 tool analysis
2. 再把 analysis 追加回官方 MTBench prompt
3. 让 final reasoning LLM 产出最终答案

例如 `mcqa`：

- `timeclaw/benchmark/pipeline_eval.py:574-577`

`trend`：

- `timeclaw/benchmark/pipeline_eval.py:641-645`

因此：

**当前测试期 prompt 实际上是“双阶段 prompt”**：

- 第一阶段：agent 执行 prompt
- 第二阶段：official task prompt + appended tool analysis

这既是优点，也是论文里要说明的评测差异点。

---

## 6. 创新点判断：哪些能强写，哪些要谨慎写

## 6.1 先说总体原则

截至 2026-04-15，我建议把创新点分成三档：

1. **强主张**：当前代码和文献都支持。
2. **谨慎主张**：方向成立，但“首个/唯一”不稳。
3. **不建议主张**：当前证据不够，或容易被最新工作顶掉。

---

## 6.2 强主张

### A. 面向 time-series agent 的“探索-比较-蒸馏-回注”外部化学习闭环

这是我认为你最强的主创新点。

现有工作里：

- TimeART 强在 tool-use training
- TS-Debate / competition 强在 inference-time 协作
- ASDA 强在 skill distillation

但 TimeClaw 把下面四件事做成了一条闭环：

1. 训练期候选探索
2. GT 比较
3. 外部 knowledge distillation
4. 测试期 knowledge reinjection

这点是能成立的。

### B. 训练期“强制探索”机制，用于数值型任务的 agent 学习

你的原话“第一个针对数值型任务的 agent 学习范式（强制探索）”，我建议改写成：

**one of the first explicit forced-exploration learning protocols for time-series agents on numeric tasks**

为什么不要写“the first”：

- 现在已经有 competition / debate / tool-training 类工作；
- 它们也会比较多个 path 或 agent；
- 但它们通常不是把“强制分叉 + GT 比较 + 蒸馏”作为主学习协议。

所以最稳的点不是“有人从没探索过”，而是：

**你把探索本身制度化成了训练协议。**

### C. tool-dropout：用结构化机制对抗 scope-local 工具先验

这个点很有价值，而且我认为可以强写成你的方法贡献。

代码与设计文档都很清楚：

- 实现：`timeclaw/agents/tool_dropout.py:11-40`
- 设计：`reports/04_tool_dropout_final_design.md:1-158`

它做的不是普通随机 mask，而是：

- 统计 per-scope 工具使用频率
- 高频工具更容易被隐藏
- 最冷门工具不主动被 mask
- subagent 独立采样不同工具子集

而且这个创新点和最新工作是同向的：

- [SABEval / Do LLMs Know Tool Irrelevance?](https://arxiv.org/abs/2604.11322) 在 2026-04-13 指出 LLM 有明显的 structural alignment bias in tool invocation

SABEval 不是做 time-series agent 的，但它说明：

- 工具先验/结构偏差是真问题
- TimeClaw 的 tool-dropout 正好是在 time-series agent 训练里正面处理这个问题

所以这点你完全可以写，但建议写成：

**a scope-aware tool-dropout mechanism to reduce tool popularity bias / tool prior during agent learning**

不要写成“彻底解决”。

### D. 层级化经验蒸馏，不改权重的测试期增益路径

这点也很强。

当前系统不是简单做 reflection note，而是明确区分：

- Notes
- MEMORY
- TOOLS
- SKILLS
- history
- state

而且是增量、带 watermark、带 fingerprint 地运行。

这使得 TimeClaw 的学习不是“一次性反思”，而是：

**a managed external memory system for continual agent improvement**

### E. scope-specific + cross-domain 双层迁移

当前蒸馏不只保留 exact scope 规则，还把同 task bucket 聚成 `cross_domain__*`。

这意味着你不是只在背“weather_aligned_short_forecast 的经验”，而是在同时学：

- 本 scope 的精细策略
- 同类任务跨 domain 的抽象规则

这点是很有论文味道的，可以作为一个独立小贡献。

---

## 6.3 谨慎主张

### F. “开发了一个新的学习数据集，用于检验 agent 学习能力”

这个点要小心。

当前更准确的说法不是“全新 benchmark dataset”，而是：

**a derived agent-learning corpus and evaluation protocol built on official MTBench train/test splits**

为什么？

因为现在代码里最清楚的事实是：

- 训练数据默认还是 official MTBench processed pool 的 train split
- 但 TimeClaw 在上面生成了新的 learning artifacts：
  - candidate traces
  - learning summary
  - analysis_structured
  - hierarchical memory snapshots

也就是说，你现在已经有了一个：

- **新的 agent learning 数据生产流程**
- 以及由此产生的 **新型学习语料**

但如果你想把它写成“new dataset”，最好满足两件事：

1. 把这个学习语料单独命名、打包、公开。
2. 明确它的样本单位、字段结构、评价目标。

在当前状态下，我建议写成：

**we construct a new benchmark-derived learning corpus for studying whether a time-series agent can improve from its own exploratory executions**

这会更稳。

### G. “首个数值型 agent learning 范式”

不建议直接写 “first”。

建议改成：

- first **time-series-agent-oriented**
- or one of the first **explicit**
- or the first to combine **forced exploration + GT comparison + external distillation**

把 claim 收窄，才能稳。

---

## 6.4 不建议主张

### H. “首个通用多模态 time-series agent”

不建议。

原因很直接：

1. 当前 benchmark 主线没有 image。
2. MAS4TS、TS-Debate、MMTS-BENCH 等 2026 工作已经把 multimodal 讲得更完整。
3. 你的强点不在“模态更多”，而在“学习闭环更强”。

### I. “严格意义上的 raw MTBench SOTA”

也不建议直接写。

因为当前 non-forecast 任务评测协议和 raw official MTBench evaluation 不完全一致。

### J. “完全不依赖模型权重学习”

也不要绝对化。

更准确的说法是：

- **the main improvement path in TimeClaw is externalized, not weight-updating**

---

## 6.5 我建议你最终写成的贡献列表

下面这版更适合写论文：

1. **We propose TimeClaw, an externalized learning framework for time-series agents.**
   It turns benchmark execution into a closed loop of explore, compare, distill, and reinject.

2. **We introduce a forced-exploration learning protocol for numeric time-series tasks.**
   During training, the agent is encouraged to branch into multiple candidate paths, compare them against ground truth, and summarize winner-aligned reusable rules.

3. **We design a scope-aware tool-dropout mechanism to mitigate tool popularity bias.**
   Instead of repeatedly overusing familiar tools, sub-agents are exposed to diversified tool subsets conditioned on per-scope usage statistics.

4. **We build a hierarchical external memory stack for agent improvement without weight updates.**
   The system distills raw run outcomes into scope-level memory, tool notes, skill prompts, and cross-domain abstractions that are reinjected at test time.

5. **We construct a benchmark-derived learning corpus for studying agent improvement on time-series reasoning tasks.**
   The corpus is not just raw benchmark data, but candidate traces, comparisons, learning summaries, and distillation snapshots generated from exploratory executions.

6. **We provide a concrete mechanism for separating exploration-time priors from execution-time evidence.**
   Historical path preference can guide branch construction during training, but final decision evidence remains grounded in actual candidate outputs and GT-based comparison.

7. **We make agent learning observable and reproducible through artifact-native design.**
   Train rows, traces, distillation state, and history snapshots allow researchers to inspect how experience accumulates over time.

---

## 7. 论文里建议怎么写“与现有 time-series + agent 工作的区别”

我建议你用下面这段逻辑。

### 7.1 不要写成“我们比别人更会做题”

因为这很容易被 reviewer 问：

- 你是不是只是 benchmark protocol 改了？
- 你是不是只是 prompt 更长？
- 你是不是用了更多工具？

### 7.2 建议写成“我们研究的是另一层问题”

即：

- TimeART / TS-Agent / TimeSeriesScientist：研究 agent 如何做 time-series reasoning
- TimeClaw：研究 **agent 如何从 time-series reasoning 中学习**

这两层问题不一样。

前者是：

- execution capability

后者是：

- improvement mechanism

这会让你的论文主轴更清楚。

---

## 8. 当前实现的边界与非主张

为了避免后面写论文时被自己坑，下面这些点建议直接在内部统一口径。

### 8.1 当前 benchmark 主线不是 image benchmark

代码支持 image，但当前 MTBench 主线没有 image loader。

### 8.2 当前测试协议对 non-forecast 有 final reasoning 追加步骤

所以要明确写：

- “built on MTBench”
- 而不是“identical to raw official MTBench evaluation”

### 8.3 当前系统的主创新不在 backbone model

TimeClaw 更像：

- learning system
- orchestration system
- distillation system

而不是新 foundation model。

### 8.4 当前“新数据集”更像派生 learning corpus

如果要变成强 benchmark contribution，需要额外产品化和公开化。

---

## 9. 最后给你的一个总判断

如果你后续论文想打得更稳，我建议把标题和摘要的主句都往下面这条线上收：

**TimeClaw is a time-series agent learning framework, not merely a tool-augmented time-series agent.**

你最值钱的不是：

- 比别人多了几个工具
- 比别人多了个模态
- 比别人多了个 agent

而是你把下面这件事做成了系统：

**agent 在数值时序任务上，如何通过强制探索、候选比较和外部化蒸馏，真正形成可迁移、可回注、可观察的经验。**

这才是你和 MTBench、TimeART 以及大多数 related works 最不一样的地方。
