# TimeClaw 全流程与记忆模块设计说明

更新时间：2026-04-04

本文档基于两部分信息整理：

1. 当前代码实现。
2. 之前多轮设计讨论中已经明确的约束与取舍。

目标不是写一份宣传性介绍，而是把 TimeClaw 目前真正落地的训练 / 测试 / 记忆链路说明白，方便后续继续开发、对齐行为、排查偏差。

---

## 1. 文档范围

本文覆盖两块：

1. 全流程：
   从 MTBench 数据接入、训练、蒸馏、测试，到 `PipelineRunner` / `ExecutorAgent` 的执行链路。
2. 记忆模块：
   `Notes`、`MEMORY`、`TOOLS.md`、`SKILLS/` 的结构、更新逻辑、蒸馏时机、注入方式。

不展开介绍：

1. 各单个工具的内部算法实现。
2. 底层 LLM / MCP 基础设施细节。
3. 旧版多 Agent 架构的历史实现。

---

## 2. 术语与新定义

### 2.1 `scope_key`

新定义：
`scope_key` 是当前系统中训练、蒸馏、注入的硬边界主键。

它直接对应一个 benchmark scope，例如：

- `weather_aligned_short_forecast`
- `finance_aligned_long_indicator_macd`

当前设计里，scope 是第一优先级边界。记忆不会跨 scope 混检后再做相似度召回。

### 2.2 `Notes`

新定义：
`Notes` 是原始经验记录层，只作为蒸馏源，不直接注入 agent。

这和较早期设计不同。当前代码里：

- 会写 notes
- 会从 notes 蒸馏出 MEMORY / TOOLS / SKILLS
- 但不会把 note 本身直接塞回 prompt

### 2.3 `released task` 与 `扩展 task`

新定义：

1. `released task`
   指官方 `MTBench_eval` 发布脚本中实际跑过的 task。
2. `扩展 task`
   指我们代码里支持、但不一定属于官方 released eval 的 task。

当前代码已经把 benchmark 训练 / 测试任务收敛到了一个较小子集，见第 3 节。

### 2.4 `defer_learning_record`

新定义：
测试阶段如果需要借用训练态的 GT 评估能力来产生学习材料，但不希望 executor 自己返回 `learning_summary`，就启用 `defer_learning_record=True`。

此时：

1. Executor 仍可看到 GT 评估工具。
2. 但最终任务答案仍需保持原 task type。
3. 学习记录由外层框架统一生成。

---

## 3. 当前代码基线与任务范围

### 3.1 主入口

统一入口在：

- `timeclaw/benchmark/run_learning.py`

支持四步：

1. `split`
2. `train`
3. `distill`
4. `test`

`--all` 就是顺序跑完这四步。

### 3.2 当前 benchmark 数据注册

当前 benchmark 数据注册在：

- `timeclaw/benchmark/mtbench_adapter.py`

当前代码不是按“更大的历史 scope 设想”来跑，而是按当前 registry 中启用的 task 跑。

代码里已经明确写了：

- Weather QA 任务当前不进 benchmark 主链
- Weather indicator 当前也被排除
- 目标 expanded task 数是 `18`

也就是说，当前代码基线是“18 个 expanded benchmark tasks”的版本，而不是更早讨论过的 23-scope 全量方案。

### 3.3 当前启用的 benchmark task 集

按当前 `DATASET_REGISTRY` 与 `expand_dataset_tasks()`，训练 / 测试主链实际启用的是：

1. `weather_aligned_short`
   - `forecast`
   - `trend`
   - `trend_past`
2. `weather_aligned_long`
   - `forecast`
   - `trend`
   - `trend_past`
3. `finance_aligned_short`
   - `forecast`
   - `trend`
   - `indicator_macd`
   - `indicator_upper_bb`
4. `finance_aligned_long`
   - `forecast`
   - `trend`
   - `indicator_macd`
   - `indicator_upper_bb`
5. `finance_qa_short`
   - `mcqa`
   - `correlation`
6. `finance_qa_long`
   - `mcqa`
   - `correlation`

总计：18 个 expanded tasks。

### 3.4 Prompt 来源

当前 prompt 构造已经不是手写简化模板，而是官方 prompt bridge：

- `timeclaw/benchmark/mtbench_prompts.py`
- `timeclaw/benchmark/mtbench_support/task_descriptions.py`

设计要点：

1. `mtbench_prompts.py` 会桥接到 `mtbench_official/` 下的官方 prompt 生成代码。
2. `task_descriptions.py` 只负责把 `(sample, task)` 映射成最终 `description`。
3. 因为 prompt 是 dataset-task 级别绑定，所以“不同窗口 / 不同任务的描述不能混用”是这个模块的核心约束。

---

## 4. 全流程总览

完整链路可以概括为：

1. 用官方 processed data 和 splits 生成样本列表。
2. 训练时按 `(dataset, task)` 逐样本调用 `pipeline.run(mode="train")`。
3. 训练输出写成 notes，并触发分层蒸馏。
4. 测试时按官方 prompt 重新运行 pipeline。
5. 对非 forecast 任务，再用官方原始 prompt + TimeClaw 工具分析做最终结构化回答。

可以写成：

`MTBench sample -> PipelineRunner -> ExecutorAgent -> execution_log -> learning record -> Notes -> MEMORY / TOOLS / SKILLS -> test-time injection`

---

## 5. 训练流程设计

训练主循环在：

- `timeclaw/benchmark/train_loop.py`

### 5.1 训练循环做什么

对每个样本、每个支持 task：

1. 提取 ground truth。
2. 通过 `build_task_description()` 构造任务描述。
3. 调 `pipeline.run(..., mode="train", ground_truth=gt)`。
4. 让 executor 自主探索多条路径。
5. 对候选结果做 GT 评估。
6. 把学习结果写入记忆系统。

### 5.2 Ground truth 提取原则

训练不是统一一个 GT 结构，而是 task-native：

1. `forecast`
   - `output_window`
2. `trend`
   - weather 用 3-way label
   - finance 用 `trend_label` 或从 future 窗口推导
3. `trend_past`
   - 仅 weather 有 GT
4. `indicator`
   - weather 用 `max/min/diff`
   - finance indicator 已拆成 `indicator_macd` / `indicator_upper_bb`
5. `mcqa`
   - 用题目答案
6. `correlation`
   - 用标注的 correlation label

### 5.3 训练时 executor 的硬约束

当前代码已落地的关键约束：

1. 训练模式必须先做至少一次 GT evaluation，不能直接结束。
2. 训练模式必须探索至少 2 条有效 candidate path。
3. 如果 answer type 与任务类型不匹配，会触发自纠正。
4. 如果 final payload 类型不合法，会触发自纠正。
5. 子 agent 看不到 evaluation 工具。

这些约束主要在：

- `timeclaw/agents/executor.py`
- `timeclaw/agents/prompts/executor_prompt.py`

### 5.4 为什么要至少 2 条 candidate

这是我们之前讨论后明确保留的设计：

1. 训练不是为了把单次答案做出来。
2. 训练是为了比较方案，沉淀“什么时候该用哪条流程”。
3. 所以单路径训练样本的学习价值有限。

当前代码里 executor 会检查候选数，不足 2 条会要求继续探索。

---

## 6. 测试流程设计

测试主循环在：

- `timeclaw/benchmark/pipeline_eval.py`

### 6.1 Forecast 与非 Forecast 的分流

测试时分成两类：

1. `forecast`
   - 直接使用 pipeline 的预测结果
   - 必要时做结构化纠正
2. 非 forecast
   - 先运行 pipeline，拿到工具分析痕迹
   - 再走一次最终 LLM 调用
   - 最终 LLM 调用使用：
     - 官方原始 prompt
     - 原始输入上下文
     - TimeClaw 工具分析摘要

这样做的原因是：

1. 非 forecast 任务往往更依赖最终结构化表达。
2. 单靠中间 agent 结果容易丢掉 prompt 细节。
3. 所以当前测试链路更偏向“官方 prompt + 工具增强分析”的二阶段设计。

### 6.2 `defer_learning_record`

测试态如果 `_record_enabled` 为真，会把测试过程也包成一种“外层统一学习记录”模式：

1. Executor 可以使用训练态的 GT 评估工具。
2. 但最终返回的仍是 task-native answer。
3. 外层框架再从 execution log 中抽取候选、评估结果和最终 winner，统一写记忆。

这套模式主要是为了：

1. 保持测试结果结构正确。
2. 仍然能收集高质量的学习材料。

---

## 7. PipelineRunner 设计

`PipelineRunner` 是全流程的中心编排器：

- `timeclaw/pipeline/runner.py`

### 7.1 五步主链

当前 `run()` 的主链是：

1. `Preprocess`
2. `Memory injection`
3. `Executor`
4. `Post-process`
5. `Deliver`

### 7.2 Preprocess

预处理是纯代码，不走 LLM。

主要做三件事：

1. 模态检测
2. 指纹提取
3. profiling tools

当前默认 profiling 包括：

1. `basic_stats`
2. `autocorrelation`
3. `stationarity_test`
4. `detect_trend`
5. `detect_anomaly`

输出进入：

1. `fingerprint`
2. `profiling_results`
3. `modalities`

### 7.3 Memory injection

当前注入到 executor 的只有三块：

1. `SOUL`
2. `MEMORY`
3. `TOOLS.md`
4. `SKILLS`

注意：

1. `memory_l1` 当前明确被置空。
2. 也就是 Notes 本身不注入。

### 7.4 Post-process

当前后处理是纯代码质量门：

1. `QualityGate`
2. `bound_check`
3. `consistency_check`

不会再额外用 LLM 去做通用兜底。

### 7.5 训练记录入口

训练态的学习记录由 `_save_learning_outcome()` 统一生成。

这一步会做：

1. 从 execution log 提取 candidate traces
2. 汇总所有 evaluation metrics
3. 选 winner
4. 生成 `analysis_structured`
5. 写入 `HierarchicalMemory.record()`

这一步是“Notes 质量”的关键来源。

---

## 8. ExecutorAgent 设计

Executor 是当前单 agent 架构的核心：

- `timeclaw/agents/executor.py`

### 8.1 为什么是单 Executor

当前代码不再走旧的 Manager + Analyzer + Executor 多 agent 主链。

设计转向是：

1. 预处理阶段用纯代码完成静态诊断。
2. 主执行阶段只保留一个 ReAct executor。
3. 复杂探索通过 `spawn_subagent` 做局部并行。

这样做的好处是：

1. 数据和任务描述不在主链里多次压缩转述。
2. 模态信息、指纹、profiling、工具信息都在一个 agent 上统一使用。
3. 训练日志与候选路径更容易对齐。

### 8.2 工具过滤原则

当前工具不是按 task 过滤，而是按模态过滤。

实现上：

1. 根据 `required_modalities()` 决定工具是否暴露。
2. 如果任务是多模态，兼容这些模态的工具都能看到。
3. `llm_reasoning_*` 作为独立工具暴露，不是单一总工具。

这是我们之前讨论后保留的方向：

1. task 过滤太强，会抑制探索。
2. 模态过滤更像真正的输入约束。

### 8.3 Thought 结构要求

当前 executor 和 sub-agent 都要求 `thought` 里显式包含：

1. `Reason:`
2. `Action:`

不满足会触发自纠正。

这样做的目的：

1. 让中间轨迹更可读。
2. 让 notes / learning analysis 有可用的推理材料。

### 8.4 Sub-agent 设计

`spawn_subagent` 用于并行探索。

当前实现要点：

1. 最多并发 `4` 个 sub-agent。
2. sub-agent 继承主 agent 的：
   - task description
   - fingerprint
   - profiling section
   - system memory
   - skill prompt
   - tool notes
3. sub-agent 看不到 GT evaluation 工具。
4. sub-agent 如果被分配了 `tools_hint`，就必须真实调用其中一个工具后才能结束。

这对应我们之前的设计共识：

1. 候选必须先独立生成。
2. evaluation 必须由主 agent 统一做。

### 8.5 GT 生命周期

训练时 ground truth 会注入到 executor 元数据中，但不是永久留在对话里。

当前机制是：

1. `evaluate_against_gt` / batch evaluate 执行时可以看到 GT。
2. agent 在本轮读取 evaluation 结果后，
3. 框架会把对话里的 GT 信息清洗掉，避免后续继续“看答案做题”。

这对应我们之前反复确认过的要求：

1. 允许学习阶段看 GT。
2. 但不允许看过 GT 后继续自由改写最终答案。

---

## 9. Notes 设计

Notes 在：

- `.timeclaw/hierarchical_memory/notes/*.md`

生成逻辑在：

- `timeclaw/memory/hierarchical.py` 的 `Notes`

### 9.1 Notes 的定位

当前定位非常明确：

1. 它是原始蒸馏源。
2. 它不是在线检索记忆。
3. 它兼顾人工巡检与下游蒸馏。

### 9.2 当前 note 内容

一个 note 主要包含：

1. `Scope`
2. `Task`
3. `Text Signals`
4. `Winner Tool`
5. `Prediction`
6. `Metrics`
7. `Fingerprint`
8. `Approaches`
9. `Candidates`
10. `Analysis`

### 9.3 为什么保留 candidate traces

我们之前明确过：

1. note 不能只写抽象总结。
2. 需要保留足够真实的 candidate 级别证据。
3. 不然后续 MEMORY / TOOLS / SKILLS 蒸馏会失真。

当前 `candidate_traces` 会保留：

1. 候选名
2. tool chain
3. prediction
4. steps
5. metrics

---

## 10. `analysis_structured` 设计

单样本学习分析在：

- `timeclaw/benchmark/mtbench_support/learning_analysis.py`

输出五个字段：

1. `text_signals`
2. `sample_conditions`
3. `winner_evidence`
4. `failure_signals`
5. `reusable_rule`

### 10.1 设计目标

这一步不是为了生成漂亮总结，而是为了产出可蒸馏材料。

所以要求：

1. 必须是 evidence-grounded
2. 必须区分 winner stronger 的原因和 loser 的失败信号
3. `reusable_rule` 必须是未来可复用的规则
4. 禁止写框架动作词
   - 如 `spawn_subagent`
   - `evaluate_batch_against_gt`

### 10.2 为什么有 `text_signals`

这是我们之前专门讨论并保留的设计：

1. 文本原文太长，直接塞 note 信息密度太低。
2. 但完全不保留文本线索，又会丢掉关键证据。
3. 所以这里保留 3-6 条高信息片段，作为蒸馏原料。

---

## 11. MEMORY 设计

MEMORY 存储在：

- `.timeclaw/hierarchical_memory/memory/*.json`

核心结构在：

- `MemoryEntry`
- `MemoryStore`

### 11.1 MEMORY 的定位

MEMORY 不是原始经验堆积，也不是 SOP。

它的定位是：

1. scope 内的结构化规则层
2. 强调“什么时候该怎么做，为什么”
3. 作为 SKILLS 和 TOOLS 的中间知识层

### 11.2 Rule Schema

当前一条 MEMORY rule 包含：

1. `scope`
2. `rule_id`
3. `kind`
4. `summary`
5. `conditions`
6. `preferred_tools`
7. `avoid_tools`
8. `rationale`
9. `evidence`
10. `confidence`
11. `injectable`
12. `status`
13. `timestamp`

### 11.3 设计原则

我们之前讨论后，当前 schema 体现了几条关键原则：

1. scope 是硬边界，所以 `conditions` 不是检索主键。
2. rule 数量有上限，当前实现最终裁剪到 `20` 条。
3. 允许存在冲突规则。
4. 是否可注入由 `injectable` 控制。
5. 规则强弱主要由：
   - `support_count`
   - `last_seen_at / timestamp`
   - 冲突支持度
   综合计算出 `confidence`

### 11.4 为什么 MEMORY 不直接做复杂筛选

这是我们之前讨论里很关键的一点：

1. scope 已经是强边界。
2. 每个 scope 的规则数又被严格限制。
3. 所以不需要再做复杂语义检索。

当前注入原则偏简单：

1. 保留 scope 内规则。
2. 优先注入 `injectable=true` 的规则。
3. 按 `confidence` 和 `support_count` 排序。

---

## 12. TOOLS.md 设计

TOOLS 存储在：

- `.timeclaw/hierarchical_memory/TOOLS.md`

实现类：

- `ToolsStore`

### 12.1 TOOLS 的定位

TOOLS 不是“工具大全”，而是单工具边界卡片。

它回答的是：

1. 这个工具什么时候有帮助
2. 什么时候该避免
3. 输出有什么典型风险

### 12.2 当前字段

每个工具卡片包含：

1. `when_it_helps`
2. `when_to_avoid`
3. `output_risks`

### 12.3 当前蒸馏来源

这是我们后来明确调整过的一点：

1. TOOLS 不再直接从原始 notes 的 winner 工具统计直接总结。
2. 当前 TOOLS 主要从已经蒸馏好的 MEMORY rules 再提炼。
3. notes 只继续提供 scope usage count。

这样做的原因：

1. 原始探索 winner 不一定可靠。
2. 直接从 notes 提炼容易重叠度过高。
3. 先经过 MEMORY 层再总结，更容易得到清晰边界。

### 12.4 注入策略

当前注入不是把所有工具卡片都塞进去。

而是：

1. 先按当前可见工具集合过滤。
2. 再按 scope usage 与相关性裁剪。
3. 最多注入少量高价值工具卡片。

---

## 13. SKILLS 设计

SKILLS 存储在：

- `.timeclaw/hierarchical_memory/skills/*.md`

实现类：

- `SkillsStore`

### 13.1 SKILLS 的定位

这是我们之前讨论里最强调的一点：

SKILLS 不能只是 MEMORY 的重复表述。

SKILLS 的定位应当是：

1. scene-specific workflow
2. 条件触发的操作流程
3. 比 MEMORY 更接近“怎么做”

换句话说：

1. MEMORY 回答“哪些规律成立”
2. TOOLS 回答“单工具边界”
3. SKILLS 回答“满足什么条件时，走哪条工作流”

### 13.2 当前技能模板

每条 skill 按固定结构输出：

1. `Condition`
2. `Why`
3. `Workflow`
4. `Guardrails`（可选）

### 13.3 当前蒸馏原则

从 MEMORY 到 SKILLS 的蒸馏规则是：

1. 优先使用 `injectable=true`
2. 优先高 `support_count`
3. 输出不超过 `5` 条 skill
4. 要求 workflow 去重，不能只是同义改写

---

## 14. Cross-domain 设计

系统里保留了 `cross_domain` 这个特殊 scope。

它的定位是：

1. 从所有 scope 的 MEMORY 中提炼更泛化的策略。
2. 用于沉淀跨领域共通经验。

但要注意当前代码状态：

1. `cross_domain` 规则会被存储。
2. DistillationAgent 也会尝试蒸馏它。
3. 但当前 `MemoryStore.get_prompt_section()` 中，`cross_entries` 实际没有加入注入输出。

也就是说：

1. `cross_domain` 当前在存储层存在
2. 但当前主链注入里并没有真正用上

这是“设计存在，但当前主链未启用”的状态。

---

## 15. Distillation cascade 设计

DistillationAgent 的级联是：

1. `Notes -> MEMORY`
2. `MEMORY -> cross_domain`
3. `MEMORY -> SKILLS`
4. `MEMORY -> TOOLS`

### 15.1 Distill 触发阈值

当前阈值：

1. `L1 -> L2`
   - 每 `2` 条 note 触发
2. `L2 -> cross_domain`
   - 每累计 `20` 条 MEMORY entry 左右触发

### 15.2 为什么这么低

这是当前实验性迭代配置，不是最终生产阈值。

目的是：

1. 让小样本训练时也能尽快看到蒸馏结果
2. 方便检查 memory / tools / skills 的质量

### 15.3 增量蒸馏与强制蒸馏

当前支持两种：

1. `force_distill_all()`
   - 重算全量
2. `distill_pending_all()`
   - 只完成挂起的增量蒸馏

---

## 16. 当前注入策略总表

测试 / 训练执行前，真正进入 agent 上下文的内容如下：

1. System prompt
   - `SOUL`
   - `MEMORY`
2. User turn
   - 完整 `task_description`
   - `fingerprint`
   - `profiling_section`
   - `tool_names`
   - `tool_notes_section`
   - `skills_section`

当前明确不注入：

1. `Notes`
2. 向量检索结果
3. `cross_domain` MEMORY（当前实现未接入 prompt）

---

## 17. 与历史讨论对齐后的关键设计结论

这一节专门记录我们之前讨论里已经沉淀成代码行为的部分。

### 17.1 已落地

1. Notes 不再注入 prompt。
2. Scope 是硬边界。
3. Finance indicator 已拆成 `indicator_macd` / `indicator_upper_bb`。
4. Prompt 已切到官方 bridge，而不是继续维护手写弱化版模板。
5. 训练要求至少 2 条 candidate path。
6. 子 agent 不可见 evaluation 工具。
7. GT 在 evaluation 后会从对话里清洗。
8. `llm_reasoning_*` 是独立工具，不是单个泛化工具。
9. 工具暴露按模态过滤，不按 task 强过滤。
10. TOOLS 从 MEMORY 提炼，不直接从 raw notes winner 工具统计生成。

### 17.2 当前代码状态与历史讨论仍有偏差的地方

1. Benchmark task 集当前是 18-task 子集，不是更早讨论过的更大 scope 集。
2. `cross_domain` 有存储，但当前没有真正注入。
3. `vector_store.py` 仍在仓库里，但已不属于当前主链设计。
4. Prompt bridge 支持的 dataset-task 范围大于当前 benchmark registry 启用范围。

---

## 18. 推荐的阅读顺序

如果后续要继续修改系统，建议按这个顺序读代码：

1. `timeclaw/benchmark/run_learning.py`
2. `timeclaw/benchmark/train_loop.py`
3. `timeclaw/benchmark/pipeline_eval.py`
4. `timeclaw/pipeline/runner.py`
5. `timeclaw/agents/executor.py`
6. `timeclaw/benchmark/mtbench_prompts.py`
7. `timeclaw/benchmark/mtbench_support/task_descriptions.py`
8. `timeclaw/memory/hierarchical.py`
9. `timeclaw/benchmark/mtbench_support/learning_analysis.py`

---

## 19. 一句话总结

当前 TimeClaw 的主设计可以概括为：

1. 用官方 MTBench prompt 和官方 processed data 驱动任务定义。
2. 用单 Executor + 可控 sub-agent 的方式做多路径工具探索。
3. 用 GT evaluation 保证训练样本能产生可比较的候选证据。
4. 用 Notes 承接原始学习痕迹。
5. 再蒸馏成 scope-local MEMORY、tool boundary cards 和 workflow-style SKILLS。
6. 测试时只注入 SOUL / MEMORY / TOOLS / SKILLS，不回灌 raw notes。
