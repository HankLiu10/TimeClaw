# TimeClaw 当前设计总览（Codex 0405）

更新时间：2026-04-07

本文档目标：

1. 用一份文档把 TimeClaw 当前已经落地的整体设计讲清楚。
2. 以“当前代码真实行为”为准，而不是以早期讨论中的理想方案为准。
3. 明确哪些部分是当前正式设计，哪些部分是当前实现中的已知限制或偏差。

本文主要参考：

- `timeclaw/benchmark/mtbench_adapter.py`
- `timeclaw/benchmark/run_learning.py`
- `timeclaw/pipeline/runner.py`
- `timeclaw/agents/executor.py`
- `timeclaw/agents/prompts/executor_prompt.py`
- `timeclaw/memory/hierarchical.py`
- `docs/pipeline_memory_design.md`

---

## 1. 总体目标

TimeClaw 是一个面向 MTBench 风格多模态时间序列任务的 agent 系统。

它的核心思路不是“让一个 LLM 直接答题”，而是：

1. 先把任务、时序、文本、图像等输入组织进统一上下文。
2. 让 agent 在工具空间里逐步探索。
3. 在训练阶段把探索结果写成结构化经验。
4. 再把经验蒸馏成更稳定的 `MEMORY / TOOLS / SKILLS`。
5. 在后续测试中把这些蒸馏产物重新注入，帮助 agent 更稳定地解题。

一句话概括：

`官方 MTBench prompt + TimeClaw 工具探索 + 分层经验蒸馏 + 测试时经验回注入`

---

## 2. 当前任务范围

### 2.1 新定义：`scope`

新定义：
`scope` 是当前训练、蒸馏、注入的硬边界。

例子：

- `weather_aligned_short_forecast`
- `finance_aligned_long_indicator_macd`

这意味着：

1. notes 按 scope 写入。
2. MEMORY 按 scope 学习。
3. TOOLS 和 SKILLS 也按 scope 组织。
4. 默认不会先跨 scope 混合检索，再做软召回。

### 2.2 当前正式 benchmark 任务数

当前主链不是 23 个任务，而是 **18 个 expanded benchmark tasks**。

正式启用的数据集是 6 个：

1. `weather_aligned_short`
2. `weather_aligned_long`
3. `finance_aligned_short`
4. `finance_aligned_long`
5. `finance_qa_short`
6. `finance_qa_long`

展开后的 18 个 scope-task 为：

1. weather short: `forecast / trend / trend_past`
2. weather long: `forecast / trend / trend_past`
3. finance aligned short: `forecast / trend / indicator_macd / indicator_upper_bb`
4. finance aligned long: `forecast / trend / indicator_macd / indicator_upper_bb`
5. finance qa short: `mcqa / correlation`
6. finance qa long: `mcqa / correlation`

当前刻意不纳入主链的任务：

1. weather indicator
2. weather qa

这是为了对齐当前采用的 TimeART 风格 18-scope 子集。

---

## 3. Prompt 来源与原则

### 3.1 当前 prompt 来源

TimeClaw 当前任务描述不是手写模板，而是桥接到官方 MTBench prompt。

这层约束很重要，因为我们当前的任务定义希望尽量贴近官方 prompt，而不是自己重新发明一套任务描述。

### 3.2 当前 prompt 组装原则

当前主 agent 和子 agent 共享同一套核心 ReAct prompt 骨架：

1. `Task Prompt`
2. `Execution Context`
3. `Output Contract`
4. `Sample Fingerprint`
5. `Profiling Results`
6. `Support`
7. `Available Tools`

新定义：
主 agent 和子 agent 的差异，尽量只通过下面三种东西体现：

1. 可见工具不同
2. `Execution Context` 中插入的块不同
3. 训练 / 测试模式下可见工具与 guidance 不同

例子：

- 主 agent 可以看到 `spawn_subagent` 和 `evaluate_*`
- 子 agent 看不到它们
- 所以 prompt 会自动出现或不出现对应 guidance

### 3.3 当前 system prompt 的核心规则

当前 executor 的核心规则是：

1. 一次只能做一件事
2. 一轮只能输出一个顶层 JSON
3. 不是调用工具，就是正式结束
4. `done=true` 时必须给出合法 `answer`

这样做的目的，是压制以前常见的“双顶层 JSON”“一边 tool call 一边提前写 final answer”的混乱输出。

---

## 4. Pipeline 总流程

当前主链可以概括成 4 步：

1. `split`
2. `train`
3. `distill`
4. `test`

统一入口在：

- [run_learning.py](/home/lhc/workspace/TimeClaw/timeclaw/benchmark/run_learning.py)

更细一点的运行顺序是：

1. 用官方数据与 split 列表取样本
2. `PipelineRunner.run(...)`
3. 预处理：模态识别、fingerprint、profiling
4. executor 进入 ReAct 循环
5. 得到任务答案与执行轨迹
6. 训练阶段将轨迹转成 notes，并登记增量 distill 工作
7. 测试阶段只做推理，不更新记忆

---

## 5. 训练与测试的核心区别

### 5.1 训练模式

训练模式下：

1. 主 agent 可见 `spawn_subagent`
2. 主 agent 可见 `evaluate_against_gt / evaluate_batch_against_gt / batch_evaluate_against_gt`
3. 会把 GT 放进训练上下文，但评估结果会在后续步骤中做去污染处理
4. 训练主 agent 的最终 `answer_type` 必须是 `learning_summary`
5. 至少需要 3 条有效 candidate path
6. 至少要做一次 GT evaluation

例子：

如果主 agent 只探索了 2 条路径就想结束，系统会自纠，要求继续探索。

### 5.2 测试模式

测试模式下：

1. `spawn_subagent` 不可见
2. `evaluate_*` 不可见
3. 不会更新 notes / memory / tools / skills
4. 不涉及训练态 GT

新定义：
测试阶段与训练阶段的 executor 本体尽量保持同构，但通过“可见工具不同”体现行为差异。

---

## 6. 主 agent 与子 agent 设计

### 6.1 主 agent 的职责

主 agent 的职责不是直接硬编码整条路径，而是：

1. 看见当前样本的完整任务与支持信息
2. 在必要时决定是否要做多分支探索
3. 如果需要，就调用 `spawn_subagent`
4. 收集子路径候选
5. 在训练态做 GT evaluation
6. 最终输出 `learning_summary`

### 6.2 子 agent 的职责

子 agent 的职责更窄：

1. 接收主 agent 指定的 `Branch Goal`
2. 接收该分支的 `tools_hint`
3. 在自己可见的工具子集里做短路径探索
4. 至少命中一个 hinted tool
5. 输出该分支的 task-typed 候选答案

### 6.3 子 agent 看见的额外信息

子 agent 的 `Execution Context` 当前会插入：

1. `Branch Goal`
2. `Assigned Exploration Path`
3. 可选的 `Parent Branch Metadata`

新定义：
`Parent Branch Metadata` 只允许传短、结构化、可审计的信息。

例子：

- `branch_id`
- `comparison_target`
- `candidate_family`

不允许混入：

1. 长推理
2. ground truth
3. evaluate 结果
4. 冗长工具观测复述

---

## 7. Tool-dropout 与分支设计

### 7.1 当前目标

tool-dropout 的目标不是随机搞残工具表，而是：

1. 给不同 sub-agent 造出不同的可见工具子集
2. 逼主 agent 设计更有差异的候选路径
3. 避免所有分支最后都走成同一条工具链

### 7.2 当前实现

当前训练模式下，主 agent 在 `plan()` 阶段会预先生成一批 candidate slots。

每个 slot 包含：

1. `slot_id`
2. `visible_candidates`
3. `keep_probabilities`
4. `never_drop`

然后 prompt 里会插入 `Candidate Sub-Agent Slots`，告诉主 agent：

1. 选不同 `slot_id`
2. 每个分支只能基于该 slot 的 visible tools 设计
3. 每个分支只需要是“短 advisory path suggestion”

### 7.3 当前 train / test 差异

tool-dropout 只在训练态对主 agent 的分支设计生效。

测试阶段：

1. 不做 spawn
2. 不做 dropout
3. 直接让单 agent 在完整测试工具集合里推理

---

## 8. 预处理与支持信息

进入 executor 之前，`PipelineRunner` 会先做 deterministic 预处理。

包括：

1. 模态检测
2. fingerprint 提取
3. profiling 工具

这些结果之后会进入 prompt 的：

1. `Sample Fingerprint`
2. `Profiling Results`

这样做的目的，是让 agent 不必先花一步去做最基础的静态画像。

---

## 9. Notes 设计

### 9.1 新定义：`Notes`

`Notes` 是蒸馏源，不再直接注入 prompt。

这和更早讨论过的“把最近 notes 直接喂回 agent”不同。

当前 notes 只承担两个职责：

1. 保存训练样本的原始学习记录
2. 为后续 MEMORY / TOOLS / SKILLS 蒸馏提供原料

补充一点当前真实行为：

1. `HierarchicalMemory.record()` 会先把 note 写到对应 scope 文件
2. 然后调用 `distiller.on_commit(scope_keys)`
3. 但这里不会同步等待整条蒸馏链全部跑完
4. 它做的是“登记新 note 已可用，并确保对应后台 worker 被触发”

### 9.2 Notes 当前存储形式

按 scope 分文件存储：

- `.timeclaw/hierarchical_memory/notes/<scope>.md`

一条 note 目前大致包含：

1. `Scope`
2. `Task Prompt`
3. `Numeric Time Series`
4. `Text Context`
5. `Text Signals`
6. `Winner Tools`
7. `Prediction`
8. `Metrics`
9. `Fingerprint`
10. `Approaches`
11. `Candidates`
12. `Analysis`

### 9.3 Notes 中保留 candidate traces 的原因

原因很直接：

1. 不能只写一句抽象总结
2. 后续 distillation 需要看到“候选路径长什么样”
3. 需要知道 winner 是怎么从候选里被比较出来的

但会过滤掉 framework 工具：

1. `spawn_subagent`
2. `evaluate_*`
3. `batch_evaluate_against_gt`

也不会把 ground truth 直接写进 candidate trace 的推理路径。

---

## 10. MEMORY 设计

### 10.1 新定义：`MEMORY`

当前 MEMORY 不是自由文本，而是结构化 rule。

每条 rule 包含：

1. `scope`
2. `rule_id`
3. `kind`
4. `summary`
5. `conditions`
6. `preferred_tools`
7. `avoid_tools`
8. `rationale`
9. `evidence`
10. `source_rule_ids`
11. `confidence`
12. `injectable`
13. `status`

### 10.2 support_count 的当前定义

新定义：
`support_count` 的真实口径是“支持这条规则的唯一 note id 数量”。

也就是：

`support_count = len(dedup(sample_refs))`

所以现在代码不会盲信磁盘里原始写下来的 `support_count`，而是按 `sample_refs` 现算。

### 10.3 当前 MEMORY 的去重与合并

当前有两层：

1. **按 `rule_id` 去重**
   - 这只是数据清洗
   - 它的作用是“同一 rule_id 重复写入时，只保留更强的一条”
   - 它不是语义合并
2. **按 `merge_groups` 做语义合并**
   - 这一步依赖 distiller 的 LLM 决策
   - 如果 LLM 没把近义规则放进 `merge_groups`，代码不会自动看懂它们是重复的

所以当前已知问题是：

1. 计数口径基本是对的
2. 但 `result_constraint` 和 `cross_domain` 中，近义规则的合并约束仍然偏弱

### 10.4 当前 MEMORY 的增量来源

新定义：
当前 `Notes -> MEMORY` 不是“每次取最近 N 条 notes”那种滑窗式做法，而是严格按水位线处理新增 notes。

当前实现里，每个 scope 都维护：

1. `notes_total_seq`
2. `notes_to_memory_done_seq`

真正送入一次 `Notes -> MEMORY` 的，是：

- `done_seq` 之后到本次 `target_seq` 之间的新增 note 切片

例子：

1. 某 scope 已经蒸馏到第 20 条 note
2. 现在累积到了第 31 条
3. 触发后送给 `Notes -> MEMORY` 的是第 21 到第 31 条，而不是“最近 30 条”

### 10.5 当前 MEMORY 的存储与注入上限

当前代码中的硬上限：

1. 每个 scope 最多保留 `20` 条 MEMORY rules
2. 每条 rule 的 `sample_refs` 最多保留 `200` 个 note id

当前 prompt 注入时的限制：

1. scope-local MEMORY 当前只注入最近 `8` 条 injectable rules
2. `cross_domain` 当前虽然有存储，但没有真正注入 prompt

这里是一个重要的“当前实现偏差”：

1. `cross_domain` 已经存在于存储层
2. 但 `MemoryStore.get_prompt_section()` 里，`cross_entries` 当前仍是空的
3. 所以 agent 实际推理时用不到 cross-domain MEMORY

---

## 11. TOOLS 设计

### 11.1 新定义：`TOOLS`

TOOLS 不是硬编码工具说明，而是“工具原生说明 + 按 scope 学到的软边界”。

更具体地说：

1. 工具原生 schema / description 负责说明：
   - 工具做什么
   - 输入是什么
   - 输出是什么
2. TOOLS.md 负责补充：
   - 什么时候它更有帮助
   - 什么时候应该避免
   - 它有哪些典型输出风险

### 11.2 当前 TOOLS 来源

当前 TOOLS 不再直接从 raw notes 的 winner tools 统计硬凑结论。

而是：

1. 先从 MEMORY 中抽取 tool-centered evidence
2. 再按当前 full MEMORY view 重建单工具边界卡片

这能避免早期那种“只是把 winner 工具频率抄一遍”的低质量经验。

### 11.3 当前 TOOLS 注入上限

当前每个 scope 注入给 agent 的 tool notes 最多 `15` 个。

每个工具卡片内部，每个字段通常只保留前 `2` 条边界。

---

## 12. SKILLS 设计

### 12.1 新定义：`SKILLS`

SKILLS 是从 MEMORY 再往上一层蒸馏出来的 workflow-style SOP。

它不是简单重复 MEMORY，而是把多条 rule 组织成“可执行操作流程”。

### 12.2 当前生成条件

当前如果某个 scope 的 MEMORY 少于 3 条，系统会直接删除该 scope 的 skill。

这意味着：

1. 技能层必须建立在足够多的稳定 rule 之上
2. 没有足够 MEMORY 时，不强行保留一个空洞 skill

### 12.3 当前 skill 输出约束

当前 prompt 要求：

1. 最多 5 条 skills
2. 总体保持简洁
3. 更倾向少而清晰，而不是堆很多近义 workflow

---

## 13. cross_domain 设计

### 13.1 新定义：`cross_domain`

`cross_domain` 不是跨全库乱混，而是按 `task bucket` 聚合。

例子：

- `cross_domain__aligned_short_forecast`
- `cross_domain__qa_long_mcqa`

它的目标是：

1. 从多个同 bucket 的 scope-local MEMORY 中抽更泛化的策略
2. 形成更通用的 rule

### 13.2 当前行为

当前 `cross_domain` 已经具备：

1. 独立存储
2. 从 source rule ids 回溯 note 证据
3. 独立 `support_count`

但当前仍有两个限制：

1. `cross_domain` 当前没有真正注入 prompt
2. 近义 rule 的合并约束仍然偏弱

---

## 14. 当前蒸馏链路

### 14.1 当前链路

当前蒸馏链路是：

1. `Notes -> MEMORY`
2. `MEMORY -> TOOLS`
3. `MEMORY -> SKILLS`
4. `MEMORY -> cross_domain`

### 14.2 当前触发机制

新定义：
当前 `Notes -> MEMORY` 不再是“旧版同步串行、每次只看最近若干条 note”的逻辑，而是：

1. 一个 scope 累积到至少 `10` 条未蒸馏 notes，触发一次 `Notes -> MEMORY`
2. 真正送入 distiller 的是“上次 done_seq 之后的新增 notes 切片”
3. `MEMORY -> TOOLS / SKILLS / cross_domain` 不再按 note 数简单阈值跑
4. 而是看 source fingerprint 是否变化，若变化则重建

也就是说，当前更像：

1. note 层按 sequence 增量推进
2. MEMORY 按新增 notes 增量更新
3. downstream 层按最新 full MEMORY view 重建

这里再明确三个重要点：

1. `TOOLS` 当前是 `replace(scope, ...)`
   - 不是在旧工具卡片上增量追加
   - 而是基于当前 full MEMORY 重新生成该 scope 的完整 TOOLS 卡
2. `SKILLS` 当前也是从该 scope 的 full MEMORY 重新生成
   - 如果 MEMORY 不足 3 条，会直接删掉 skill
3. `cross_domain` 也是从某个 bucket 的 full scope-local MEMORY 重新生成
   - 如果 bucket 内有效 MEMORY 不足 4 条，会直接清空该 bucket 的 cross-domain rules

### 14.3 当前 runtime 机制

当前 memory 不再把整条蒸馏链都同步塞进一个 sample 的收尾里。

而是：

1. 每个 scope 有自己的 worker
2. 每个 task bucket 有自己的 worker
3. 状态通过 `distill_state.json` 维护
4. scope worker 负责：
   - `Notes -> MEMORY`
   - `MEMORY -> SKILLS`
   - `MEMORY -> TOOLS`
5. bucket worker 负责：
   - `MEMORY -> cross_domain`
6. `distill_pending_all()` 用于把当前积压的蒸馏工作全部 drain 完

新定义：
当前 distill 的推进单位不是“某个函数这次被调用了没”，而是：

1. scope 侧看 `notes_to_memory_done_seq`
2. downstream 侧看 `memory fingerprint`
3. bucket 侧看 `memory_to_cross_done_fingerprint`

这套设计的目标是：

1. 保证按 scope / bucket 正确推进
2. 让训练样本写完 note 后尽快返回，而不是长期阻塞在完整蒸馏上
3. 用 fingerprint 保证 downstream 重建只在 source 真变化时才发生
4. 保留“最终可 drain 干净”的语义

### 14.4 当前状态与历史产物

distill 这套机制现在不只写最终产物，还会写两类辅助状态：

1. `state/distill_state.json`
   - 记录 scope / bucket 的当前水位、fingerprint、inflight 标记
2. `history/<scope>/`
   - 记录每次 `Notes -> MEMORY` 成功后的 scope-local MEMORY snapshot

新定义：
`memory history snapshot` 指某个 scope 在一次成功的 `Notes -> MEMORY` 之后，被完整保存下来的 MEMORY 快照。

它记录：

1. 这次快照覆盖的 notes 序号范围
2. 当时的 `memory_fingerprint`
3. 当时完整的 MEMORY entries

这类产物主要用于：

1. 回溯某个 scope 的 MEMORY 是如何一步步演化出来的
2. 排查某次 distill 后规则突然异常，是哪一批 notes 导致的

---

## 15. 训练产物、trace 与结果文件

### 15.1 当前训练产物

训练后主要会有这些产物：

1. `notes/*.md`
2. `memory/*.json`
3. `TOOLS.md`
4. `skills/*.md`
5. `train_results/*.json`
6. `traces/*.jsonl`

### 15.2 trace 的用途

trace 的主要用途不是给人看漂亮日志，而是：

1. 复盘 agent 每一步做了什么
2. 看 prompt 和执行轨迹是否串样本
3. 看 self-correction 是因为什么触发
4. 看哪一步调用了哪些工具

### 15.3 当前 trace 设计原则

当前追求的是：

1. 样本级隔离
2. 不混 GT 到错误的位置
3. 不把 framework 工具错误地当成候选求解路径

---

## 16. 当前已知限制与偏差

下面这些不是理想设计，而是当前代码状态下已经确认的点：

1. `MEMORY` 当前只注入最近 8 条 scope-local rules，不是全量注入。
2. `cross_domain` 当前有存储，但没有真正注入到 prompt。
3. `MEMORY` 的语义合并依赖 LLM 的 `merge_groups`，因此近义 rule 可能残留。
4. `result_constraint` 与 `cross_domain` 里的近义 rule 去重约束仍偏弱。
5. TOOLS 的 `_MIN_RUNS = 5` 常量当前没有真正作为硬门槛生效。

---

## 17. 当前设计的一句话总结

当前 TimeClaw 的真实设计可以概括为：

1. 用官方 MTBench prompt 和当前 18 个正式 benchmark scopes 定义任务。
2. 用主 agent + 可选 sub-agent 的 ReAct 工具探索完成训练。
3. 训练态通过 `spawn + evaluate` 获得多候选与 GT 反馈。
4. 将训练产物写成结构化 notes。
5. 再把 notes 按 scope 水位线增量蒸馏成 scope-local MEMORY，并基于最新 full MEMORY view 重建 TOOLS / SKILLS / cross_domain。
6. 测试态不再使用 `spawn / evaluate`，只使用蒸馏后的经验与普通工具推理。

如果后续要继续改系统，最重要的是始终分清三件事：

1. 什么是“设计目标”
2. 什么是“当前代码真实行为”
3. 什么是“当前实现偏差，但还没修”

这份文档主要记录的是第 2 和第 3 项。
