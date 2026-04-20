# TimeClaw 汇报 PPT 设计文档

更新时间：2026-04-16

这份文档的目标是给你一份可以直接喂给 AI 生成 PPT 的设计说明。它不只是列 slide 标题，而是把：

- 讲述主线
- 每页要传达的核心信息
- 建议视觉
- 可直接放进页内的文案
- 讲稿提示
- 事实边界与引用来源

都整理好。

---

## 1. 汇报目标

这场汇报最适合被讲成一件事：

**TimeClaw 不是“又一个 time-series agent”，而是把 benchmark-grounded 的时序任务执行，变成一个可比较、可蒸馏、可复用的 external learning process。**

同时，这场汇报还要完成两个扩展目标：

1. 让听众快速理解 `OpenClaw` 代表的是什么新型 agent 范式。
2. 用一条更大的技术演进线索，把 `prompt engineering -> context engineering -> harness engineering` 讲清楚，并自然过渡到 `TimeClaw`。

---

## 2. 推荐汇报风格

### 2.1 总体风格

建议做成：

- 学术风
- 黑白灰主色 + 一种冷色强调色
- 信息密度高，但每页只保留一个主结论
- 图多于字
- 少做“功能列表”，多做“问题-机制-结果”结构

不建议做成：

- 创业路演风
- 花哨渐变 + 大量图标
- 每页很多小 bullet
- 系统架构图堆满模块名

### 2.2 视觉建议

推荐配色：

- 主色：`#111111`
- 次色：`#6B7280`
- 强调色：`#0F766E`
- 对比色：`#B91C1C`
- 背景：`#F8FAFC`

推荐字体方向：

- 中文：思源黑体 / HarmonyOS Sans / 阿里巴巴普惠体
- 英文：IBM Plex Sans / Source Sans 3 / Helvetica Neue

推荐版式：

- 标题大
- 主图居中
- 文字左右不超过 5 行
- 每页一个 takeaway

---

## 3. 建议页数

推荐做成 `14` 页主报告 + `4~6` 页备份页。

如果时间只有 `12` 分钟，可以压成 `10` 页。

如果时间在 `18~20` 分钟，建议保留 `14` 页主报告。

下面默认按 `14` 页主报告设计。

---

## 4. 整体叙事主线

建议按下面这条线讲：

1. AI 工程范式正在升级：从 `prompt` 到 `context` 再到 `harness`
2. `OpenClaw` 代表了 persistent/open agent runtime 的兴起
3. 但 scientific / time-series 任务与 open-world agent 不同：它们是可验证、可比较、指标敏感的
4. 现有 time-series agents 主要提升“当前任务怎么做”，没有很好回答“agent 如何从探索中学习”
5. `TimeClaw` 的核心就是：
   - 强制探索
   - 路径比较
   - 外部蒸馏
   - 推理复用
6. 所以它既连接了时序 agent，也连接了新一代 agent systems 的更大趋势

一句话版本：

**OpenClaw tells us agents should persist. TimeClaw asks how a scientific agent should learn.**

---

## 5. 关键定义

这一部分最好在 PPT 里明确说，不然听众会混淆。

### 5.1 Prompt Engineering

新的定义说明：

- `prompt engineering` 主要是设计一轮输入里“说什么”
- 关注指令写法、角色设定、few-shot 示例、输出格式
- 核心对象是单轮或短链路交互

一句话：

**Prompt engineering 管的是这次对模型怎么说。**

### 5.2 Context Engineering

新的定义说明：

- `context engineering` 不只关心 prompt 文本，而是设计模型在当前步骤“看到什么”
- 包括：
  - system prompt
  - retrieved knowledge
  - memory
  - tool schema
  - examples
  - profile / metadata
  - history selection

一句话：

**Context engineering 管的是模型这一步知道什么。**

### 5.3 Harness Engineering

新的定义说明：

- `harness engineering` 指的是围绕 agent 的完整运行外壳设计
- 它不仅包括 context，还包括：
  - 执行循环
  - 约束
  - 权限
  - 验证
  - error recovery
  - orchestration
  - trace / audit
  - runtime state

一句话：

**Harness engineering 管的是 agent 在什么制度和运行环境里做事。**

重要边界：

- `harness engineering` 在 2026 年是快速升温的术语和研究方向
- 但它还不是一个像 supervised learning 那样完全稳定的标准学科名词
- 因此汇报里最稳的说法是：
  - `an emerging systems framing`
  - `a useful lens for understanding modern agents`

不要讲成：

- “已经完全成为标准教科书定义”
- “所有人都一致这样划分”

### 5.4 OpenClaw

新的定义说明：

- `OpenClaw` 不应只讲成某一个 repo
- 更适合讲成 2026 年兴起的一类 open/persistent personal agent runtime 生态
- 它强调：
  - 长时运行
  - 本地或半本地执行
  - skills / plugins / watchers
  - 外部状态
  - runtime safety / governance

一句话：

**OpenClaw 代表的是“agent 不再只是一次调用，而是一个持续运行的系统”。**

### 5.5 TimeClaw

新的定义说明：

- `TimeClaw` 是一个面向 time-series agent 的 external learning framework
- 学习发生在 `exploration phase`
- 推理发生在 `inference phase`
- 学习结果主要沉淀到模型参数之外的 artifact 中

一句话：

**TimeClaw 研究的不是 agent 会不会做题，而是 agent 能不能从探索中学会以后更好地做题。**

---

## 6. 14 页主报告设计

---

### Slide 1. 标题页

**标题**

TimeClaw: Benchmark-Grounded External Learning for Time-Series Agents

**副标题**

From Prompt to Harness, and from Execution to Learning

**这一页要传达的唯一信息**

这不是一篇普通的时序 agent 系统工作，而是一篇关于 scientific agent learning 的工作。

**建议视觉**

- 深色标题 + 一张抽象流程图背景
- 图里只保留四个词：
  - Explore
  - Compare
  - Distill
  - Reinject

**讲稿提示**

开场直接立题：

“今天我想讲的不是一个更会做时序任务的 agent，而是一个让 time-series agent 从探索中学习的框架。”

---

### Slide 2. Why Now: Agent Engineering Is Evolving

**标题**

Why now: AI engineering is moving from prompts to systems

**建议页内文案**

- `Prompt engineering`: optimize what we say to the model
- `Context engineering`: optimize what the model sees at each step
- `Harness engineering`: optimize the runtime that governs what the agent can do, remember, verify, and reuse

**这一页要传达的唯一信息**

Agent 的核心挑战已经不只是“怎么写 prompt”，而是“怎么设计整个运行系统”。

**建议视觉**

做一个三层递进图：

- Prompt
- Context
- Harness

每层下面放一句中文解释：

- 说什么
- 知道什么
- 在什么环境中行动

**讲稿提示**

这里一定要提醒听众：

“这个三段式是一个非常有用的分析框架，但尤其是 harness engineering，还属于快速形成中的前沿系统语言，不是完全固定的教科书分类。”

---

### Slide 3. What OpenClaw Represents

**标题**

OpenClaw and the rise of persistent open agents

**建议页内文案**

- OpenClaw-style systems represent a shift from stateless assistants to persistent runtimes
- Core traits:
  - long-lived state
  - skills/plugins/watchers
  - runtime governance and safety
  - continuous or recurring user interaction
- Key message:
  - the agent is no longer a prompt wrapper
  - it is an operating system-like runtime layer

**这一页要传达的唯一信息**

OpenClaw 代表的是 agent 从“一次性问答接口”走向“持续运行的系统”。

**建议视觉**

左边画传统 chat assistant。

右边画 persistent runtime：

- user
- memory/state
- skills/plugins
- safety/governance
- long-running loop

**讲稿提示**

可以这样讲：

“OpenClaw 的价值不在于某一个具体功能，而在于它把 agent 重新定义成一个持续、可扩展、可治理的 runtime。”

---

### Slide 4. But Scientific Agents Are Different

**标题**

Scientific agents are not open-world agents

**建议页内文案**

左列：Open-world agent

- open-ended goals
- ambiguous success criteria
- personalization and autonomy

右列：Scientific / time-series agent

- benchmark-grounded tasks
- task-valid outputs
- metric-based comparison
- auditable and reproducible evaluation

底部一句：

**TimeClaw focuses on the second setting.**

**这一页要传达的唯一信息**

TimeClaw 不做开放世界自动化，而是研究可验证 scientific benchmark 下的 agent learning。

**建议视觉**

左右对照表。

**讲稿提示**

这页是你后面所有 claim 的安全垫。一定要把任务边界说清。

---

### Slide 5. The Gap in Existing Time-Series Agents

**标题**

Existing time-series agents mostly optimize execution-time capability

**建议页内文案**

- TimeART, TS-Agent, and related work significantly improve:
  - tool use
  - reasoning
  - workflow orchestration
- But they mainly answer:
  - how to solve the current task better
- They less directly answer:
  - how to learn from exploratory execution and reuse that experience across tasks

**这一页要传达的唯一信息**

现有工作强在“当前任务求解”，而不是“跨任务经验学习”。

**建议视觉**

上半部分画单次任务求解流程。

下半部分画 TimeClaw 关心的闭环：

execution -> comparison -> distillation -> future reuse

**讲稿提示**

推荐一句：

“TimeART teaches a model to use tools for time-series reasoning; TimeClaw asks how an executing agent should learn from tool-augmented exploration.”

---

### Slide 6. The Core Problem

**标题**

For numeric tasks, completion is not enough

**建议页内文案**

在 forecasting / indicator / correlation / trend tasks 中：

- multiple paths can all be task-valid
- but their quantitative quality can differ substantially
- binary success is therefore too weak as a learning signal

公式可保留一条：

\[
\pi^\star(x) = \arg\max_{\pi_k \in \Pi_{\mathrm{valid}}(x)} q(\pi_k; x)
\]

并在旁边标注：

- `task-valid`: 满足输出契约
- `q`: 路径质量，由 `MAE/RMSE/accuracy/...` 决定

**这一页要传达的唯一信息**

TimeClaw 的核心不是“完成任务”，而是“学会哪条路径更优”。

**建议视觉**

两条都合法的路径：

- Path A: valid, MAE = 0.42
- Path B: valid, MAE = 0.17

底部一句：

**Both succeed. Only one should be learned.**

---

### Slide 7. TimeClaw at a Glance

**标题**

TimeClaw: Explore, Compare, Distill, Reinject

**建议页内文案**

四步闭环：

1. Explore: generate multiple candidate paths
2. Compare: evaluate task-valid candidates under task metrics
3. Distill: convert experience into external artifacts
4. Reinject: reuse learned artifacts at inference time

**这一页要传达的唯一信息**

TimeClaw 的方法对象是一个外部学习闭环。

**建议视觉**

这是全场最重要的一张图。

画一个 4-step 环形图。

每个阶段下面只放一个关键词：

- Explore: branching
- Compare: metric supervision
- Distill: artifacts
- Reinject: prompt reuse

---

### Slide 8. Exploration-Time Learning

**标题**

Learning happens in exploration, not in deployment

**建议页内文案**

- Training-time contract:
  - at least 2 task-valid candidate paths
  - genuine branching, not paraphrases
  - comparison before completion
  - `learning_summary` only after evaluation
- Inference-time rule:
  - no online adaptation
  - only reuse distilled artifacts

**这一页要传达的唯一信息**

TimeClaw 不是 online self-improving runtime，而是 exploration-trained, inference-reuse。

**建议视觉**

做一个上下分层图：

- 上：Exploration phase
- 下：Inference phase

用红色划掉：

- spawn_subagent
- evaluate_*

放在 inference 区域。

---

### Slide 9. Why Forced Exploration Matters

**标题**

Without forced exploration, the agent learns too little

**建议页内文案**

- A single successful path is often insufficient
- The agent must see alternatives to learn reusable preferences
- If prior path families already exist, TimeClaw still forces:
  - one prior-guided branch
  - one alternative branch

一句总结：

**TimeClaw turns execution into comparative evidence.**

**这一页要传达的唯一信息**

强制探索不是工程细节，而是学习成立的前提。

**建议视觉**

画一个错误流程和正确流程：

- 错误：single path -> success -> stop
- 正确：branch -> compare -> learn

---

### Slide 10. Tool-Prior Collapse and Tool Dropout

**标题**

Tool-prior collapse is a hidden failure mode

**建议页内文案**

- During exploration, agents quickly overcommit to familiar tools
- This reduces candidate diversity
- It weakens comparison and distillation

TimeClaw solution:

- scope-aware tool dropout
- overused tools become less visible during training-time sub-agent exposure
- task-critical / hinted tools are protected

**这一页要传达的唯一信息**

Tool dropout 在这里不是 regularization trick，而是恢复探索多样性的学习机制。

**建议视觉**

左图：

- tool usage collapsed on 1~2 tools

右图：

- usage distribution becomes flatter after dropout

如果 AI 能画图，就让它画一个 entropy 提升的示意图。

---

### Slide 11. External Learning Artifacts

**标题**

TimeClaw learns outside the model weights

**建议页内文案**

Artifact hierarchy:

- `SOUL`: fixed human-authored identity
- `Notes`: sample-level experience records
- `MEMORY`: structured reusable rules
- `TOOLS`: tool-specific usage guidance
- `SKILLS`: scope-local SOPs

强调一句：

**Notes are not the main runtime memory. They are distillation sources.**

**这一页要传达的唯一信息**

TimeClaw 的“学习”主要沉淀为可编辑、可审核、可复用的外部产物。

**建议视觉**

做一个金字塔或分层堆叠图。

底部：

- Notes

中间：

- MEMORY / TOOLS / SKILLS

顶部：

- Reused prompt context

---

### Slide 12. Conflict-Aware Memory and Prompt Injection

**标题**

Not all learned experience should enter the prompt

**建议页内文案**

Memory is not append-only:

- merge near-duplicates
- accumulate support
- register conflicts
- mark unstable items as non-injectable

Inference-time injection uses only:

- scope-matched
- condition-matched
- injectable rules

一句总结：

**TimeClaw separates storage from injection.**

**这一页要传达的唯一信息**

你们的 memory 不是“越存越多”，而是“长期保留 + 冲突消解 + 有条件注入”。

**建议视觉**

左边：

Notes -> Distill -> MEMORY state

右边：

MEMORY state -> filter -> prompt

旁边标红两个标签：

- stored
- injected

---

### Slide 13. Positioning: TimeClaw vs TimeART vs OpenClaw

**标题**

Where does TimeClaw sit?

**建议页内表格**

| Property | TimeART | OpenClaw-style runtimes | TimeClaw |
|---|---|---|---|
| Time-series agent setting | Yes | No / not primary | Yes |
| Persistent external state | Limited / not central | Yes | Yes |
| Learns from execution | Limited | Often yes | Yes |
| Path-level comparison | No | Usually no | Yes |
| Benchmark-grounded scientific supervision | Yes | No | Yes |
| Inference-time reuse without online adaptation | Not central | Mixed | Yes |

**这一页要传达的唯一信息**

TimeClaw 站在两条线的交叉处：

- 一边连接 time-series agents
- 一边连接 persistent agent systems

**建议视觉**

除了表格，还可以配一个二维定位图：

- 横轴：execution-time capability -> learning-from-execution
- 纵轴：open-world runtime -> benchmark-grounded scientific agent

把 `TimeART`、`OpenClaw`、`TimeClaw` 标出来。

---

### Slide 14. Takeaways

**标题**

Takeaways

**建议页内文案**

1. AI agents are moving from prompt design to system design.
2. OpenClaw shows why agents should persist.
3. TimeClaw asks how a scientific time-series agent should learn.
4. For metric-evaluable tasks, better-path learning is more appropriate than completion-only learning.
5. TimeClaw turns exploratory execution into reusable external intelligence.

**这一页要传达的唯一信息**

把全场收束成 3~5 句 reviewer 级别的结论。

**建议视觉**

纯文字页即可，但要大字、少字。

---

## 7. 备份页建议

建议额外准备下面 `5` 页 backup slides。

### Backup 1. 具体任务例子

内容：

- forecasting task
- two valid paths
- different MAE/RMSE

用途：

- 回答“为什么 completion-only 不够”

### Backup 2. Prompt Assembly

内容：

- official task description
- task boundary / output contract
- fingerprint / profiling
- SOUL + MEMORY
- tool notes + skills
- visible tool catalog

用途：

- 回答“你们推理时具体给模型看了什么”

### Backup 3. Memory Lifecycle

内容：

- Notes -> Distill -> MEMORY
- merge / support / conflict / non-injectable
- history snapshot

用途：

- 回答“memory 为什么不是简单 RAG”

### Backup 4. Tool Dropout Details

内容：

- scope-aware keep probability
- never_drop tools
- training-only exposure control

用途：

- 回答“tool dropout 到底是不是一个 trick”

### Backup 5. Experiment Plan

内容：

- evaluation on TimeART-aligned MTBench subset
- metrics: MAE / RMSE / accuracy
- ablations:
  - w/o path comparison
  - w/o tool dropout
  - w/o artifact reinjection
  - w/o MEMORY / TOOLS / SKILLS

用途：

- 回答“你们怎么验证方法”

---

## 8. 可直接喂给 AI 的生成指令

下面这段可以直接给 PPT 生成模型。

```text
Please generate a 14-slide academic presentation in a clean NeurIPS-style visual language.

Topic:
TimeClaw: Benchmark-Grounded External Learning for Time-Series Agents

Core thesis:
TimeClaw is not just another time-series agent. It turns benchmark-grounded time-series execution into a metric-supervised external learning process. Learning happens in a dedicated exploration phase through candidate-path comparison, and inference reuses distilled external artifacts without online adaptation.

Required narrative:
1. Start from the broader evolution of AI engineering:
   prompt engineering -> context engineering -> harness engineering
2. Briefly explain OpenClaw as a 2026-style persistent/open agent runtime ecosystem
3. Explain why scientific/time-series agents differ from open-world agents
4. Introduce the gap in current time-series agents: most improve execution-time capability, but not cross-task learning from exploratory execution
5. Introduce TimeClaw as an external learning framework
6. Emphasize metric-supervised path learning for numeric and verifiable tasks
7. Explain forced exploration and scope-aware tool dropout
8. Explain external artifacts: SOUL, Notes, MEMORY, TOOLS, SKILLS
9. Emphasize that Notes are distillation sources, not the main runtime memory
10. Explain conflict-aware memory updates and prompt-time injection filtering
11. Position TimeClaw against TimeART and OpenClaw-style runtimes
12. End with concise takeaways

Design constraints:
- academic, sharp, minimal, high-end
- avoid startup pitch aesthetics
- light background, dark text, teal accent
- one key message per slide
- use diagrams more than paragraphs
- use modern sans-serif typography
- keep slides uncluttered

Important factual tone:
- describe harness engineering as an emerging systems framing, not a universally fixed academic discipline
- describe OpenClaw as representing the rise of persistent open agents
- describe TimeClaw as a benchmark-grounded external learning framework for time-series agents
- do not call TimeClaw an online self-improving runtime

Need visuals:
- a 3-step evolution diagram: prompt -> context -> harness
- a comparison chart: open-world agents vs scientific agents
- a 4-stage cycle: Explore -> Compare -> Distill -> Reinject
- a path-quality illustration with two valid paths but different metric outcomes
- a tool-prior collapse vs dropout diagram
- an artifact hierarchy diagram
- a positioning table or 2D map for TimeART / OpenClaw / TimeClaw
```

---

## 9. 你汇报时最该避免的 6 个坑

### 9.1 不要把 TimeClaw 讲成“更强的工具系统”

应该讲：

- a learning framework

而不是：

- a tool-rich agent platform

### 9.2 不要把 OpenClaw 讲成你们的直接 baseline

应该讲：

- a neighboring systems paradigm

而不是：

- a direct task-level competitor

### 9.3 不要把 harness engineering 讲成稳定教科书概念

应该讲：

- an emerging systems framing

### 9.4 不要把 TimeClaw 讲成 online self-improving agent

应该讲：

- exploration-trained, inference-reuse

### 9.5 不要把 memory 讲成简单 memory bank

应该讲：

- conflict-aware structured external learning artifacts

### 9.6 不要把你们的创新点讲成“模块很多”

应该讲成下面三层：

1. problem formulation
2. learning mechanism
3. external reuse structure

---

## 10. 建议你在汇报中突出强调的 4 个贡献点

这 4 点最适合放在口头表达里反复出现。

### Contribution 1

**We formulate benchmark-grounded external learning for time-series agents.**

中文讲法：

“我们把时序 agent 的 benchmark 执行，上升成一个可审计、可蒸馏、可复用的外部学习过程。”

### Contribution 2

**We show that path-quality supervision is more suitable than completion-only supervision for metric-evaluable time-series tasks.**

中文讲法：

“对于 forecasting、indicator、correlation 这类任务，关键不是会不会完成，而是哪条路径更优。”

### Contribution 3

**We identify tool-prior collapse as a major obstacle to exploratory tool learning and mitigate it through scope-aware tool dropout.**

中文讲法：

“我们发现 agent 在学习时会迅速偏向少数熟悉工具，导致探索塌缩；tool-dropout 是为了解决这个结构性问题。”

### Contribution 4

**We externalize learning into conflict-aware, reusable artifacts rather than online parameter updates.**

中文讲法：

“我们的学习结果不是写回模型参数，而是沉淀成可以长期保留、冲突消解、按条件注入的外部知识产物。”

---

## 11. 事实性说明与引用建议

这部分是给你做讲稿或在答疑时使用的。

### 11.1 关于 OpenClaw

截至 `2026-04-16`，可以稳妥引用下面几篇来说明 OpenClaw 所代表的趋势：

- `OpenClaw-RL: Train Any Agent Simply by Talking`，`2026-03-10`
  - 强调通过真实交互信号进行持续学习
  - 链接：<https://arxiv.org/abs/2603.10165>
- `ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers`，`2026-03-25`
  - 强调 skills/plugins/watchers 与安全治理
  - 链接：<https://arxiv.org/abs/2603.24414>
- `A Systematic Security Evaluation of OpenClaw and Its Variants`，`2026-04-03`
  - 强调 persistent tool-augmented agents 的系统性安全风险
  - 链接：<https://arxiv.org/abs/2604.03131>

最稳的汇报说法：

**OpenClaw is best presented as a representative ecosystem of persistent open agents, rather than as a single isolated paper.**

### 11.2 关于 Context Engineering

截至 `2026-04-16`，`context engineering` 已经不只是社区 buzzword，而是有更明确论文化表达的方向：

- `CL4SE: Benchmarking Context Learning on Software Engineering`，`2026-02-27`
  - 链接：<https://arxiv.org/abs/2602.23047>
- `Context Engineering: A Practitioner Methodology for Structured Human-AI Collaboration`，`2026-04-06`
  - 链接：<https://arxiv.org/abs/2604.04258>

最稳的说法：

**Context engineering is already emerging as a more explicit test-time systems discipline, especially around what information the model should see at each step.**

### 11.3 关于 Harness Engineering

截至 `2026-04-16`，`harness engineering` 很热，但更准确地说，它处于“快速学术化”的阶段。

可以引用：

- `AutoHarness`，`2026-02-10`
  - 链接：<https://arxiv.org/abs/2603.03329>
- `Natural-Language Agent Harnesses`，`2026-03-26`
  - 链接：<https://arxiv.org/abs/2603.25723>
- `Meta-Harness`，`2026-03-30`
  - 链接：<https://arxiv.org/abs/2603.28052>
- `SemaClaw`，`2026-04-14`
  - 链接：<https://arxiv.org/abs/2604.11548>

最稳的说法：

**Harness engineering is an emerging framing for the runtime layer that governs agent execution, memory, verification, and control.**

不要讲成：

**Harness engineering is already a fully standardized academic subfield.**

---

## 12. 你如果想讲得更“高大上”，推荐的开场白

你可以直接这么讲：

> In 2026, the frontier of AI engineering is no longer just how to prompt a model, but how to build a runtime around it. OpenClaw-style systems show that agents are becoming persistent, stateful, and governable. Our question is what this evolution means for scientific agents, especially in time-series domains where outputs are verifiable and path quality matters. TimeClaw is our answer: a benchmark-grounded external learning framework that turns exploratory execution into reusable intelligence.

中文版本：

> 到了 2026 年，AI 工程的前沿问题已经不只是怎么写 prompt，而是怎么围绕模型构建一个可持续运行、可治理、可学习的 agent runtime。OpenClaw 代表了这种趋势。但我们的问题是：当 agent 进入 scientific setting，特别是 time-series 这种可验证、可比较、指标敏感的任务时，它应该如何学习？TimeClaw 就是在这个问题上的回答。

---

## 13. 如果你只想做 10 页极简版

可以压缩成：

1. Title
2. Prompt -> Context -> Harness
3. OpenClaw and persistent agents
4. Why scientific/time-series agents are different
5. Gap in existing time-series agents
6. TimeClaw overview
7. Path-quality supervision
8. Tool-prior collapse + tool dropout
9. External artifacts + inference reuse
10. Positioning + takeaways

---

## 14. 最后建议

如果你的目标是：

- 组会汇报
- 论文 pre-talk
- 给老师/合作者讲整个项目定位

那么这套 PPT 最重要的不是“把所有模块讲全”，而是让听众记住下面三句话：

1. `OpenClaw` 代表了 agent runtime 的兴起。
2. `TimeClaw` 研究的是 scientific setting 下 agent 如何从探索中学习。
3. 对于 metric-evaluable 的时序任务，学习“更优路径”比学习“是否完成”更合理。

