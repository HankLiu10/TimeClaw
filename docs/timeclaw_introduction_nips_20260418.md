# TimeClaw NeurIPS Introduction Draft

更新时间：2026-04-18

本文档是 Introduction 的更新版，基于 0416 版进行了以下修改：

- 新增 **external learning** 的精确定义段落，明确区分 ICL、meta-learning、continual learning、RAG
- 弱化 "benchmark-grounded" 的复合术语用法，改为自然使用
- 将 path-quality supervision 的对比对象从 RL reward shaping 修正为 agentic binary supervision
- tool-prior collapse 定义中增加 "prematurely" 限定，并区分 exploration/inference 阶段
- Table 1 列名更新
- 保留 broader positioning 段落不变

---

## 0. 先给结论

以 NeurIPS Spotlight 的标准看，TimeClaw 的 `Introduction` 应该被组织成这条主线：

**TimeClaw turns time-series execution into a metric-supervised external learning process for time-series agents. It learns during a dedicated exploration phase by comparing candidate execution paths under task-valid supervision, and improves downstream inference through reusable distilled artifacts without online adaptation.**

这句话里有 4 个必须保留的定义。

### 0.1 `exploration phase`

- `exploration phase` 指专门为了收集候选执行路径、比较路径质量、沉淀外部经验而运行的学习阶段；
- 这一阶段允许真实分支探索、候选比较和 GT/metric 验证；
- 它不是部署时的在线学习。

### 0.2 `inference phase`

- `inference phase` 指部署或测试时的求解阶段；
- 这一阶段只复用探索阶段蒸馏出的 artifact；
- 不再继续在线学习，也不再暴露训练专用的探索工具。

### 0.3 `task-valid candidate`

- `task-valid candidate` 不是"看起来像答案"的中间推理，而是满足当前任务输出契约的合法候选输出；
- 例如：forecasting 任务中的数值数组；indicator 任务中的结构化 JSON；trend / correlation 任务中的合法标签；QA 任务中的最终文本答案或选项标签。

### 0.4 `path-quality supervision`

- `path-quality supervision` 指对不同候选执行路径的监督，不是基于"是否完成任务"的二值信号，而是基于任务认可的质量指标；
- 对 numeric / verifiable tasks，这类监督通常来自 `MAE`、`RMSE`、accuracy 或任务特定误差；
- 它回答的是"哪条路径更优"，而不仅是"哪条路径可行"。
- **对比对象**：在 agentic 场景中，大多数现有系统（如 OpenClaw 风格的 agent）使用二值成功/失败信号来评估执行结果。TimeClaw 的 path-quality supervision 提供了更细粒度的学习信号。注意：这里的对比对象不是 RL 的 reward shaping——深度学习模型可以通过梯度直接优化损失函数，不需要"路径比较"；但 agent 无法对执行路径做反向传播，因此需要通过显式的路径质量比较来学习。

### 0.5 `external learning`

- `external learning` 指 TimeClaw 的学习结果主要沉淀在模型参数之外；
- 学习产物被组织为外部 artifact，例如 structured memory、tool notes、skills 和历史快照；
- 推理阶段通过 prompt reinjection 复用这些 artifact，而不是靠在线参数更新。
- **与 ICL 的区分**：in-context learning (ICL) 同样通过上下文注入来利用信息，但 ICL 直接使用给定的示例，没有主动的学习过程。TimeClaw 虽然最终也通过上下文注入来复用学习产物，但它有一个主动的学习过程（explore, compare, distill）来生成这些产物。换言之，ICL 的"学习"发生在 forward pass 中，而 TimeClaw 的学习发生在一个独立的探索阶段。
- **与 meta-learning 的区分**：meta-learning 通常学习一个可以快速适应新任务的参数初始化或学习规则；TimeClaw 不修改模型参数，而是生成外部 artifact。
- **与 continual learning 的区分**：continual learning 通常指模型在持续接收新数据时更新参数而不遗忘旧知识；TimeClaw 的学习产物存储在外部，不涉及参数更新。
- **与 RAG 的区分**：RAG 从外部知识库检索相关文档来增强生成；TimeClaw 的外部 artifact 不是预先存在的知识库，而是通过主动探索和比较学习生成的结构化经验。

---

## 1. 用 Spotlight 标准看，Introduction 应该解决什么

一个能冲 Spotlight 的 `Introduction`，至少要完成 5 件事。

### 1.1 先把问题抬高

不要把文章写成：

- 一个更强的多模态时序 agent
- 一个带更多工具的 benchmark system
- 一个更完整的 time-series workflow 平台

更好的问题定义应该是：

**Most current time-series agents can execute tasks, but they still lack a principled way to learn from exploratory execution and reuse that experience across tasks.**

### 1.2 再把缺口说尖锐

#### Gap 1: Existing time-series agents mainly optimize execution-time capability

现有系统大多关心当前任务如何调工具、当前样本如何推理、当前流程如何更自动化。但缺少 `execution -> comparison -> distillation -> future reuse` 的学习闭环。

#### Gap 2: Numeric time-series tasks need better-path learning, not only task completion

在 forecasting、indicator prediction、trend estimation、correlation reasoning 这类任务中，往往不止一条执行路径，多条路径都能产生合法答案，但这些路径在 `MAE`、`RMSE` 等指标上可能差异很大。

**the agent should learn which path is better, not only whether a path finishes the task**

注意：这里的对比对象是 agentic 场景中的二值监督（成功/失败），而不是 RL 的 reward shaping。深度学习模型可以通过梯度直接优化损失函数；但 agent 无法对执行路径做反向传播，因此需要显式的路径质量比较。

#### Gap 3: Learning collapses when the agent prematurely overcommits to familiar tools

- `tool-prior collapse`：agent 在探索阶段**过早地**对少数工具形成过强偏置，导致候选路径多样性下降，进而削弱后续比较学习和蒸馏质量。
- 关键限定：这一现象特指**探索阶段**。在推理阶段，agent 基于已学习的经验偏好某些工具是合理的——这是学习的结果，不是 collapse。Collapse 之所以是问题，是因为它发生在探索阶段，此时 agent 尚未积累足够的比较证据就过早收敛。

### 1.3 再把方法学贡献讲成"范式"，不是"拼装"

更好的写法是把所有实现层元素统一收束到这条方法学主线：

**TimeClaw turns time-series execution into a metric-supervised external learning loop.**

### 1.4 最后要让 reviewer 记住一句话

**For metric-evaluable time-series tasks, TimeClaw learns from path quality rather than completion alone, and externalizes that learning into reusable artifacts.**

### 1.5 所有 claim 都要高但稳

更稳的说法：external learning, exploration-trained, metric-supervised, learning-enabled, without online adaptation.

---

## 2. TimeClaw 最适合的论文定位

### 2.1 第一层定位：time-series agent learning

**TimeClaw is an external learning framework for time-series agents.**

### 2.2 第二层定位：scientific-domain instantiation of learning-enabled agents

**TimeClaw is a scientific-domain instantiation of persistent, learning-enabled agents.**

### 2.3 TimeClaw 和 `tool learning` 的关系

**TimeClaw intersects with tool learning, but goes beyond tool selection or tool-use correctness by studying metric-supervised path learning and externalized experience reuse in time-series agents.**

---

## 3. 最适合写进论文开头的主问题

**How can a time-series agent learn from exploratory executions collected during a dedicated learning phase, rather than merely solving each task independently at inference time?**

This question is particularly important for numeric and verifiable time-series tasks, where multiple candidate execution paths may all produce task-valid answers, yet differ substantially in quantitative quality under metrics such as MAE or RMSE.

---

## 4. 最应该打透的三层创新

### 4.1 主创新 A：外部化学习闭环

**TimeClaw defines an external learning loop for time-series agents: Explore -> Compare -> Distill -> Reinject.**

### 4.2 主创新 B：数值任务上的 metric-supervised path learning

**TimeClaw shifts the agent learning objective from task completion to path-quality optimization for metric-evaluable tasks.**

对比对象：在 agentic 场景中，大多数现有系统使用二值成功/失败信号。深度学习模型可以通过梯度直接优化损失函数，不需要路径比较；但 agent 无法对执行路径做反向传播，因此需要通过显式的路径质量比较来学习。

### 4.3 主创新 C：tool-prior collapse 与 scope-aware tool dropout

**We identify tool-prior collapse — the premature concentration on familiar tools during exploration before sufficient comparative evidence is gathered — as a structural barrier to exploratory tool learning, and mitigate it via forced exploration with scope-aware tool dropout.**

### 4.4 支撑创新 D：探索学习数据与外部产物证据

**We construct a dedicated exploratory learning setup for time-series agents, enabling controlled study of path diversity, path comparison, and reusable artifact distillation.**

---

## 5. Introduction 最推荐的叙事顺序

### Paragraph 1: 大背景 + 大问题

- LLM agents 正从一次性系统走向带工具、带状态、带外部记忆的系统；
- time-series scientific tasks 很适合 agent，但现有工作主要提升执行能力；
- 真正缺的是从执行中学习并跨任务复用经验。

### Paragraph 2: 为什么数值型时序任务更难

- binary success 不够；
- 多条路径都可能产出合法答案；
- 关键是 quantitative quality；
- 因而需要 path-level comparative learning。
- 对比对象是 agentic binary supervision，不是 RL reward shaping。

### Paragraph 3: 为什么学习本身会失败

- exploration 不稳定；
- agent **过早地**对少数工具形成过强偏置（tool-prior collapse）；
- 候选集一旦塌缩，后续 comparison 和 distillation 都会失效。
- 注意区分：探索阶段的 entropy 下降 = collapse（因为探索目标是工具覆盖）；推理阶段的工具偏好是学习结果，不是 collapse。

### Paragraph 4: 你们提出了什么

- TimeClaw 是 external learning framework；
- 学习发生在 dedicated exploration phase；
- 流程是 `Explore -> Compare -> Distill -> Reinject`；
- inference 只复用蒸馏产物，不做在线学习。

### Paragraph 5: 核心机制 + broader positioning + contributions

---

## 6. 最推荐写进论文的贡献组织

1. **We formulate external learning for time-series agents, and instantiate it in TimeClaw as a learning architecture that separates exploration-time experience acquisition from inference-time reuse.**

2. **We identify path-quality supervision as a more suitable learning signal than completion-only supervision for numeric and verifiable time-series tasks, and operationalize this insight through metric-supervised path learning.**

3. **We identify tool-prior collapse — the premature concentration on familiar tools during exploration — as a major obstacle to exploratory tool learning and propose forced exploration with scope-aware tool dropout to improve candidate-path diversity and supervision quality.**

4. **We build a dedicated exploratory learning setup for time-series agents and show that distilled external artifacts improve downstream reasoning without parameter updates or online adaptation.**

---

## 7. 放进正文 Introduction 的直观区分表

| Work | Tool-Augmented Execution | Exploration-Time Learning | External Artifact Learning | Path-Level Comparative Learning | Cross-Task Reuse |
| --- | --- | --- | --- | --- | --- |
| TimeART | ✓ |  | △ |  |  |
| TS-Agent | ✓ |  |  |  |  |
| TimeSeriesScientist | ✓ |  |  |  |  |
| Cast-R1 | ✓ | △ | △ | △ | △ |
| MemSkill / ASDA / Skill-SD | △ |  | ✓ |  | ✓ |
| OpenClaw-style runtimes | ✓ |  | ✓ |  | ✓ |
| **TimeClaw** | **✓** | **✓** | **✓** | **✓** | **✓** |

说明：

- `✓`：该维度是该工作的明确主目标或核心设计点
- `△`：该维度部分涉及，但不是其主方法学中心
- Table 1 列名从 "Benchmark-Grounded Learning" 改为 "Exploration-Time Learning"，弱化 benchmark- 复合术语

建议英文表题：

**Table 1: Positioning TimeClaw against representative neighboring lines of work. TimeClaw is distinguished not by having more components, but by jointly combining exploration-time learning, external artifact distillation, path-level comparative learning, and cross-task reuse in a time-series agent setting.**

---

## 8. 可直接放进论文的英文 Introduction 草稿

---

## **Introduction**

Large language model (LLM) agents are evolving from one-shot assistants into persistent systems that interact with tools, maintain external state, and support increasingly complex workflows. This trend is especially promising for scientific domains such as time-series analysis, where solving a task often requires combining statistical computation, contextual reasoning, and structured decision making. Recent work has made substantial progress on time-series benchmarks and tool-augmented time-series agents [MTBench; TemporalBench; TimeART; TS-Agent; TimeSeriesScientist]. However, most existing time-series agent systems still focus primarily on **execution-time capability**: they improve how an agent solves the current task, but only partially address how experience from exploratory execution should be accumulated and reused across future tasks.

This limitation is particularly important for numeric and verifiable time-series tasks. In forecasting, indicator prediction, and related settings, multiple execution paths may all produce task-valid answers, yet differ substantially in quantitative quality under metrics such as MAE or RMSE. In such cases, binary notions of success are insufficient for learning. Most existing agentic systems evaluate execution outcomes through completion-only supervision — a binary signal indicating whether the task was solved — but this collapses quantitatively unequal paths into the same label. The question is not merely whether an agent can complete a task, but whether it can identify, preserve, and reuse **better execution paths**. Note that this challenge is specific to the agentic setting: deep learning models can directly optimize a loss function via gradient descent, but agents cannot backpropagate through their execution paths. This suggests that agent learning in time-series domains should move beyond completion-only supervision toward **path-quality-aware learning**, where the learning signal reflects quantitative differences between valid candidate paths [TimeART; TS-Agent; TimeSeriesScientist; Cast-R1].

A second challenge is that exploratory learning itself is unstable. During training, an agent often prematurely develops a strong bias toward a small set of familiar tools that frequently succeed early, then overcommits to them and suppresses further exploration. We refer to this failure mode as **tool-prior collapse**: the premature concentration of tool usage on a narrow subset before sufficient comparative evidence has been gathered. Once this happens, candidate diversity shrinks, comparative supervision weakens, and external distillation becomes less informative. It is important to distinguish this from inference-time tool preference: after learning, an agent may reasonably favor certain tools based on accumulated evidence — that is a consequence of learning, not a failure. Tool-prior collapse is specifically an exploration-phase pathology, where entropy in the tool-usage distribution decreases before the agent has explored broadly enough to justify convergence.

In this paper, we propose **TimeClaw**, an external learning framework for time-series agents. Rather than treating each task instance as an isolated execution problem, TimeClaw turns exploration-time execution into a reusable source of scientific experience. Its core design is an external learning loop: the agent explores multiple candidate execution paths, compares them under task-valid supervision, distills the resulting experience into editable external artifacts, and reinjects those artifacts into future problem solving. In this way, execution becomes not only an evaluation target, but also a metric-supervised learning process. Crucially, TimeClaw separates learning from inference: learning happens during a dedicated exploration phase, while downstream inference reuses distilled artifacts without further online adaptation.

We use the term **external learning** to describe this paradigm. In external learning, the learning process occurs in a dedicated exploration phase; the results are deposited into external artifacts (structured memory, tool guidance, reusable skills) rather than model parameters; and inference reuses these artifacts via prompt injection without further learning. This distinguishes external learning from several superficially related paradigms. **In-context learning (ICL)** also injects information through the context window at inference time, but ICL directly uses whatever examples are provided — it has no active learning process that generates the injected content. TimeClaw shares the injection mechanism with ICL but adds an active explore-compare-distill loop that produces the artifacts. **Meta-learning** typically learns a parameter initialization or learning rule that enables fast adaptation to new tasks; TimeClaw does not modify model parameters. **Continual learning** updates model parameters on a stream of new data while mitigating catastrophic forgetting; TimeClaw stores all learning outcomes externally. **Retrieval-augmented generation (RAG)** retrieves from a pre-existing knowledge base to augment generation; TimeClaw's external artifacts are not pre-existing but are actively constructed through exploratory execution and comparative learning.

TimeClaw is built around three key ideas. First, we introduce **metric-supervised path learning** for numeric and verifiable time-series tasks, enabling the agent to learn not only whether a task is solvable, but which candidate path achieves better quantitative outcomes. Second, to mitigate tool-prior collapse, we propose a **forced exploration protocol** with **scope-aware tool dropout**, which improves exploratory tool learning by reducing overreliance on dominant tools and increasing candidate-path diversity. Third, we externalize learning through a hierarchical artifact stack that stores reusable experience outside model weights, including path-level lessons, tool guidance, and higher-level reusable skills. In this sense, TimeClaw intersects with tool learning, but goes beyond tool-use correctness by learning from comparative path quality and reusing distilled experience across tasks [Toolformer; Gorilla; ToolLLM; MemSkill; ASDA; Skill-SD]. [Add one final experimental teaser sentence here once results are fixed.]

More broadly, TimeClaw can be viewed as a scientific-domain instantiation of the broader move from stateless tool use toward persistent, learning-enabled agent systems. Unlike open-world agent runtimes that emphasize continuous operation, personalization, or computer-use autonomy, TimeClaw specializes this broader systems philosophy to a verifiable scientific setting, where experience accumulation is comparable, verifiable, and reusable. This positioning allows us to study not only whether a time-series agent can solve benchmark tasks, but whether it can **learn from exploratory execution in a measurable and interpretable way** [OpenClaw-RL; CUA-Skill; ClawKeeper].

Our contributions are summarized as follows:

1. We formulate external learning for time-series agents, and instantiate it in TimeClaw as a learning architecture that separates exploration-time experience acquisition from inference-time reuse.
2. We identify path-quality supervision as a more suitable learning signal than completion-only supervision for numeric and verifiable time-series tasks, and operationalize this insight through metric-supervised path learning.
3. We identify tool-prior collapse — the premature concentration on familiar tools during exploration before sufficient comparative evidence is gathered — as a major obstacle to exploratory tool learning and propose forced exploration with scope-aware tool dropout to improve candidate-path diversity and supervision quality.
4. We build a dedicated exploratory learning setup for time-series agents and show that distilled external artifacts improve downstream reasoning without parameter updates or online adaptation.

**Table 1: Positioning TimeClaw against representative neighboring lines of work.** TimeClaw is distinguished not by having more components, but by jointly combining exploration-time learning, external artifact distillation, path-level comparative learning, and cross-task reuse in a time-series agent setting.

| Work | Tool-Augmented Execution | Exploration-Time Learning | External Artifact Learning | Path-Level Comparative Learning | Cross-Task Reuse |
| --- | --- | --- | --- | --- | --- |
| TimeART | ✓ |  | △ |  |  |
| TS-Agent | ✓ |  |  |  |  |
| TimeSeriesScientist | ✓ |  |  |  |  |
| Cast-R1 | ✓ | △ | △ | △ | △ |
| MemSkill / ASDA / Skill-SD | △ |  | ✓ |  | ✓ |
| OpenClaw-style runtimes | ✓ |  | ✓ |  | ✓ |
| **TimeClaw** | **✓** | **✓** | **✓** | **✓** | **✓** |

✓ = core design target; △ = partially addressed but not the main methodological focus.

---

## Citation Notes For The English Draft

如果后面转成 LaTeX，建议把方括号里的标题替换成 bibkey。当前推荐挂的引用如下。

- `MTBench` — `MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering` — https://arxiv.org/abs/2503.16858
- `TemporalBench` — `TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks` — https://arxiv.org/abs/2602.13272
- `TimeART` — `TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation` — https://arxiv.org/abs/2601.13653
- `TS-Agent` — `TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering` — https://arxiv.org/abs/2510.07432
- `TimeSeriesScientist` — `TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis` — https://arxiv.org/abs/2510.01538
- `Cast-R1` — `Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting` — https://arxiv.org/abs/2602.13802
- `Toolformer` — `Toolformer: Language Models Can Teach Themselves to Use Tools` — https://arxiv.org/abs/2302.04761
- `Gorilla` — `Gorilla: Large Language Model Connected with Massive APIs` — https://arxiv.org/abs/2305.15334
- `ToolLLM` — `ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs` — https://arxiv.org/abs/2307.16789
- `MemSkill` — `MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents` — https://arxiv.org/abs/2602.02474
- `ASDA` — `ASDA: Automated Skill Distillation and Adaptation` — https://arxiv.org/abs/2603.16112
- `Skill-SD` — `Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents` — https://arxiv.org/abs/2604.10674
- `OpenClaw-RL` — `OpenClaw-RL: Train Any Agent Simply by Talking` — https://arxiv.org/abs/2603.10165
- `CUA-Skill` — `CUA-Skill: Develop Skills for Computer Using Agent` — https://arxiv.org/abs/2601.21123
- `ClawKeeper` — `ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers` — https://arxiv.org/abs/2603.24414
