# TimeClaw NeurIPS Spotlight Introduction Draft

更新时间：2026-04-16

这份文档的目标不是继续“润色一版 intro”，而是把 `TimeClaw` 的 `Introduction` **完整重组**成更接近 NeurIPS Spotlight 稿件的结构。

这里的 `Spotlight 标准` 不是官方评分表，而是一套更实际的写作要求：

- reviewer 在前 1 页内就能看懂你在研究什么；
- reviewer 在前 2 页内就能判断这不是“又一个时序 agent 系统”；
- 贡献不是功能清单，而是一个清晰的研究故事；
- 读者能记住一句主结论，而不是一堆模块名；
- 风格足够高，但 claim 足够稳，不会因为措辞过猛被反打。

---

## 0. 先给结论

以 NeurIPS Spotlight 的标准看，TimeClaw 的 `Introduction` 应该被组织成这样一条主线：

**TimeClaw turns benchmark-grounded time-series execution into a metric-supervised external learning process for time-series agents. It learns during a dedicated exploration phase by comparing candidate execution paths under task-valid supervision, and improves downstream inference through reusable distilled artifacts without online adaptation.**

这句话里有 4 个必须保留的定义。

### 0.1 `benchmark-grounded`

这里指学习不是发生在完全开放、无法验证的环境里，而是发生在：

- 有任务定义；
- 有 ground truth 或有效评价标准；
- 有路径间可比较监督；
- 有可重复实验的 benchmark setting。

### 0.2 `external learning`

这里指学习结果主要不写回模型参数，而是沉淀到外部产物中，例如：

- path-level experience
- structured memory
- tool guidance
- reusable skills
- distilled cross-task patterns

这些产物在后续任务中被重新注入 agent。

### 0.3 `dedicated exploration phase`

这里指 TimeClaw 的学习发生在一个**专门的探索阶段**。  
新的定义说明：

- exploration phase：为了收集候选路径、比较路径质量、蒸馏经验而进行的学习阶段；
- inference phase：部署或测试阶段，只复用探索阶段沉淀出的外部产物，不继续在线学习。

这一定义很重要，因为它决定了你们不应该把自己写成强意义上的 `self-improving online agent`。

### 0.4 `task-valid supervision`

这里指路径比较的监督信号不是泛泛的“看起来合理”，而是任务本身认可的标准，例如：

- forecasting 的 `MAE` / `RMSE`
- classification 的正确率
- indicator / correlation task 的目标误差
- benchmark 提供的明确答案或可验证规则

---

## 1. 用 Spotlight 标准看，Introduction 应该解决什么

一个能冲 Spotlight 的 `Introduction`，至少要完成 5 件事。

### 1.1 先把问题抬高

不要把文章写成：

- 一个更强的多模态时序 agent
- 一个带更多工具的 benchmark system
- 一个更完整的 time-series workflow 平台

这些写法都太像系统论文。

更好的问题定义应该是：

**Most current time-series agents can execute tasks, but they still lack a principled way to learn from exploratory execution and reuse that experience across tasks.**

这样主问题就从“怎么做题”抬升成：

- agent learning
- externalized experience reuse
- verifiable scientific-domain learning

### 1.2 再把缺口说尖锐

TimeClaw 的 `Introduction` 不应该只说“现有工作还不够好”，而应该直接指出三类更本质的缺口。

#### Gap 1: Existing time-series agents mainly optimize execution-time capability

现有系统大多关心：

- 当前任务如何调工具；
- 当前样本如何推理；
- 当前流程如何更自动化。

但缺少：

- `execution -> comparison -> distillation -> future reuse` 的学习闭环。

#### Gap 2: Numeric time-series tasks need better-path learning, not only task completion

这是 TimeClaw 最关键的学术点之一。

在 forecasting、indicator prediction、trend estimation、correlation reasoning 这类任务中：

- 往往不止一条执行路径；
- 多条路径都能产生合法答案；
- 但这些路径在 `MAE`、`RMSE` 等指标上可能差异很大。

因此：

**the agent should learn which path is better, not only whether a path finishes the task**

这比一般 agent work 里的 `success / failure` 监督更强，也更贴合数值任务本身。

#### Gap 3: Learning collapses when the agent overcommits to familiar tools

这是你们最重要的机制动机之一。

新的定义说明：

- `tool-prior collapse`：本文在当前 benchmark-grounded time-series setting 下使用的术语，指 agent 在探索训练中由于某些工具更熟悉、更常成功，迅速对少数工具形成过强偏置，导致候选路径多样性下降，进而削弱后续比较学习和蒸馏质量。

这不是简单的 prompt 偏好，而是探索动作空间中的结构性偏置。

### 1.3 再把方法学贡献讲成“范式”，不是“拼装”

Spotlight 不喜欢功能罗列。  
所以 `Introduction` 不能把下面这些直接平铺：

- multimodal input
- memory
- notes
- skills
- tool notes
- dropout
- candidate paths

这些都是实现层元素，不是研究问题本身。

更好的写法是把它们统一收束到这条方法学主线：

**TimeClaw turns benchmark-grounded time-series execution into a metric-supervised external learning loop.**

### 1.4 最后要让 reviewer 记住一句话

如果 reviewer 看完 `Introduction` 只能记住一句话，我建议是：

**For metric-evaluable time-series tasks, TimeClaw learns from path quality rather than completion alone, and externalizes that learning into reusable artifacts.**

这句话比“我们是一个更强的时序 agent”更有论文味。

### 1.5 所有 claim 都要高但稳

NeurIPS reviewer 对这些词会比较敏感：

- first
- SOTA
- general
- fully autonomous
- self-improving

其中 `self-improving` 对你们尤其危险，因为它很容易被理解为测试时在线学习。

更稳的说法应该是：

- benchmark-grounded
- external learning
- exploration-trained
- metric-supervised
- learning-enabled
- without online adaptation

---

## 2. TimeClaw 最适合的论文定位

TimeClaw 最好不要被写成“another time-series agent”。  
更强、更稳的定位有两层。

### 2.1 第一层定位：time-series agent learning

最直接的写法是：

**TimeClaw is an external learning framework for time-series agents.**

这句话的重点不在“agent 能做什么”，而在：

- agent 怎样探索；
- agent 怎样比较候选路径；
- agent 怎样蒸馏经验；
- agent 怎样把经验稳定迁移到后续推理。

### 2.2 第二层定位：scientific-domain instantiation of learning-enabled agents

如果要把格局抬高，可以再往上写成：

**TimeClaw is a scientific-domain instantiation of persistent, learning-enabled agents.**

这里的意思不是说你们和 OpenClaw 技术上完全同源，而是：

- OpenClaw-style systems代表开放世界、长生命周期、带技能和外部状态的 agent runtime；
- TimeClaw 则把这种“持久化、可积累、可复用”的思想放到了可验证的 scientific benchmark 里。

你们和这类系统共享的是更高层的系统哲学：

- persistence
- external state
- reusable skills
- long-horizon accumulation

但你们的目标不是 open-world automation，而是：

**verifiable experience accumulation under benchmark supervision**

这个区分必须写清楚。

### 2.3 TimeClaw 和 `tool learning` 的关系

这个概念值得融入 `Introduction`，但不适合成为全文的总标签。

新的定义说明：

- `tool learning`：agent 学习何时、为何以及如何调用外部工具来完成任务；
- `exploratory tool learning`：agent 不只学会调用工具，还要在探索阶段形成多样化候选路径，并从路径比较结果中学习。

TimeClaw 和这条线关系很近，因为你们确实研究：

- agent 如何在 time-series 任务中调用工具；
- 为什么 agent 会对少数熟悉工具形成过强先验；
- 如何通过 `scope-aware tool dropout` 改善工具探索质量。

但 TimeClaw 又不应被简单写成一篇 `tool learning paper`，因为你们真正更高层的问题是：

- 不只是学“该不该调工具”；
- 而是学“哪条由工具参与形成的执行路径 quantitatively 更优”；
- 并把这种学习外部化为可复用 artifact。

所以更准确的写法是：

**TimeClaw intersects with tool learning, but goes beyond tool selection or tool-use correctness by studying metric-supervised path learning and externalized experience reuse in time-series agents.**

---

## 3. 最适合写进论文开头的主问题

我建议把主问题直接写成下面这个版本：

**How can a time-series agent learn from exploratory executions collected during a dedicated learning phase, rather than merely solving each task independently at inference time?**

紧跟着补一句解释：

This question is particularly important for numeric and verifiable time-series tasks, where multiple candidate execution paths may all produce task-valid answers, yet differ substantially in quantitative quality under metrics such as MAE or RMSE.

这两句出来之后，论文的主语就变成了：

- a learning problem
- a time-series scientific reasoning problem
- a metric-supervised agent training problem

这会明显强于“我们做了一个多模态时序系统”。

---

## 4. 最应该打透的三层创新

你们前面讨论过很多贡献，但如果按 Spotlight 标准组织，真正应该写成一级贡献的只有三层主创新和一层资源支撑。

### 4.1 主创新 A：外部化学习闭环

这是最大的帽子。

**TimeClaw defines a benchmark-grounded external learning loop for time-series agents: Explore -> Compare -> Distill -> Reinject.**

这条贡献说明的不是一个模块，而是一种学习框架。

它回答的是：

- 学习发生在哪里；
- 学习信号来自哪里；
- 学习结果存在哪里；
- 推理时如何使用这些结果。

### 4.2 主创新 B：数值任务上的 metric-supervised path learning

这是最有机会被审稿人当作“方法论推进”的点。

你们真正重要的判断是：

- 对许多数值时序任务而言，`completion-only supervision` 不够；
- 路径应该按 quantitative quality 来比较；
- 训练目标不应只是“完成”，而应是“找到更优路径并保留下来”。

这可以概括成一句话：

**TimeClaw shifts the agent learning objective from task completion to path-quality optimization for metric-evaluable tasks.**

如果这个点实验上打得实，它完全可以作为核心创新点之一。

### 4.3 主创新 C：tool-prior collapse 与 scope-aware tool dropout

这是最强的机制创新。

它不是一个普通训练 trick，而是：

- 先识别了一个结构性障碍；
- 再提出有针对性的探索干预。

推荐的高层表述：

**We identify tool-prior collapse in benchmark-grounded time-series agent learning as a structural barrier to exploratory tool learning, and mitigate it via forced exploration with scope-aware tool dropout.**

这句话的优点是：

- 先给问题命名；
- 再给解决机制；
- 不会把 `tool-dropout` 写成一个零散技巧。

### 4.4 支撑创新 D：探索学习数据与外部产物证据

如果你们已经构造了专门服务探索阶段的数据集或 protocol，这条贡献就非常有价值。

更合适的写法不是“顺便造了个数据集”，而是：

**We construct a dedicated exploratory learning setup for time-series agents, enabling controlled study of path diversity, path comparison, and reusable artifact distillation.**

它的价值在于：

- 让方法不只是概念；
- 让学习能力可评估；
- 让外部产物的作用可验证；
- 让论文更像“方法 + protocol + evidence”。

---

## 5. Introduction 最推荐的叙事顺序

真正投 NeurIPS 时，`Introduction` 最好按 5 段来写，而不是想到哪说到哪。

### Paragraph 1: 大背景 + 大问题

写：

- LLM agents 正从一次性系统走向带工具、带状态、带外部记忆的系统；
- time-series scientific tasks 很适合 agent，但现有工作主要提升执行能力；
- 真正缺的是从执行中学习并跨任务复用经验。

这一段的目标是把问题抬高。

### Paragraph 2: 为什么数值型时序任务更难

写：

- binary success 不够；
- 多条路径都可能产出合法答案；
- 关键是 quantitative quality；
- 因而需要 path-level comparative learning。

这一段的目标是把你们的“metric-supervised path learning”变成必然问题，而不是任选设计。

### Paragraph 3: 为什么学习本身会失败

写：

- exploration 不稳定；
- strong tool prior 会导致路径多样性塌缩；
- 候选集一旦塌缩，后续 comparison 和 distillation 都会失效。

这一段的目标是给 `tool-prior collapse` 和 `tool-dropout` 一个足够强的动机，并把它们明确放进 `exploratory tool learning` 的问题框架下。

### Paragraph 4: 你们提出了什么

写：

- TimeClaw 是 benchmark-grounded external learning framework；
- 学习发生在 dedicated exploration phase；
- 流程是 `Explore -> Compare -> Distill -> Reinject`；
- inference 只复用蒸馏产物，不做在线学习。

这一段的目标是准确、清楚、去误解。

### Paragraph 5: 核心机制 + broader positioning + contributions

写：

- metric-supervised path learning
- forced exploration + scope-aware tool dropout
- hierarchical external artifacts
- 与 broader open agent runtime 的关系
- contribution bullets

如果实验结果已经出来，这一段里最好加一行结果 teaser。

新的定义说明：

- `result teaser`：在 `Introduction` 后半段用一句非常短的话给出最关键的实验结论或数量级提升，目的是向 reviewer 证明这不只是 framing，而是真有结果支撑。

例如：

`Across MTBench and related time-series tasks, TimeClaw consistently improves downstream performance and path quality over strong tool-augmented baselines, with especially large gains on metric-evaluable forecasting settings.`

等你们实验出结果后，这句应该换成带数字的版本。

---

## 6. 最推荐写进论文的贡献组织

如果你们最后只保留 4 条 contribution，我建议就是下面这版。

1. **We formulate benchmark-grounded external learning for time-series agents, and instantiate it in TimeClaw as a learning architecture that separates exploration-time experience acquisition from inference-time reuse.**

2. **We identify path-quality supervision as a more suitable learning signal than completion-only supervision for numeric and verifiable time-series tasks, and operationalize this insight through metric-supervised path learning.**

3. **We identify tool-prior collapse in benchmark-grounded time-series agent learning as a major obstacle to exploratory tool learning and propose forced exploration with scope-aware tool dropout to improve candidate-path diversity and supervision quality.**

4. **We build a dedicated exploratory learning setup for time-series agents and show that distilled external artifacts improve benchmark-grounded reasoning without parameter updates or online adaptation.**

这 4 条的组织逻辑是：

- 第 1 条定义框架；
- 第 2 条定义新的学习目标；
- 第 3 条解决关键训练障碍；
- 第 4 条提供证据与实验支撑。

这会比“memory / skills / notes / tools / dropout / data”横向平铺强很多。

---

## 7. 放进正文 Introduction 的直观区分表

这张表**应该放进正文**。  
它的作用不是取代 `Related Work`，而是帮助 reviewer 在前两页就迅速看懂：

- TimeClaw 不只是又一个 time-series agent；
- 它和 `TimeART`、`TS-Agent`、`TimeSeriesScientist`、`Cast-R1`、`MemSkill/ASDA` 这些最接近的工作到底差在哪。

对一篇想冲 NeurIPS Spotlight 的文章来说，这张表的价值很高，因为它能在正文早期直接完成两件事：

- 压缩口头比较成本；
- 把 TimeClaw 从“系统堆叠”快速拉到“方法学定位”。

这张表不应该太大，也不应该写成产品对比卡。  
最适合 intro 的维度，我建议保留下面 5 列：

- `Tool-Augmented Execution`
- `Benchmark-Grounded Learning`
- `External Artifact Learning`
- `Path-Level Comparative Learning`
- `Cross-Task Reuse`

推荐表格如下。

| Work | Tool-Augmented Execution | Benchmark-Grounded Learning | External Artifact Learning | Path-Level Comparative Learning | Cross-Task Reuse |
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

这张表最该传达的不是“TimeClaw 什么都有”，而是：

**TimeClaw is unusual in jointly combining benchmark-grounded learning, external artifact learning, path-level comparative learning, and cross-task reuse within a time-series agent setting.**

### 建议正文放置位置

这张表建议**实际放进正文**的两个位置之一：

- `Introduction` 末尾、contribution 后面；或者
- `Introduction` 最后一段之后、`Related Work` 之前。

对你们这篇 paper，我更推荐：

- 放在 `Introduction` 末尾；
- 紧跟 contribution bullets 之后；
- 作为 reviewer 在进入 `Related Work` 前的最后一个定位锚点。

不要放在论文最开头，否则会显得像 summary table，而不是问题驱动型 introduction。

### 建议英文表题

如果你们直接放进正文，我建议 caption 写成：

**Table 1: Positioning TimeClaw against representative neighboring lines of work. TimeClaw is distinguished not by having more components, but by jointly combining benchmark-grounded learning, external artifact distillation, path-level comparative learning, and cross-task reuse in a time-series agent setting.**

---

## 8. 可直接放进论文的英文 Introduction 草稿

下面这版比之前更接近真正投稿时可用的 opening。  
它的目标不是文学性，而是：

- 主问题大；
- 动机硬；
- 创新集中；
- train / inference 边界清楚；
- 适合后续接 `Related Work` 和 `Methodology`。

---

## **Introduction**

Large language model (LLM) agents are evolving from one-shot assistants into persistent systems that interact with tools, maintain external state, and support increasingly complex workflows. This trend is especially promising for scientific domains such as time-series analysis, where solving a task often requires combining statistical computation, contextual reasoning, and structured decision making. Recent work has made substantial progress on time-series benchmarks and tool-augmented time-series agents [MTBench; TemporalBench; TimeART; TS-Agent; TimeSeriesScientist]. However, most existing time-series agent systems still focus primarily on **execution-time capability**: they improve how an agent solves the current task, but only partially address how experience from exploratory execution should be accumulated and reused across future tasks.

This limitation is particularly important for numeric and verifiable time-series tasks. In forecasting, indicator prediction, and related settings, multiple execution paths may all produce task-valid answers, yet differ substantially in quantitative quality under metrics such as MAE or RMSE. In such cases, binary notions of success are insufficient for learning. The question is not merely whether an agent can complete a task, but whether it can identify, preserve, and reuse **better execution paths**. This suggests that agent learning in time-series domains should move beyond completion-only supervision toward **path-quality-aware learning**, a distinction that is only partially captured by existing time-series forecasting and reasoning agents [TimeART; TS-Agent; TimeSeriesScientist; Cast-R1].

A second challenge is that exploratory learning itself is unstable. During training, an agent often develops a strong bias toward a small set of familiar tools that frequently succeed early, then overcommits to them and suppresses further exploration. We refer to this failure mode in benchmark-grounded time-series agent learning as **tool-prior collapse**. Once this happens, candidate diversity shrinks, comparative supervision weakens, and external distillation becomes less informative. For benchmark-grounded scientific tasks, this creates a fundamental bottleneck: without sufficiently diverse exploratory trajectories, the agent cannot reliably learn which behaviors are actually better.

In this paper, we propose **TimeClaw**, a benchmark-grounded external learning framework for time-series agents. Rather than treating each benchmark instance as an isolated execution problem, TimeClaw turns exploration-time execution into a reusable source of scientific experience. Its core design is an external learning loop: the agent explores multiple candidate execution paths, compares them under task-valid supervision, distills the resulting experience into editable external artifacts, and reinjects those artifacts into future problem solving. In this way, benchmark execution becomes not only an evaluation target, but also a metric-supervised learning process. Crucially, TimeClaw separates learning from inference: learning happens during a dedicated exploration phase, while downstream inference reuses distilled artifacts without further online adaptation.

TimeClaw is built around three key ideas. First, we introduce **metric-supervised path learning** for numeric and verifiable time-series tasks, enabling the agent to learn not only whether a task is solvable, but which candidate path achieves better quantitative outcomes. Second, to mitigate tool-prior collapse, we propose a **forced exploration protocol** with **scope-aware tool dropout**, which improves exploratory tool learning by reducing overreliance on dominant tools and increasing candidate-path diversity. Third, we externalize learning through a hierarchical artifact stack that stores reusable experience outside model weights, including path-level lessons, tool guidance, and higher-level reusable skills. In this sense, TimeClaw intersects with tool learning, but goes beyond tool-use correctness by learning from comparative path quality and reusing distilled experience across tasks [Toolformer; Gorilla; ToolLLM; MemSkill; ASDA; Skill-SD]. [Add one final experimental teaser sentence here once results are fixed.]

More broadly, TimeClaw can be viewed as a scientific-domain instantiation of the broader move from stateless tool use toward persistent, learning-enabled agent systems. Unlike open-world agent runtimes that emphasize continuous operation, personalization, or computer-use autonomy, TimeClaw specializes this broader systems philosophy to a benchmark-grounded scientific setting, where experience accumulation is comparable, verifiable, and reusable. This positioning allows us to study not only whether a time-series agent can solve benchmark tasks, but whether it can **learn from exploratory execution in a measurable and interpretable way** [OpenClaw-RL; CUA-Skill; ClawKeeper].

Our contributions are summarized as follows:

1. We formulate benchmark-grounded external learning for time-series agents, and instantiate it in TimeClaw as a learning architecture that separates exploration-time experience acquisition from inference-time reuse.
2. We identify path-quality supervision as a more suitable learning signal than completion-only supervision for numeric and verifiable time-series tasks, and operationalize this insight through metric-supervised path learning.
3. We identify tool-prior collapse in benchmark-grounded time-series agent learning as a major obstacle to exploratory tool learning and propose forced exploration with scope-aware tool dropout to improve candidate-path diversity and supervision quality.
4. We build a dedicated exploratory learning setup for time-series agents and show that distilled external artifacts improve benchmark-grounded reasoning without parameter updates or online adaptation.

### Citation Notes For The English Draft

为避免在 `Introduction` 里堆太多文献，上面英文稿我只保留了最关键的 4 组引用。这个数量对 NeurIPS `Introduction` 比较合适，够支撑主问题，又不会把文献综述提前写成 `Related Work`。

如果你们后面转成 LaTeX，建议把方括号里的标题替换成 bibkey。当前推荐挂的引用如下。

- `MTBench`
  - `MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering`
  - https://arxiv.org/abs/2503.16858
- `TemporalBench`
  - `TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks`
  - https://arxiv.org/abs/2602.13272
- `TimeART`
  - `TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation`
  - https://arxiv.org/abs/2601.13653
- `TS-Agent`
  - `TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering`
  - https://arxiv.org/abs/2510.07432
- `TimeSeriesScientist`
  - `TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis`
  - https://arxiv.org/abs/2510.01538
- `Cast-R1`
  - `Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting`
  - https://arxiv.org/abs/2602.13802
- `Toolformer`
  - `Toolformer: Language Models Can Teach Themselves to Use Tools`
  - https://arxiv.org/abs/2302.04761
- `Gorilla`
  - `Gorilla: Large Language Model Connected with Massive APIs`
  - https://arxiv.org/abs/2305.15334
- `ToolLLM`
  - `ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs`
  - https://arxiv.org/abs/2307.16789
- `MemSkill`
  - `MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents`
  - https://arxiv.org/abs/2602.02474
- `ASDA`
  - `ASDA: Automated Skill Distillation and Adaptation`
  - https://arxiv.org/abs/2603.16112
- `Skill-SD`
  - `Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents`
  - https://arxiv.org/abs/2604.10674
- `OpenClaw-RL`
  - `OpenClaw-RL: Train Any Agent Simply by Talking`
  - https://arxiv.org/abs/2603.10165
- `CUA-Skill`
  - `CUA-Skill: Develop Skills for Computer Using Agent`
  - https://arxiv.org/abs/2601.21123
- `ClawKeeper`
  - `ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers`
  - https://arxiv.org/abs/2603.24414

---

## 9. 这版为什么比上一版更适合冲 Spotlight

### 8.1 主问题更像研究问题，而不是系统描述

它讨论的不再是：

- 有多少工具；
- 有多少 memory 结构；
- 能支持多少模态。

而是：

- agent 如何学习；
- 学什么；
- 为什么现有监督不够；
- 为什么探索会失败；
- 学到的东西如何稳定复用。

### 8.2 贡献层级更清楚

现在的层级是：

- 框架贡献：external learning loop
- 方法贡献：metric-supervised path learning
- 机制贡献：tool-prior collapse + tool dropout
- 资源与证据贡献：exploratory learning setup

这比并列列出很多系统组件更有学术感。

### 8.3 明确修正了最危险的术语风险

你们现在最该避免的是被 reviewer 误读为：

- 测试时在线学习
- runtime self-improvement
- open-world continual adaptation

这版里已经明确写清楚：

- 学习发生在 exploration phase
- inference 只复用 distilled artifacts
- 不进行 online adaptation

这一步对审稿非常重要。

### 8.4 给 `Methodology` 留出了非常自然的接口

后续方法部分几乎可以直接按这个顺序展开：

1. problem formulation
2. dedicated exploration phase
3. candidate-path generation
4. path comparison under task-valid metrics
5. tool-prior collapse and scope-aware tool dropout
6. hierarchical artifact distillation
7. inference-time reinjection

---

## 10. 这一版仍然需要你们后续补齐的两件事

### 9.1 补实验 teaser

如果要更像 Spotlight，`Introduction` 里最好有一句结果级信息。  
现在还只能放占位句。

推荐形式：

- `TimeClaw improves ... by ... on MTBench`
- `The gains are especially pronounced on metric-evaluable forecasting tasks`
- `The improvements come from better path quality rather than only higher completion rate`

### 9.2 补“探索学习数据”的正式命名

如果这个数据集或 protocol 是你们的正式贡献，最好尽早定名。  
因为没有名字，会让 contribution 4 显得偏虚。

例如你们后续可以统一成某个正式名字：

- `TimeClaw-Explore`
- `TimeClaw-EL`
- `TimeClaw Exploration Corpus`

这里只是举例，不是建议你现在定稿。

---

## 11. 一句话总括

如果整篇 `Introduction` 最后只能留下一个中心句，我建议保留：

**TimeClaw is a benchmark-grounded external learning framework for time-series agents that learns from path quality during a dedicated exploration phase and improves downstream inference through reusable distilled artifacts without online adaptation.**

这句话基本可以压住全文的：

- title 气质
- abstract framing
- introduction opening
- related work 收束
- methodology 问题定义
