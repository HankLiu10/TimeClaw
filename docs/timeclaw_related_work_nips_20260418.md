# TimeClaw NeurIPS Related Work Draft

更新时间：2026-04-18

本文档是 Related Work 的更新版，基于 0416 版进行了以下修改：

- 弱化 "benchmark-grounded" 的复合术语用法，改为自然使用
- 修改 TimeART 切分句：从 "teaches a time-series reasoning model to use tools" 改为 "teaches a time-series reasoning model to use tools through expert trajectories; TimeClaw teaches an executing agent to learn from comparative tool-use outcomes and externalize the result into reusable artifacts"
- 全文 "benchmark-grounded" 不再作为复合术语使用，仅在自然需要时出现

---

## 0. 核心判断

如果按现在这版 `Introduction` 的主线来写，TimeClaw 的 `related work` 不应该再以 benchmark 为第一主轴，也不应该写成"我们功能更多"的系统对比。

最合理的组织方式是只保留 **三条主线**：

1. **time-series agent systems**  
   前人如何提升执行能力、协作能力和 forecasting workflow。

2. **tool learning, external memory, and skill distillation**  
   前人如何让 agent 学会用工具、积累外部经验、在 test time 复用 artifact。

3. **persistent open agent runtimes**  
   更大的 systems picture，也就是 OpenClaw 这类工作所在的方向。

如果只保留一句总定位，我建议是：

**TimeClaw should be framed not as a more capable time-series agent, but as an external learning framework that connects tool-augmented execution, comparative path learning, external artifact distillation, and persistent agent runtimes.**

---

## 1. 正文写作策略

如果目标是 NeurIPS Spotlight，那么 related work 的任务不是"把文献讲全"，而是尽快帮助 reviewer 完成三件事：

- 明白 TimeClaw 不是又一个 executor
- 明白 TimeClaw 不是泛泛的 memory/skills system
- 明白 TimeClaw 的 novelty 在于**把探索执行变成外部学习过程**

最重要的 5 句切分句：

- **TimeART teaches a time-series reasoning model to use tools through expert trajectories; TimeClaw teaches an executing agent to learn from comparative tool-use outcomes and externalize the result into reusable artifacts.**
- **TS-Agent and TimeSeriesScientist strengthen how an agent reasons or forecasts within a task, whereas TimeClaw additionally asks how exploratory execution should be converted into reusable cross-task experience.**
- **Cast-R1 brings tool-augmented policy learning closer to time-series forecasting, whereas TimeClaw emphasizes path comparison, external artifact distillation, and inference-time reuse.**
- **MemSkill, ASDA, and Skill-SD show that agents can learn through external memory and skill artifacts; TimeClaw specializes this idea to time-series agents under metric-supervised comparative learning.**
- **OpenClaw-style systems point toward persistent, learning-enabled runtimes in open environments; TimeClaw instantiates a domain-grounded version of this idea for verifiable scientific tasks.**

---

## 2. 最终正文应围绕的三条线

### 2.1 Time-Series Agent Systems

这一组工作是最直接的领域近邻，必须写，但重点不在"逐篇列举"，而在于说明它们主要优化的是**执行阶段**。

核心工作：

- `TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation` <https://arxiv.org/abs/2601.13653>
- `TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering` <https://arxiv.org/abs/2510.07432>
- `TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis` <https://arxiv.org/abs/2510.01538>
- `TimeCopilot` <https://arxiv.org/abs/2509.00616>
- `Empowering Time Series Forecasting with LLM-Agents` <https://arxiv.org/abs/2508.04231>
- `TS-Debate: Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning` <https://arxiv.org/abs/2601.19151>
- `Visual Reasoning over Time Series via Multi-Agent System` <https://arxiv.org/abs/2602.03026>
- `Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting` <https://arxiv.org/abs/2602.13802>

最稳的总结句：

**Prior time-series agent systems primarily improve execution-time reasoning, collaboration, or workflow control, whereas TimeClaw treats exploratory execution itself as a reusable learning signal.**

### 2.2 Tool Learning, External Memory, and Skill Distillation

核心工作：

- `Toolformer: Language Models Can Teach Themselves to Use Tools` <https://arxiv.org/abs/2302.04761>
- `Gorilla: Large Language Model Connected with Massive APIs` <https://arxiv.org/abs/2305.15334>
- `ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs` <https://arxiv.org/abs/2307.16789>
- `MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents` <https://arxiv.org/abs/2602.02474>
- `ASDA: Automated Skill Distillation and Adaptation` <https://arxiv.org/abs/2603.16112>
- `Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents` <https://arxiv.org/abs/2604.10674>
- `Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers` <https://arxiv.org/abs/2603.07670>

最稳的一句：

**TimeClaw is not the first work on tool learning or external artifact learning in general, but it is among the first to combine them into an external learning process for time-series agents.**

### 2.3 Persistent Open Agent Runtimes

核心工作：

- `OpenClaw-RL: Train Any Agent Simply by Talking` <https://arxiv.org/abs/2603.10165>
- `CUA-Skill: Develop Skills for Computer Using Agent` <https://arxiv.org/abs/2601.21123>
- `ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers` <https://arxiv.org/abs/2603.24414>
- `ClawSafety: "Safe" LLMs, Unsafe Agents` <https://arxiv.org/abs/2604.01438>
- `SafeClaw-R: Towards Safe and Secure Multi-Agent Personal Assistants` <https://arxiv.org/abs/2603.28807>

最稳的一句：

**OpenClaw-style systems represent a broader move toward persistent, learning-enabled runtimes; TimeClaw can be viewed as a scientific-domain instantiation of this shift, where persistence serves verifiable experience accumulation rather than open-world automation.**

### 2.4 Benchmark 在正文里怎么处理

benchmark 不需要成为 related work 的独立主线。最合理的处理方式是在 related work 的最后一段用一句到两句点出。

最稳的一句：

**Benchmarks such as MTBench and TemporalBench define what should be evaluated in time-series reasoning, whereas TimeClaw asks how execution on such benchmarks should be turned into reusable learning.**

---

## 3. 建议放进正文的表

intro 放"定位表"，Related Work 不再放表，直接用精炼文字切分。以 Spotlight 的标准，表只放 intro，Related Work 只保留精炼文字。

---

## 4. 最稳的 claim 边界

下面这些句子建议避免：

- `the first time-series agent that ...`
- `no prior work learns from execution`
- `none of existing systems supports closed-loop learning`
- `the first external memory system for agents`

更稳的替代写法：

- `prior work covers parts of the loop`
- `most existing systems optimize execution-time capability`
- `TimeClaw integrates exploration, comparison, distillation, and reuse into an external learning process`
- `among the first to operationalize ... in a time-series setting`

---

## 5. 可直接放进论文的英文 Related Work 草稿

---

## **Related Work**

Recent work related to TimeClaw spans three closely connected directions: time-series agent systems, agent learning through tools and external artifacts, and persistent open agent runtimes.

**Time-series agent systems.**  
Recent work has substantially strengthened time-series agents through tool use, iterative reasoning, forecasting workflow orchestration, and multi-agent collaboration. TimeART teaches a time-series reasoning model to use tools through expert trajectories; TimeClaw teaches an executing agent to learn from comparative tool-use outcomes and externalize the result into reusable artifacts. TS-Agent emphasizes grounded reasoning over raw numeric sequences through iterative evidence gathering, while TimeSeriesScientist and other forecasting-oriented frameworks focus on automating forecasting pipelines and improving end-to-end workflow quality. In parallel, systems such as TS-Debate and MAS4TS improve performance through debate, modality specialization, or multi-agent decomposition within a single task instance. Cast-R1 brings learning-based tool use closer to time-series forecasting by formulating forecasting as sequential decision making with reflective tool use. These approaches primarily optimize execution-time capability. TimeClaw differs in objective: it treats exploratory execution itself as a reusable learning signal, and asks how experience from exploration should be accumulated, distilled, and reused across future tasks.

**Tool learning, external memory, and skill distillation.**  
Beyond the time-series domain, recent work has increasingly treated tool use, external memory, and skill artifacts as explicit learning media for agents. Toolformer, Gorilla, and ToolLLM study how models can invoke external tools more effectively, but mainly focus on tool-use competence and API grounding. A complementary line of work studies persistent learning through external artifacts. MemSkill explores evolving memory skills; ASDA distills structured skill files that can be injected at inference time; Skill-SD shows how multi-turn trajectories can be compressed into reusable guidance. Cast-R1 brings learning-based tool use closer to time-series forecasting by formulating forecasting as sequential decision making with reflective tool use. TimeClaw is complementary to these directions: it is not the first work on tool learning or external artifact learning in general, but it is among the first to combine them into an external learning process for time-series agents, where candidate execution paths are explicitly compared under metric supervision, distilled into structured artifacts, and reused at inference time without online adaptation.

**Persistent open agent runtimes.**  
At the systems level, 2026 has also seen the rapid rise of persistent open agent runtimes, exemplified by OpenClaw-style ecosystems and related work on online training, skill development, runtime governance, and safety. These systems are characterized by long-lived state, reusable skills, plugins, and runtime control layers, but are typically designed for open-world automation rather than verifiable scientific reasoning. TimeClaw shares the broader systems philosophy of persistence and reusable external intelligence, but specializes it to a verifiable scientific setting, where persistence serves experience accumulation grounded in task metrics rather than open-world autonomy.

Finally, TimeClaw is built on benchmark substrates such as MTBench and TemporalBench, but does not aim to contribute another evaluation benchmark. Instead, it asks how execution on such benchmarks should be turned into a reusable learning process. Its novelty lies in combining tool-augmented execution, path-level comparative learning, external artifact distillation, and cross-task reuse into a single external learning framework for time-series agents.

---

## 6. Citation Notes For The English Draft

如果后面转成 LaTeX，建议把上面英文稿里的工作名替换成 bibkey。当前推荐的核心引用如下。

- `MTBench` — `MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering` — https://arxiv.org/abs/2503.16858
- `TemporalBench` — `TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks` — https://arxiv.org/abs/2602.13272
- `TimeART` — `TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation` — https://arxiv.org/abs/2601.13653
- `TS-Agent` — `TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering` — https://arxiv.org/abs/2510.07432
- `TimeSeriesScientist` — `TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis` — https://arxiv.org/abs/2510.01538
- `TS-Debate` — `TS-Debate: Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning` — https://arxiv.org/abs/2601.19151
- `MAS4TS` — `Visual Reasoning over Time Series via Multi-Agent System` — https://arxiv.org/abs/2602.03026
- `TimeCopilot` — https://arxiv.org/abs/2509.00616
- `Empowering TS Forecasting` — `Empowering Time Series Forecasting with LLM-Agents` — https://arxiv.org/abs/2508.04231
- `Cast-R1` — `Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting` — https://arxiv.org/abs/2602.13802
- `Toolformer` — `Toolformer: Language Models Can Teach Themselves to Use Tools` — https://arxiv.org/abs/2302.04761
- `Gorilla` — `Gorilla: Large Language Model Connected with Massive APIs` — https://arxiv.org/abs/2305.15334
- `ToolLLM` — `ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs` — https://arxiv.org/abs/2307.16789
- `MemSkill` — `MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents` — https://arxiv.org/abs/2602.02474
- `ASDA` — `ASDA: Automated Skill Distillation and Adaptation` — https://arxiv.org/abs/2603.16112
- `Skill-SD` — `Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents` — https://arxiv.org/abs/2604.10674
- `Memory Survey` — `Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers` — https://arxiv.org/abs/2603.07670
- `OpenClaw-RL` — `OpenClaw-RL: Train Any Agent Simply by Talking` — https://arxiv.org/abs/2603.10165
- `CUA-Skill` — `CUA-Skill: Develop Skills for Computer Using Agent` — https://arxiv.org/abs/2601.21123
- `ClawKeeper` — `ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers` — https://arxiv.org/abs/2603.24414
- `ClawSafety` — `ClawSafety: "Safe" LLMs, Unsafe Agents` — https://arxiv.org/abs/2604.01438
- `SafeClaw-R` — `SafeClaw-R: Towards Safe and Secure Multi-Agent Personal Assistants` — https://arxiv.org/abs/2603.28807

---

## 7. 一句话总结

**TimeClaw should be framed not as a more capable time-series agent, but as an external learning framework that connects tool-augmented execution, comparative path learning, external artifact distillation, and persistent agent runtimes.**