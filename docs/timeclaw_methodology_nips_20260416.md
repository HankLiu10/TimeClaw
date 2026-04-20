# TimeClaw NeurIPS Methodology Draft

更新时间：2026-04-16

这份文档的目标不是解释仓库怎么跑，而是把 `TimeClaw` 的方法部分组织成一版更接近 **NeurIPS Spotlight** 的 `Methodology`。

它严格遵守下面三条约束：

- 与当前 `Introduction` 和 `Related Work` 的主线完全一致。
- 只基于当前仓库中已经存在、且能被代码与测试支撑的事实来写。
- 主文按你确认的**最终论文协议**展开：
  - 训练：`independent MTBench-compatible corpus`
  - 测试：`TimeART-aligned subset of MTBench`

同时要先说清楚一个关键边界：

- 仓库里仍然保留了基于 `data/splits/` 的官方 benchmark utility，用于历史对照实验和旧跑法；
- 但论文的 methodology 主协议不以这条 legacy split 路径为中心；
- 论文主叙事仍然是：
  - **独立训练语料负责探索学习**
  - **与 TimeART 对齐的 MTBench 子集负责最终评估**

---

## 0. 核心定位

这篇 paper 的方法部分，应该回答的不是“TimeClaw 有哪些模块”，而是：

**How can a time-series agent turn exploratory execution into a benchmark-grounded external learning process, rather than solving each task independently at inference time?**

因此，methodology 的真正主线只有一句：

**TimeClaw is a benchmark-grounded external learning framework for time-series agents, where learning occurs during a dedicated exploration phase through metric-supervised path comparison, and inference reuses distilled external artifacts without online adaptation.**

这句话中的几个术语，必须和 intro 保持一致。

### 0.1 `exploration phase`

新的定义说明：

- `exploration phase` 指专门为了收集候选执行路径、比较路径质量、沉淀外部经验而运行的学习阶段；
- 这一阶段允许真实分支探索、候选比较和 GT/metric 验证；
- 它不是部署时的在线学习。

### 0.2 `inference phase`

新的定义说明：

- `inference phase` 指部署或测试时的求解阶段；
- 这一阶段只复用探索阶段蒸馏出的 artifact；
- 不再继续在线学习，也不再暴露训练专用的探索工具。

### 0.3 `task-valid candidate`

新的定义说明：

- `task-valid candidate` 不是“看起来像答案”的中间推理，而是满足当前任务输出契约的合法候选输出；
- 例如：
  - forecasting 任务中的数值数组；
  - indicator 任务中的结构化 JSON；
  - trend / correlation 任务中的合法标签；
  - QA 任务中的最终文本答案或选项标签。

### 0.4 `path-quality supervision`

新的定义说明：

- `path-quality supervision` 指对不同候选执行路径的监督，不是基于“是否完成任务”的二值信号，而是基于任务认可的质量指标；
- 对 numeric / verifiable tasks，这类监督通常来自 `MAE`、`RMSE`、accuracy 或任务特定误差；
- 它回答的是“哪条路径更优”，而不仅是“哪条路径可行”。

### 0.5 `external learning`

新的定义说明：

- `external learning` 指 TimeClaw 的学习结果主要沉淀在模型参数之外；
- 学习产物被组织为外部 artifact，例如 structured memory、tool notes、skills 和历史快照；
- 推理阶段通过 prompt reinjection 复用这些 artifact，而不是靠在线参数更新。

---

## 1. Methodology 应该怎样写

按 NeurIPS Spotlight 的标准，这一节不应该写成：

- 一个多模态时序系统的模块说明书；
- 一个“我们也有 memory / tools / skills”的工程清单；
- 一个 ReAct agent 的流水账。

更合适的组织方式不是拆成很多零碎小节，而是压缩成少数几个高密度、强收束的核心块。更推荐的主文结构是下面这 `6` 节：

1. **Problem Setup**
2. **Data Protocol**
3. **TimeClaw Learning Framework**
4. **Exploration-Time Agent Learning**
5. **Tool-Prior Collapse and Scope-Aware Tool Dropout**
6. **Hierarchical Distillation and Inference-Time Reuse**

这样组织有两个优点：

- reviewer 更容易抓住唯一的方法对象：`Explore -> Compare -> Distill -> Reinject`
- 细节仍然完整，但不会被写成层层展开的系统说明书

如果这 6 节都写得足够紧，那么 methodology 就更像 Spotlight 论文，而不是长篇技术报告。

---

## 2. 可直接放进论文的英文 Methodology 草稿

下面这版以“几乎可直接贴进论文”为目标。它尽量避免仓库说明语气，保留真正必要的形式化对象、算法框和实现边界。

---

## **3. Methodology**

### **3.1 Problem Setup**

We study **benchmark-grounded external learning** for time-series agents. Each task instance is written as

\[
x = (z, c, \tau, s),
\]

where \(z\) denotes the time-series input, \(c\) denotes optional multimodal context such as text, \(\tau\) is the task type, and \(s\) is the scope under which reusable experience is stored. Rather than committing to a single execution chain, the agent may explore a set of candidate paths

\[
\Pi(x) = \{\pi_1,\ldots,\pi_K\},
\]

where each path \(\pi_k\) is a structured trajectory of tool selections, intermediate observations, and a final task output. Let

\[
\hat y_k = \mathrm{Exec}(x,\pi_k)
\]

denote the output produced by path \(\pi_k\). A path is **task-valid** if \(\hat y_k\) satisfies the output contract of task \(\tau\), e.g. a forecast array for forecasting, a structured JSON object for indicator prediction, or a legal label for trend and correlation tasks.

The key methodological issue is that, for numeric and verifiable time-series tasks, multiple paths may all be task-valid while still differing substantially in quality. We therefore distinguish **completion** from **path quality**. Given task-valid supervision, the quality of a candidate path is written as

\[
q(\pi_k; x) = -\mathcal{L}_{\tau}(\hat y_k, y^\star),
\]

where \(y^\star\) is the task target and \(\mathcal{L}_{\tau}\) is task-dependent, e.g. MAE or RMSE for continuous outputs and a correctness-based loss for discrete outputs. The preferred path is then

\[
\pi^\star(x) = \arg\max_{\pi_k \in \Pi_{\mathrm{valid}}(x)} q(\pi_k; x),
\]

where \(\Pi_{\mathrm{valid}}(x)\subseteq \Pi(x)\) is the set of task-valid candidates. If no candidate is validated, the learning episode records failure evidence rather than forcing an artificial winner.

This formalization clarifies why completion-only supervision is insufficient. If two paths \(\pi_a\) and \(\pi_b\) both yield task-valid outputs, then a binary success signal maps both to the same supervision value even when

\[
\mathcal{L}_{\tau}(\hat y_a, y^\star) \neq \mathcal{L}_{\tau}(\hat y_b, y^\star).
\]

Thus, completion-only supervision cannot distinguish paths that are all executable but quantitatively unequal. TimeClaw therefore learns from **path-quality supervision**, not only from completion.

This is not a standalone theory claim; it is a methodological justification for the learning signal. For metric-evaluable tasks, any learning rule that aims to preserve stronger future execution must distinguish between valid-but-unequal paths rather than collapsing them into the same binary success label.

TimeClaw separates the agent lifecycle into two phases. In the **exploration phase**, the agent generates and compares candidate execution paths under task-valid supervision. In the **inference phase**, the agent does not learn online; instead, it reuses distilled external artifacts accumulated during exploration. This separation keeps the learning process auditable, reusable, and benchmark-grounded.

### **3.2 Data Protocol**

TimeClaw adopts a two-corpus protocol. For learning, we construct an **independent MTBench-compatible corpus** whose task contracts, schemas, and scope structure are aligned with MTBench, but whose samples are disjoint from the official benchmark. This corpus is used only for exploration-time experience acquisition. For evaluation, all reported experiments use a **TimeART-aligned subset of MTBench**. The role of the independent corpus is to expose whether the agent can acquire reusable execution knowledge; the role of the MTBench subset is to provide a stable and directly comparable evaluation target.

This protocol is preferable to simply splitting MTBench into train and test subsets. A benchmark split is sufficient for measuring temporal generalization, but it is less well suited for isolating whether exploratory agent learning transfers beyond the benchmark samples themselves. By learning on an aligned but disjoint corpus and evaluating on the same MTBench subset used by TimeART, TimeClaw centers the methodology on reusable learning while preserving direct head-to-head comparability. Legacy benchmark-split utilities may still be used for ablations, but they are not the primary protocol of this paper.

### **3.3 TimeClaw Learning Framework**

TimeClaw instantiates external learning as a four-stage loop:

1. **Explore**: generate multiple candidate execution paths for the same task.
2. **Compare**: evaluate task-valid candidates under benchmark-grounded supervision.
3. **Distill**: compress path-level evidence into reusable external artifacts.
4. **Reinject**: reuse the distilled artifacts during later inference without online adaptation.

This `Explore -> Compare -> Distill -> Reinject` loop is the main methodological object of the paper. Prior time-series agent systems mainly strengthen execution-time capability within a task instance. TimeClaw instead treats execution itself as a reusable learning signal. The key transition is from candidate-path exploration to artifact update. For a single training instance \(x\), TimeClaw first constructs a sample-level learning record

\[
n_x = \mathrm{Summarize}\!\left(x,\Pi(x), \{q(\pi_k;x)\}_{k=1}^{K}, \pi^\star(x)\right),
\]

which captures the strongest validated path, decisive evidence, and failure signals from weaker paths. However, the stored learning signal is not the raw reflective text emitted by the agent. TimeClaw further rewrites the candidate-level comparison into reusable execution evidence:

\[
e_x = \mathrm{Clean}\!\left(n_x\right),
\]

where \(\mathrm{Clean}(\cdot)\) removes task-answer leakage, framework-control language, and orchestration-specific wording that would not transfer to future tasks. The purpose is to preserve reusable operational evidence rather than surface-form self-commentary. The cleaned evidence is then used to update the external artifact state associated with scope \(s\):

\[
\mathcal{A}_{s}^{(m+1)} = \mathrm{Distill}\!\left(\mathcal{A}_{s}^{(m)}, e_x\right),
\]

where \(\mathcal{A}_{s}\) denotes the current external state for that scope. The update target is not a model parameter vector but a structured artifact state containing notes, reusable rules, tool guidance, and operational procedures. This is precisely why TimeClaw is an **external learning** framework rather than an online parameter-learning runtime.

We summarize the learning protocol below.

```text
Algorithm 1: TimeClaw External Learning Loop
Input: task x, task type t, scope s

Exploration phase:
  1. Run the executor in training mode on x.
  2. Generate multiple task-valid candidate paths.
  3. Compare candidate outputs under task-valid supervision.
  4. Select the strongest validated path, or record that no path is validated.
  5. Distill sample-level evidence into external artifacts under scope s.

Inference phase:
  6. Retrieve artifacts relevant to scope s.
  7. Reinject them into the prompt and solve new tasks without online learning.
```

### **3.4 Exploration-Time Agent Learning**

The exploration phase is implemented through a constrained executor that converts autonomous execution into explicit candidate-path learning. In training mode, the main agent is allowed to access exploration-only orchestration tools, including `spawn_subagent` for branch generation and `evaluate_*` tools for post-hoc comparison against ground truth. However, this is not unconstrained free-form exploration. TimeClaw enforces a structured training contract:

- the main agent must produce **at least two task-valid candidate paths**;
- these candidates must arise from genuine branch exploration rather than stylistic paraphrases of the same path;
- when a prior successful path family already exists for the scope, the first exploration round must include both a prior-guided candidate and an alternative candidate;
- candidate comparison must occur before training completion;
- `evaluate_*` tools are only meaningful after candidate generation, and `learning_summary` is gated on completed comparison;
- the agent may finish only by returning a `learning_summary`, not by treating the training run as ordinary inference.

This contract matters because it turns what would otherwise be a loose exploratory run into a controlled learning episode. A trajectory is considered complete only after candidate generation and task-valid comparison are already satisfied. The resulting `learning_summary` is not the final task answer. Instead, it records the selected path, the decisive evidence behind that selection, and a concise reusable lesson that can later be distilled into external artifacts.

To keep candidate paths semantically comparable, intermediate tool outputs are also tracked as typed artifacts with coordinate provenance. In practice, this means downstream operations are conditioned on the current artifact space rather than blindly reusing indices from the original series. This constraint is especially important for sliced sequences, derived windows, and forecast artifacts, where the meaning of an index depends on the transformation history.

The executor therefore implements a constrained search-and-selection protocol rather than unconstrained deliberation:

```text
Algorithm 2: Exploration-Time Candidate Learning
Input: task instance x

1. Generate branch candidates with exploration-enabled tools.
2. Keep only task-valid outputs under the contract of task τ.
3. Evaluate validated candidates against task-valid supervision.
4. Select the strongest validated path π* if one exists.
5. Emit a learning_summary aligned to π* and its decisive evidence.
```

The importance of this protocol is methodological rather than procedural: it ensures that the learned signal is grounded in explicit path competition rather than in post-hoc rationalization of a single execution.

### **3.5 Tool-Prior Collapse and Scope-Aware Tool Dropout**

A major obstacle to exploratory tool learning is that the agent quickly develops strong preferences for a small subset of familiar tools. We refer to this phenomenon as **tool-prior collapse**: during benchmark-grounded exploration, historically frequent or often-successful tools can dominate future branch generation, reducing path diversity and weakening subsequent comparison and distillation.

Let \(n_i(s)\) denote the historical usage count of tool \(i\) under scope \(s\), and let \(p_i(s)\) be the normalized usage distribution over competing tools in that scope:

\[
p_i(s)=\frac{n_i(s)}{\sum_j n_j(s)}.
\]

Then one operational signature of collapse is decreasing tool-support diversity, which can be tracked by the scope-level entropy

\[
\mathcal{H}(s) = - \sum_i p_i(s)\log p_i(s).
\]

When a small number of tools dominate, \(\mathcal{H}(s)\) decreases and the explored candidate set becomes structurally narrower. In that regime, path comparison becomes less informative because multiple branches differ only superficially while relying on the same dominant tool family.

TimeClaw mitigates this problem with **scope-aware tool dropout**. The mechanism operates during **training-time sub-agent tool exposure**. For each scope, the system tracks historical tool usage and computes a keep probability for each candidate tool relative to the competing tools in that scope. Frequently used tools receive a lower keep probability, while underused tools remain more visible. Task-critical tools and explicitly hinted tools are protected through a `never_drop` rule. The purpose is not regularization for its own sake, but to preserve branch diversity so that the agent can continue to generate meaningfully different candidate paths rather than repeatedly collapsing onto the same familiar execution pattern.

Concretely, for tool \(i\) in a candidate pool under scope \(s\), the keep probability is

\[
p_{\mathrm{keep}}(i \mid s)=
\left(\frac{1+n_{\min}(s)}{1+n_i(s)}\right)^{\alpha},
\]

where \(n_{\min}(s)\) is the minimum usage count among the competing non-protected tools in that scope, and \(\alpha > 0\) controls the strength of reweighting. This form has two important properties. First, it is monotone decreasing in \(n_i(s)\), so historically dominant tools are more likely to be suppressed. Second, tools at the cold edge of the scope, i.e. \(n_i(s)=n_{\min}(s)\), receive keep probability \(1\), so the mechanism explicitly protects under-explored tool families rather than merely injecting uniform randomness. Protected tools bypass dropout altogether:

\[
p_{\mathrm{keep}}(i \mid s)=1 \quad \text{for } i \in \mathcal{T}_{\mathrm{protected}}(s).
\]

Moreover, for non-protected tools \(i\) and \(j\) in the same scope \(s\), if \(n_i(s) > n_j(s)\), then the keep rule implies

\[
p_{\mathrm{keep}}(i \mid s) < p_{\mathrm{keep}}(j \mid s)
\quad \text{for } \alpha > 0.
\]

The least-used non-protected tools in the scope satisfy \(p_{\mathrm{keep}}=1\). This makes scope-aware dropout a targeted anti-collapse mechanism rather than a generic regularizer: it suppresses dominant tools while guaranteeing visibility of under-explored ones.

### **3.6 Hierarchical Distillation, Conflict-Aware Memory, and Inference-Time Reuse**

TimeClaw externalizes learning into a hierarchical artifact stack rather than updating model parameters. The stack contains several layers with different semantic roles:

- **SOUL**: a fixed human-authored identity and behavior layer;
- **Notes**: sample-level experience records produced from individual exploration runs;
- **MEMORY**: scope-local structured reusable rules distilled from repeated notes;
- **TOOLS**: tool-specific usage guidance summarized from scope-local learning;
- **SKILLS**: scope-local operational procedures or task-solving SOPs;
- **cross-domain**: task-bucket abstractions synthesized across multiple scopes.

This hierarchy gives TimeClaw a concrete medium for persistent learning. Notes capture raw experience at sample granularity, but they are not the main runtime memory surface. Instead, they serve as distillation sources from which more stable scope-level artifacts are derived. The key runtime layer is `MEMORY`, whose entries are structured reusable rules rather than free-form reflections. A memory rule can be written abstractly as

\[
r = (\kappa,\sigma,\chi,\mathcal{T}^{+},\mathcal{T}^{-},\rho,\mathcal{E},c,\iota),
\]

where \(\kappa\) is the rule kind, \(\sigma\) is a concise summary, \(\chi\) denotes applicability conditions, \(\mathcal{T}^{+}\) and \(\mathcal{T}^{-}\) are preferred and avoided tool sets, \(\rho\) is the rationale, \(\mathcal{E}\) is supporting evidence, \(c\) is confidence, and \(\iota \in \{0,1\}\) indicates whether the rule is currently safe to inject into the prompt.

This representation is important because TimeClaw does not update memory by naive append. Given cleaned evidence \(e_x\), the scope-local memory state is updated through a conflict-aware operator

\[
\mathcal{M}_{s}^{(m+1)} = \mathrm{Update}\!\left(\mathcal{M}_{s}^{(m)}, e_x\right),
\]

where `Update` may append a new rule, merge near-duplicate evidence into an existing rule, strengthen a supported rule, or register a conflict relation between competing rules. When an item is not yet stable enough for prompt-time reuse, it can remain stored but be marked non-injectable through \(\iota=0\). This separation between storage and injection is the core mechanism by which TimeClaw preserves long-horizon experience without forcing every intermediate belief into the runtime prompt.

The same logic extends beyond `MEMORY`. `TOOLS` distills tool-specific usage advice from repeated execution evidence, while `SKILLS` compresses stronger scope-local procedures into reusable SOP-like guidance. To preserve learning over time, the system also records historical snapshots of notes, current memory state, tool notes, skill text, and cross-domain abstractions. These snapshots provide a long-term audit trail of how reusable experience evolves rather than reducing learning to the latest prompt state.

During inference, the system reuses these artifacts without online learning. Formally, inference on a new instance \(x'\) under scope \(s\) is written as

\[
\hat y = f_{\mathrm{agent}}\!\left(x';\, \mathrm{Retrieve}(\mathcal{A}_{s}), \mathcal{T}(x')\right),
\]

where \(\mathrm{Retrieve}(\mathcal{A}_{s})\) returns the relevant learned artifacts for the scope and \(\mathcal{T}(x')\) is the runtime-compatible visible tool set. In the present implementation, retrieval is filtered by scope, applicability, and injection status. Abstractly, the prompt-eligible memory subset is

\[
\mathcal{R}_{s}(x')=
\{r \in \mathcal{M}_{s} : \iota(r)=1,\; \mathrm{Match}(\chi(r), \phi(x'))=1\},
\]

where \(\phi(x')\) denotes the sample fingerprint and profiling view of the current instance. Prompt construction first assembles an official task description, then provides the task boundary and output contract, then injects the sample fingerprint and profiling signals, and finally reinjects the learned artifact layers. In the present implementation, `SOUL` and scope-local `MEMORY` are injected at the system level, while `tool notes` and `skills` are injected as reusable support guidance; the visible tool catalog is further filtered by modality and runtime compatibility.

The train-test difference is crucial. Training exposes exploration-only tools such as `spawn_subagent` and `evaluate_*` because the goal is to compare candidate paths. Inference removes these training-only tools and retains only the tools needed to solve the task itself. Thus, the runtime agent is not continuing to learn; it is solving the task under a prompt enriched by previously distilled experience. The `cross-domain` layer exists in storage and distillation, but it is not the main runtime injection path in the present implementation. Accordingly, the primary reuse mechanism is best understood as **scope-local artifact reinjection**, not as a fully cross-domain reasoning controller.

---

## 4. 写作解释与 claim 边界

上面这版英文正文，已经按论文标准压缩，但为了后续继续迭代，这里把几处最容易写歪的地方单独说清。

### 3.1 为什么不用 `self-improving`

因为当前 TimeClaw 的学习并不是：

- test-time online learning；
- 边推理边写回策略；
- 一边部署一边参数更新。

更准确的说法是：

- `exploration-trained`
- `externally learned`
- `artifact-reuse at inference time`

这和 intro 里的 `without online adaptation` 必须保持严格一致。

### 3.2 为什么 `path-quality supervision` 是主方法点

这不是一个“实验观察”，而是一个方法学选择。

TimeClaw 的关键判断是：

- 对 metric-evaluable tasks，是否完成任务只是最低要求；
- 更重要的是哪条路径在任务指标上更优；
- 因而训练应围绕路径质量展开。

所以 methodology 里必须把这一点写成：

**a more suitable learning signal for metric-evaluable time-series tasks**

而不是写成：

- “我们实验发现 MAE 很重要”
- “我们顺手比较了不同路径”

### 3.3 为什么 `tool_dropout` 可以写进 methodology 主体

因为在当前实现里，它不是一个实验 trick，而是训练阶段真实使用的探索控制机制：

- 只作用于 training-time sub-agent tool exposure；
- 基于 scope-level usage 统计；
- 会影响可见工具子集；
- 直接影响候选路径多样性。

因此它属于方法，而不是实验细节。

### 3.4 为什么 `cross-domain` 不能写得过重

当前仓库里 `cross-domain` 的事实是：

- 它在存储和蒸馏层真实存在；
- 会出现在 scope history snapshot 里；
- 会从 per-scope memory 聚合生成；
- 但当前 prompt 注入主路径并不以它为中心。

所以最安全、也最准确的写法是：

- 把它写成 distillation hierarchy 的一层；
- 不把它写成当前推理的核心控制器。

这不是弱化，而是避免 overstating。

---

## 5. Appendix-Ready 补充：Prompt 组装流程与信息来源

这部分不一定全部放进主文，但非常适合后续作为 appendix 或 supplementary。

---

### 4.1 Prompt 组装总览

当前实现中，executor 的 prompt 可以概括为 6 个来源的组合：

1. **Official task description**
2. **Task boundary and output contract**
3. **Fingerprint and profiling signals**
4. **System-level learned artifacts**
5. **User-side operational support**
6. **Visible tool catalog**

---

### 4.2 信息来源 1：Official task description

来源：

- `timeclaw/benchmark/mtbench_support/task_descriptions.py`
- `timeclaw/benchmark/mtbench_prompts.py`

作用：

- 将 benchmark task 转成规范化的 system/user task prompt；
- 保留官方任务定义；
- 让训练和测试在任务语义层面对齐。

渲染结果示意：

```text
System prompt:
<official task instruction>

User prompt:
<sample-specific question / forecast instruction / QA prompt>
```

---

### 4.3 信息来源 2：Task boundary and output contract

来源：

- `timeclaw/agents/executor.py`
  - `_build_task_boundary_section`
  - `_build_main_output_contract_section`

作用：

- 明确什么是 task-valid output；
- 防止 agent 输出错误类型答案；
- 在 training / inference 两种模式下给出不同的 completion contract。

例如：

- forecasting 任务必须输出数值数组；
- indicator 任务必须输出指定 key 的 JSON；
- training mode 不能直接用 task answer 收尾，而要输出 `learning_summary`。

---

### 4.4 信息来源 3：Fingerprint and profiling signals

来源：

- `timeclaw/fingerprint/extractor.py`
- `timeclaw/pipeline/runner.py`
- `timeclaw/agents/executor.py`

内容：

- data shape
- length
- missing ratio
- basic stats
- stationarity
- trend-related profiling
- anomaly count 或其他 profiling 结果

作用：

- 给 agent 一个压缩后的样本结构视图；
- 让 tool choice 更 grounded；
- 让后续 `MEMORY` / `SKILLS` 的适用条件更容易被调用。

渲染结果示意：

```text
Sample Fingerprint
{"length":168,"num_channels":1,"missing_ratio":0.0,"trend":"upward",...}

Profiling
- stationarity_test: ...
- detect_trend: ...
```

---

### 4.5 信息来源 4：System-level learned artifacts

来源：

- `timeclaw/pipeline/runner.py`
- `timeclaw/memory/hierarchical.py`

当前主注入层包括：

- `SOUL`
- scope-local `MEMORY`

作用：

- 提供固定身份与高层行为边界；
- 注入 scope-aware learned rules；
- 在 inference 阶段实现“经验已学会，但不在线更新”。

渲染结果示意：

```text
## MEMORY — Learned Insights

### finance_aligned_short_forecast Specific
- Rule: when short-horizon movement is smooth and recent local trend is stable, prefer ...
- Rule: when text cue is weak but series continuity is strong, avoid ...
```

说明：

- 这里展示的是渲染风格示意，不是要在本文档中完整展开 memory 文件内容。

---

### 4.6 信息来源 5：User-side operational support

来源：

- `timeclaw/memory/hierarchical.py`
- `timeclaw/agents/executor.py`

当前主注入层包括：

- `tool notes`
- `skills`

作用：

- 给 agent 提供更操作性的局部经验；
- 让 learned guidance 不只停留在抽象规则层。

渲染结果示意：

```text
Tool Notes:
- chronos2_forecast: When it helps: stable short-horizon continuation ...
- llm_reasoning_indicator: When to avoid: weak numeric evidence ...

## SKILL — Operational Guide for weather_aligned_short_forecast
1. Check horizon and continuity first.
2. Use ...
3. Avoid ...
```

---

### 4.7 信息来源 6：Visible tool catalog

来源：

- `timeclaw/agents/prompts/executor_prompt.py`
- `timeclaw/clawhub/tool_registry.py`

作用：

- 仅暴露当前模态兼容的工具；
- training mode 和 inference mode 暴露不同工具集合；
- tool catalog 还附带参数契约和用途说明。

其中最重要的训练 / 测试差异是：

- training:
  - 暴露 `spawn_subagent`
  - 暴露 `evaluate_against_gt`
  - 暴露 `evaluate_batch_against_gt`
- inference:
  - 移除这些 training-only tools
  - 只保留任务求解所需工具

---

## 6. Appendix-Ready 补充：流程与训练产物结构

这里继续按你之前要求，给出“简洁但完整”的流程和产物梳理。

---

### 5.1 总流程

最终论文协议下，TimeClaw 的高层流程是：

```text
Independent training corpus
    -> exploration-time executor runs
    -> candidate-path comparison
    -> sample-level notes
    -> hierarchical distillation
    -> reusable external artifacts
    -> inference-time reinjection
    -> evaluation on the TimeART-aligned MTBench subset
```

如果写成一句更论文化的话：

**TimeClaw learns externally from exploration on a disjoint MTBench-compatible corpus, distills the resulting path-level experience into structured artifacts, and reuses those artifacts when evaluated on the TimeART-aligned MTBench subset.**

---

### 5.2 训练阶段的主要产物

按当前实现，训练后会形成几类关键产物：

1. `train_results`
2. `trace`
3. `candidate traces`
4. `structured learning analysis`
5. `scope history snapshots`

按照你的要求，本文档不展开：

- `notes`
- `memory`
- `skills`
- `tools.md`

但会说明这些未展开产物在整体流程里的角色。

---

### 5.3 `train_results`

来源：

- `.timeclaw/train_results/`

作用：

- 按 `(dataset, task, sample)` 记录训练 episode 的可恢复结构；
- 保存 `learning_summary`、winner metadata、metrics、tool chain 等；
- 支撑后续 resume、分析和附录可视化。

渲染结果示意：

```json
{
  "dataset_name": "weather_aligned_short",
  "task": "forecast",
  "idx": 17,
  "answer_type": "learning_summary",
  "winner_candidate": "chronos2_forecast",
  "winner_tool_chain": ["chronos2_forecast"],
  "winner_metrics": {"mae": 1.2},
  "learning_insight": "The chronos-led path matched the local continuation pattern better.",
  "learning_recommendation": "Prefer the chronos-led path when short-horizon continuity is strong."
}
```

---

### 5.4 `trace`

来源：

- `.timeclaw/traces/*.jsonl`

作用：

- 记录一次 executor run 的逐步执行轨迹；
- 保留 tool calls、观测、sub-agent 行为和 dropout 信息；
- 方便回溯某个 learning episode 是如何形成的。

渲染结果示意：

```json
{"agent":"executor","step":"spawn_subagents","n_tasks":2,...}
{"tool":"chronos2_forecast","success":true,"observation":{...}}
{"tool":"evaluate_batch_against_gt","success":true,"observation":{...}}
```

---

### 5.5 `candidate traces`

来源：

- 从 `execution_log` 中提取，不一定单独落成独立主文件；
- 在 learning analysis、winner selection 和 memory distillation 中被使用。

作用：

- 将候选路径显式化；
- 保留每条路径的 tool chain、prediction、metrics、public reasoning；
- 支撑 path-level comparison，而不是只保留最终 winner。

渲染结果示意：

```json
[
  {
    "name": "chronos2_forecast",
    "tool_chain": ["chronos2_forecast"],
    "prediction": [12.1, 12.4, 12.8],
    "metrics": {"mae": 1.2}
  },
  {
    "name": "ets_forecast",
    "tool_chain": ["ets_forecast"],
    "prediction": [11.6, 11.9, 12.0],
    "metrics": {"mae": 2.5}
  }
]
```

---

### 5.6 `structured learning analysis`

来源：

- `timeclaw/benchmark/mtbench_support/learning_analysis.py`

作用：

- 把单样本探索结果转成紧凑、可蒸馏的学习材料；
- 为 `Notes -> MEMORY` 的蒸馏提供更干净的中间表示；
- 强制区分：
  - sample conditions
  - winner evidence
  - failure signals
  - reusable rule

渲染结果示意：

```json
{
  "text_signals": ["..."],
  "sample_conditions": ["short-horizon continuation is smooth", "recent trend is stable"],
  "winner_evidence": ["chronos2_forecast matched the local continuation pattern and achieved lower MAE"],
  "failure_signals": ["ets_forecast lagged the turning point and underfit the early rise"],
  "reusable_rule": ["When short-horizon continuity is strong and structural breaks are weak, prefer a continuation-oriented forecasting path before heavier decomposition."]
}
```

---

### 5.7 `scope history snapshots`

来源：

- `.timeclaw/hierarchical_memory/history/...`

作用：

- 记录某个 scope 在某个蒸馏时刻的可复用学习状态；
- 保留该 scope 的 memory entries、tool notes、skill content，以及同 bucket 的 cross-domain summary；
- 让学习状态具备时间切片和可追溯性。

渲染结果示意：

```json
{
  "scope": "finance_aligned_short_forecast",
  "snapshot_seq": 12,
  "rule_count": 5,
  "tool_note_count": 3,
  "has_skill": true,
  "cross_domain_rule_count": 2
}
```

---

## 7. 最终自检清单

这份 methodology 在写法上，应该满足下面这些检查点。

### 6.1 与 intro 保持一致

- 使用 `benchmark-grounded external learning`
- 使用 `exploration phase / inference phase`
- 使用 `metric-supervised path learning`
- 使用 `without online adaptation`
- 使用 `tool-prior collapse`

### 6.2 与 related work 保持一致

- 不把 TimeClaw 写成“又一个 time-series agent”
- 不把它缩成纯 `tool learning paper`
- 不把 benchmark 写成主要 novelty
- 继续强调：
  - path-level comparative learning
  - external artifact distillation
  - inference-time reuse

### 6.3 不过度声称

- 不说 `the first paradigm`
- 不说 `fully self-improving`
- 不说 `cross-domain` 已经成为当前推理主控层
- 不把 legacy split 说成最终主协议

---

## 8. 一句话总结

如果只允许用一句话概括这部分 methodology，我建议保留：

**TimeClaw turns exploratory execution on time-series tasks into a benchmark-grounded external learning process, where candidate paths are compared under task-valid supervision, distilled into reusable artifacts, and later reinjected at inference time without online adaptation.**
