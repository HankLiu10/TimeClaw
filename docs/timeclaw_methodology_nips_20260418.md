# TimeClaw NeurIPS Methodology Draft

更新时间：2026-04-18

本文档是 Methodology 的更新版，基于 0416 版进行了以下修改：

- 弱化 "benchmark-grounded" 的复合术语用法，改为自然使用
- 将 path-quality supervision 的对比对象修正为 agentic binary supervision，增加 agent learning scope 说明
- tool-prior collapse 定义中增加 "prematurely" 限定，区分 exploration/inference 阶段
- Algorithm 1 重写为正式伪代码格式（External Learning Loop）
- Algorithm 2 重写为 Inference-Time Reuse（而非 Exploration-Time Candidate Learning）
- 新增 Remark：与 ICL、meta-learning、continual learning、RAG 的关系
- keep probability 公式前新增理论动机段落
- 保留 appendix-ready 材料（用户后续自行挪到附录）

---

## 0. 核心定位

这篇 paper 的方法部分，应该回答的不是"TimeClaw 有哪些模块"，而是：

**How can a time-series agent turn exploratory execution into an external learning process, rather than solving each task independently at inference time?**

methodology 的真正主线只有一句：

**TimeClaw is an external learning framework for time-series agents, where learning occurs during a dedicated exploration phase through metric-supervised path comparison, and inference reuses distilled external artifacts without online adaptation.**

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
- 对比对象：agentic 场景中的二值成功/失败信号（如 OpenClaw 风格的 agent），而非 RL reward shaping；
- 深度学习模型可以通过梯度直接优化损失函数，不需要"路径比较"；但 agent 无法对执行路径做反向传播，因此需要通过显式的路径质量比较来学习。

### 0.5 `external learning`

- `external learning` 指 TimeClaw 的学习结果主要沉淀在模型参数之外；
- 学习产物被组织为外部 artifact，例如 structured memory、tool notes、skills 和历史快照；
- 推理阶段通过 prompt reinjection 复用这些 artifact，而不是靠在线参数更新。

---

## 1. Methodology 应该怎样写

按 NeurIPS Spotlight 的标准，推荐的主文结构是下面 6 节：

1. **Problem Setup**
2. **Data Protocol**
3. **TimeClaw Learning Framework**
4. **Exploration-Time Agent Learning**
5. **Tool-Prior Collapse and Scope-Aware Tool Dropout**
6. **Hierarchical Distillation and Inference-Time Reuse**

---

## 2. 可直接放进论文的英文 Methodology 草稿

---

## **3. Methodology**

### **3.1 Problem Setup**

We study **external learning** for time-series agents. Each task instance is written as

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

Thus, completion-only supervision cannot distinguish paths that are all executable but quantitatively unequal. This limitation is specific to the agentic setting. Deep learning models can directly optimize a differentiable loss function via gradient descent, so the distinction between "task completed" and "task completed well" is naturally captured by the loss landscape. Agents, however, cannot backpropagate through their execution paths — tool calls, branching decisions, and intermediate observations form a non-differentiable trajectory. The only way for an agent to learn which path is better is through explicit comparison of execution outcomes. TimeClaw therefore learns from **path-quality supervision**, not only from completion.

This is not a standalone theory claim; it is a methodological justification for the learning signal. For metric-evaluable tasks, any learning rule that aims to preserve stronger future execution must distinguish between valid-but-unequal paths rather than collapsing them into the same binary success label.

TimeClaw separates the agent lifecycle into two phases. In the **exploration phase**, the agent generates and compares candidate execution paths under task-valid supervision. In the **inference phase**, the agent does not learn online; instead, it reuses distilled external artifacts accumulated during exploration. This separation keeps the learning process auditable, reusable, and grounded in verifiable task metrics.

**Remark (Relation to existing learning paradigms).** TimeClaw's external learning shares the context-injection mechanism with in-context learning (ICL), but differs in that ICL directly uses whatever examples are provided at inference time without an active learning process, whereas TimeClaw generates its injected artifacts through a dedicated explore-compare-distill loop. External learning also differs from meta-learning, which typically learns parameter initializations or learning rules for fast adaptation; from continual learning, which updates model parameters on streaming data while mitigating forgetting; and from retrieval-augmented generation (RAG), which retrieves from a pre-existing knowledge base. In TimeClaw, the external artifacts are not pre-existing but are actively constructed through exploratory execution and comparative learning, and no model parameters are modified.

### **3.2 Data Protocol**

TimeClaw adopts a two-corpus protocol. For learning, we construct an **independent MTBench-compatible corpus** whose task contracts, schemas, and scope structure are aligned with MTBench, but whose samples are disjoint from the official benchmark. This corpus is used only for exploration-time experience acquisition. For evaluation, all reported experiments use a **TimeART-aligned subset of MTBench**. The role of the independent corpus is to expose whether the agent can acquire reusable execution knowledge; the role of the MTBench subset is to provide a stable and directly comparable evaluation target.

This protocol is preferable to simply splitting MTBench into train and test subsets. A benchmark split is sufficient for measuring temporal generalization, but it is less well suited for isolating whether exploratory agent learning transfers beyond the benchmark samples themselves. By learning on an aligned but disjoint corpus and evaluating on the same MTBench subset used by TimeART, TimeClaw centers the methodology on reusable learning while preserving direct head-to-head comparability. Legacy benchmark-split utilities may still be used for ablations, but they are not the primary protocol of this paper.

### **3.3 TimeClaw Learning Framework**

TimeClaw instantiates external learning as a four-stage loop:

1. **Explore**: generate multiple candidate execution paths for the same task.
2. **Compare**: evaluate task-valid candidates under metric-supervised comparison.
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

We summarize the full learning and inference protocol below.

```
Algorithm 1: TimeClaw External Learning Loop
────────────────────────────────────────────
Input:  Training corpus D = {x₁, …, xₙ}
        Initial artifact state A_s⁰ for each scope s
        Task-valid supervision oracle L_τ
Output: Learned artifact states {A_s} for all scopes

1:  for each training instance x = (z, c, τ, s) ∈ D do
2:      Retrieve current artifact state A_s
3:      Run executor in training mode with exploration-enabled tools
4:      Generate K ≥ 2 candidate paths Π(x) = {π₁, …, π_K}
5:      Filter to task-valid subset: Π_valid(x) ← {πₖ : ŷₖ satisfies contract of τ}
6:      if |Π_valid(x)| ≥ 2 then
7:          Compute path quality: q(πₖ; x) = −L_τ(ŷₖ, y*) for each πₖ ∈ Π_valid(x)
8:          Select preferred path: π*(x) = argmax q(πₖ; x)
9:      else if |Π_valid(x)| = 1 then
10:         Record single-path evidence (no comparative signal)
11:     else
12:         Record failure evidence; skip distillation for this instance
13:         continue
14:     end if
15:     Construct sample-level learning record: nₓ = Summarize(x, Π(x), {q}, π*)
16:     Clean to reusable evidence: eₓ = Clean(nₓ)
17:     Update artifact state: A_s ← Distill(A_s, eₓ)
18:  end for
19:  return {A_s} for all scopes
```

```
Algorithm 2: TimeClaw Inference-Time Reuse
────────────────────────────────────────────
Input:  New task instance x' = (z', c', τ', s')
        Learned artifact states {A_s}
        Runtime-compatible tool set T(x')
Output: Task prediction ŷ

1:  Retrieve learned artifacts for scope s': R_s' = Retrieve(A_s')
2:  Filter to prompt-eligible subset:
        R_s'(x') = {r ∈ M_s' : ι(r) = 1, Match(χ(r), φ(x')) = 1}
3:  Assemble prompt:
        - Official task description for τ'
        - Task boundary and output contract
        - Sample fingerprint φ(x') and profiling signals
        - System-level artifacts: SOUL + scope-local MEMORY from R_s'(x')
        - User-side support: tool notes + skills from A_s'
4:  Expose only runtime-compatible tools T(x')
        (exploration-only tools such as spawn_subagent and evaluate_* are removed)
5:  Execute agent: ŷ = f_agent(x'; R_s', T(x'))
6:  return ŷ
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

**Fallback behavior.** When the executor cannot produce the minimum required number of task-valid candidates (MIN_TRAINING_CANDIDATES = 2), the system does not discard the episode entirely. If exactly one valid candidate exists, the episode records single-path evidence without comparative signal. If no valid candidate exists, the episode records failure evidence (e.g., which tools were attempted, what went wrong) that can still inform future tool guidance. This fallback ensures that even unsuccessful exploration contributes to the learning process, albeit with weaker signal.

The importance of this protocol is methodological rather than procedural: it ensures that the learned signal is grounded in explicit path competition rather than in post-hoc rationalization of a single execution.

### **3.5 Tool-Prior Collapse and Scope-Aware Tool Dropout**

A major obstacle to exploratory tool learning is that the agent prematurely develops strong preferences for a small subset of familiar tools. We refer to this phenomenon as **tool-prior collapse**: during exploration, historically frequent or often-successful tools can dominate future branch generation before sufficient comparative evidence has been gathered, reducing path diversity and weakening subsequent comparison and distillation.

It is important to distinguish tool-prior collapse from inference-time tool preference. After learning, an agent may reasonably favor certain tools based on accumulated evidence — that is a consequence of learning, not a failure. Tool-prior collapse is specifically an exploration-phase pathology: the goal of exploration is broad tool coverage and diverse candidate generation, so entropy decrease in the tool-usage distribution during this phase signals premature convergence rather than informed narrowing.

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

**Theoretical motivation.** The keep probability is designed as a frequency-aware rule anchored at the least-explored tool. The intuition is as follows: for each non-protected tool \(i\) in scope \(s\), we compute its relative over-exploration ratio compared to the least-used tool. A tool that has been used \(n_i\) times when the least-used tool has been used \(n_{\min}\) times has an over-exploration ratio of \((1 + n_i) / (1 + n_{\min})\). The keep probability is the inverse of this ratio raised to a power \(\alpha\), yielding a power-law suppression that is strongest for the most over-explored tools and exactly 1 for the least-explored tools. This ensures that the mechanism never drops under-explored tools — it only suppresses tools that have been used disproportionately more than the cold edge of the scope.

Concretely, for tool \(i\) in a candidate pool under scope \(s\), the keep probability is

\[
p_{\mathrm{keep}}(i \mid s)=
\left(\frac{1+n_{\min}(s)}{1+n_i(s)}\right)^{\alpha},
\]

where \(n_{\min}(s)\) is the minimum usage count among the competing non-protected tools in that scope, and \(\alpha > 0\) controls the strength of reweighting. This form has two important properties. First, it is monotone decreasing in \(n_i(s)\), so historically dominant tools are more likely to be suppressed. Second, tools at the cold edge of the scope, i.e. \(n_i(s)=n_{\min}(s)\), receive keep probability \(1\), so the mechanism explicitly protects under-explored tool families rather than merely injecting uniform randomness. Protected tools bypass dropout altogether:

\[
p_{\mathrm{keep}}(i \mid s)=1 \quad \text{for } i \in \mathcal{T}_{\mathrm{protected}}(s).
\]

**Monotonicity.** For non-protected tools \(i\) and \(j\) in the same scope \(s\), if \(n_i(s) > n_j(s)\), then

\[
p_{\mathrm{keep}}(i \mid s) < p_{\mathrm{keep}}(j \mid s)
\quad \text{for } \alpha > 0.
\]

*Proof.* Since \(n_i(s) > n_j(s) \geq n_{\min}(s)\), we have \(1 + n_i(s) > 1 + n_j(s)\), hence \(\frac{1+n_{\min}(s)}{1+n_i(s)} < \frac{1+n_{\min}(s)}{1+n_j(s)}\). Both ratios lie in \((0, 1]\), and raising to a positive power \(\alpha\) preserves the strict inequality. \(\square\)

The least-used non-protected tools in the scope satisfy \(p_{\mathrm{keep}}=1\). This makes scope-aware dropout a targeted anti-collapse mechanism rather than a generic regularizer: it suppresses dominant tools while guaranteeing visibility of under-explored ones.

### **3.6 Hierarchical Distillation, Conflict-Aware Memory, and Inference-Time Reuse**

TimeClaw externalizes learning into a hierarchical artifact stack rather than updating model parameters. The stack contains several layers with different semantic roles:

- **SOUL**: a fixed human-authored identity and behavior layer;
- **Notes**: sample-level experience records produced from individual exploration runs;
- **MEMORY**: scope-local structured reusable rules distilled from repeated notes;
- **TOOLS**: tool-specific usage guidance summarized from scope-local learning;
- **SKILLS**: scope-local operational procedures or task-solving SOPs;
- **cross-domain**: task-bucket abstractions synthesized across multiple scopes.

This hierarchy gives TimeClaw a concrete medium for persistent learning. Notes capture raw experience at sample granularity, but they are not the main runtime memory surface. Instead, they serve as distillation sources from which more stable scope-level artifacts are derived. The key runtime layer is `MEMORY`, whose entries are structured reusable rules rather than free-form reflections. A memory rule can be written abstractly as a 9-tuple:

\[
r = (\kappa,\sigma,\chi,\mathcal{T}^{+},\mathcal{T}^{-},\rho,\mathcal{E},c,\iota),
\]

where:

- \(\kappa\) is the rule kind (e.g., tool preference, condition-action, avoidance);
- \(\sigma\) is a concise summary of the rule;
- \(\chi\) denotes applicability conditions (e.g., data characteristics, task subtypes);
- \(\mathcal{T}^{+}\) and \(\mathcal{T}^{-}\) are preferred and avoided tool sets;
- \(\rho\) is the rationale explaining why this rule holds;
- \(\mathcal{E}\) is supporting evidence from exploration episodes;
- \(c \in [0, 1]\) is confidence;
- \(\iota \in \{0,1\}\) indicates whether the rule is currently safe to inject into the prompt.

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

### 4.1 为什么不用 `self-improving`

当前 TimeClaw 的学习并不是 test-time online learning、边推理边写回策略、一边部署一边参数更新。更准确的说法是：exploration-trained, externally learned, artifact-reuse at inference time。这和 intro 里的 `without online adaptation` 必须保持严格一致。

### 4.2 为什么 `path-quality supervision` 是主方法点

TimeClaw 的关键判断是：对 metric-evaluable tasks，是否完成任务只是最低要求；更重要的是哪条路径在任务指标上更优。对比对象是 agentic 场景中的二值监督（成功/失败），而非 RL 的 reward shaping。深度学习模型可以通过梯度直接优化损失函数，不需要路径比较；但 agent 无法对执行路径做反向传播，因此需要显式的路径质量比较。

### 4.3 为什么 `tool_dropout` 可以写进 methodology 主体

因为在当前实现里，它不是一个实验 trick，而是训练阶段真实使用的探索控制机制：只作用于 training-time sub-agent tool exposure；基于 scope-level usage 统计；会影响可见工具子集；直接影响候选路径多样性。

### 4.4 为什么 `cross-domain` 不能写得过重

当前仓库里 `cross-domain` 在存储和蒸馏层真实存在，但当前 prompt 注入主路径并不以它为中心。最安全的写法是把它写成 distillation hierarchy 的一层，不把它写成当前推理的核心控制器。

---

## 5. Appendix-Ready 补充：Prompt 组装流程与信息来源

这部分不一定全部放进主文，但非常适合后续作为 appendix 或 supplementary。

---

### 5.1 Prompt 组装总览

当前实现中，executor 的 prompt 可以概括为 6 个来源的组合：

1. **Official task description**
2. **Task boundary and output contract**
3. **Fingerprint and profiling signals**
4. **System-level learned artifacts**
5. **User-side operational support**
6. **Visible tool catalog**

---

### 5.2 信息来源 1：Official task description

来源：`timeclaw/benchmark/mtbench_support/task_descriptions.py`, `timeclaw/benchmark/mtbench_prompts.py`

作用：将 benchmark task 转成规范化的 system/user task prompt；保留官方任务定义；让训练和测试在任务语义层面对齐。

渲染结果示意：

```text
System prompt:
<official task instruction>

User prompt:
<sample-specific question / forecast instruction / QA prompt>
```

---

### 5.3 信息来源 2：Task boundary and output contract

来源：`timeclaw/agents/executor.py` — `_build_task_boundary_section`, `_build_main_output_contract_section`

作用：明确什么是 task-valid output；防止 agent 输出错误类型答案；在 training / inference 两种模式下给出不同的 completion contract。

例如：forecasting 任务必须输出数值数组；indicator 任务必须输出指定 key 的 JSON；training mode 不能直接用 task answer 收尾，而要输出 `learning_summary`。

---

### 5.4 信息来源 3：Fingerprint and profiling signals

来源：`timeclaw/fingerprint/extractor.py`, `timeclaw/pipeline/runner.py`, `timeclaw/agents/executor.py`

内容：data shape, length, missing ratio, basic stats, stationarity, trend-related profiling, anomaly count 或其他 profiling 结果。

渲染结果示意：

```text
Sample Fingerprint
{"length":168,"num_channels":1,"missing_ratio":0.0,"trend":"upward",...}

Profiling
- stationarity_test: ...
- detect_trend: ...
```

---

### 5.5 信息来源 4：System-level learned artifacts

来源：`timeclaw/pipeline/runner.py`, `timeclaw/memory/hierarchical.py`

当前主注入层包括：`SOUL`, scope-local `MEMORY`

渲染结果示意：

```text
## MEMORY — Learned Insights

### finance_aligned_short_forecast Specific
- Rule: when short-horizon movement is smooth and recent local trend is stable, prefer ...
- Rule: when text cue is weak but series continuity is strong, avoid ...
```

---

### 5.6 信息来源 5：User-side operational support

来源：`timeclaw/memory/hierarchical.py`, `timeclaw/agents/executor.py`

当前主注入层包括：`tool notes`, `skills`

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

### 5.7 信息来源 6：Visible tool catalog

来源：`timeclaw/agents/prompts/executor_prompt.py`, `timeclaw/clawhub/tool_registry.py`

训练 / 测试差异：

- training: 暴露 `spawn_subagent`, `evaluate_against_gt`, `evaluate_batch_against_gt`
- inference: 移除这些 training-only tools，只保留任务求解所需工具

---

## 6. Appendix-Ready 补充：流程与训练产物结构

---

### 6.1 总流程

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

**TimeClaw learns externally from exploration on a disjoint MTBench-compatible corpus, distills the resulting path-level experience into structured artifacts, and reuses those artifacts when evaluated on the TimeART-aligned MTBench subset.**

---

### 6.2 训练阶段的主要产物

按当前实现，训练后会形成几类关键产物：

1. `train_results`
2. `trace`
3. `candidate traces`
4. `structured learning analysis`
5. `scope history snapshots`

---

### 6.3 `train_results`

来源：`.timeclaw/train_results/`

作用：按 `(dataset, task, sample)` 记录训练 episode 的可恢复结构；保存 `learning_summary`、winner metadata、metrics、tool chain 等；支撑后续 resume、分析和附录可视化。

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

### 6.4 `trace`

来源：`.timeclaw/traces/*.jsonl`

作用：记录一次 executor run 的逐步执行轨迹；保留 tool calls、观测、sub-agent 行为和 dropout 信息。

渲染结果示意：

```json
{"agent":"executor","step":"spawn_subagents","n_tasks":2,...}
{"tool":"chronos2_forecast","success":true,"observation":{...}}
{"tool":"evaluate_batch_against_gt","success":true,"observation":{...}}
```

---

### 6.5 `candidate traces`

来源：从 `execution_log` 中提取，在 learning analysis、winner selection 和 memory distillation 中被使用。

作用：将候选路径显式化；保留每条路径的 tool chain、prediction、metrics、public reasoning；支撑 path-level comparison。

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

### 6.6 `structured learning analysis`

来源：`timeclaw/benchmark/mtbench_support/learning_analysis.py`

作用：把单样本探索结果转成紧凑、可蒸馏的学习材料；为 `Notes -> MEMORY` 的蒸馏提供更干净的中间表示。强制区分：sample conditions, winner evidence, failure signals, reusable rule。

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

### 6.7 `scope history snapshots`

来源：`.timeclaw/hierarchical_memory/history/...`

作用：记录某个 scope 在某个蒸馏时刻的可复用学习状态；保留该 scope 的 memory entries、tool notes、skill content，以及同 bucket 的 cross-domain summary；让学习状态具备时间切片和可追溯性。

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

## 7. Appendix-Ready 补充：Memory Rule 9-Tuple 形式化

以下是 memory rule 9-tuple 的完整形式化说明，适合放入附录。

A memory rule \(r\) is a structured 9-tuple:

\[
r = (\kappa, \sigma, \chi, \mathcal{T}^{+}, \mathcal{T}^{-}, \rho, \mathcal{E}, c, \iota)
\]

| Symbol | Name | Type | Description |
| --- | --- | --- | --- |
| \(\kappa\) | kind | categorical | Rule category: tool preference, condition-action, avoidance, general insight |
| \(\sigma\) | summary | text | Concise natural-language summary of the rule |
| \(\chi\) | conditions | set of predicates | Applicability conditions: data characteristics, task subtypes, scope constraints |
| \(\mathcal{T}^{+}\) | preferred tools | set of tool IDs | Tools that should be favored when this rule applies |
| \(\mathcal{T}^{-}\) | avoided tools | set of tool IDs | Tools that should be avoided when this rule applies |
| \(\rho\) | rationale | text | Explanation of why this rule holds, grounded in exploration evidence |
| \(\mathcal{E}\) | evidence | list of episode refs | Supporting evidence from exploration episodes (sample IDs, metrics, paths) |
| \(c\) | confidence | \([0, 1]\) | Confidence score, updated as more evidence accumulates |
| \(\iota\) | injectable | \(\{0, 1\}\) | Whether the rule is currently safe to inject into the runtime prompt |

The `Update` operator acts on the memory state \(\mathcal{M}_s\) given new evidence \(e_x\) through one of four operations:

1. **Append**: if \(e_x\) introduces a genuinely new rule not covered by any existing \(r \in \mathcal{M}_s\), create a new entry with initial confidence.
2. **Merge**: if \(e_x\) provides near-duplicate evidence for an existing rule, merge the evidence and update confidence upward.
3. **Strengthen**: if \(e_x\) provides additional supporting evidence for an existing rule from a different sample or condition, add to \(\mathcal{E}\) and increase \(c\).
4. **Conflict**: if \(e_x\) contradicts an existing rule, register a conflict relation. The conflicting rule may have its \(\iota\) set to 0 (non-injectable) until the conflict is resolved by further evidence.

The separation between storage (\(\iota = 0\)) and injection (\(\iota = 1\)) ensures that the runtime prompt only contains rules that have sufficient confidence and no unresolved conflicts, while the full memory state preserves all evidence for future distillation.

---

## 8. 最终自检清单

### 8.1 与 intro 保持一致

- 使用 `external learning`（弱化 benchmark-grounded 复合用法）
- 使用 `exploration phase / inference phase`
- 使用 `metric-supervised path learning`
- 使用 `without online adaptation`
- 使用 `tool-prior collapse`（含 "prematurely" 限定）
- path-quality supervision 对比对象为 agentic binary supervision

### 8.2 与 related work 保持一致

- 不把 TimeClaw 写成"又一个 time-series agent"
- 不把它缩成纯 `tool learning paper`
- 不把 benchmark 写成主要 novelty
- 继续强调：path-level comparative learning, external artifact distillation, inference-time reuse

### 8.3 不过度声称

- 不说 `the first paradigm`
- 不说 `fully self-improving`
- 不说 `cross-domain` 已经成为当前推理主控层
- 不把 legacy split 说成最终主协议

---

## 9. 一句话总结

**TimeClaw turns exploratory execution on time-series tasks into an external learning process, where candidate paths are compared under task-valid supervision, distilled into reusable artifacts, and later reinjected at inference time without online adaptation.**
