# TimeClaw Core Sections LaTeX Draft

更新时间：2026-04-16

这份文档把当前已经完成的 `Introduction`、`Related Work`、`Methodology` 有机整合为一份更接近论文正文的 LaTeX 版终稿，并补上 `Experiments` 的可直接扩展结构。

说明：

- 这份稿子优先保持和当前仓库事实一致，而不是为了“完整”去硬凑 claim。
- `Methodology` 仍保持你前面确认过的主线：`exploration phase` 学习，`inference phase` 只复用 artifact，不做在线学习。
- `Experiments` 章节按 `MTBench` 主线组织，并显式预留与 `TimeART` 对齐的 head-to-head 设置。

建议 LaTeX 依赖：

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
```

---

## LaTeX Body

```latex
\section{Introduction}

Large language model (LLM) agents are evolving from one-shot assistants into persistent systems that interact with tools, maintain external state, and support increasingly complex workflows. This trend is especially promising for scientific domains such as time-series analysis, where solving a task often requires combining statistical computation, contextual reasoning, and structured decision making. Recent work has made substantial progress on time-series benchmarks and tool-augmented time-series agents~\cite{chen2025mtbench,weng2026temporalbench,wu2026timeart,liu2025tsagent,zhao2025timeseriesscientist}. However, most existing time-series agent systems still focus primarily on \emph{execution-time capability}: they improve how an agent solves the current task, but only partially address how experience from exploratory execution should be accumulated and reused across future tasks.

This limitation is particularly important for numeric and verifiable time-series tasks. In forecasting, indicator prediction, and related settings, multiple execution paths may all produce task-valid answers, yet differ substantially in quantitative quality under metrics such as MAE or RMSE. In such cases, binary notions of success are insufficient for learning. The question is not merely whether an agent can complete a task, but whether it can identify, preserve, and reuse \emph{better execution paths}. This suggests that agent learning in time-series domains should move beyond completion-only supervision toward \emph{path-quality-aware learning}, a distinction that is only partially captured by existing time-series forecasting and reasoning agents~\cite{wu2026timeart,liu2025tsagent,zhao2025timeseriesscientist,castrone2026castr1}.

A second challenge is that exploratory learning itself is unstable. During training, an agent often develops a strong bias toward a small set of familiar tools that frequently succeed early, then overcommits to them and suppresses further exploration. We refer to this failure mode in benchmark-grounded time-series agent learning as \emph{tool-prior collapse}. Once this happens, candidate diversity shrinks, comparative supervision weakens, and external distillation becomes less informative. For benchmark-grounded scientific tasks, this creates a fundamental bottleneck: without sufficiently diverse exploratory trajectories, the agent cannot reliably learn which behaviors are actually better.

In this paper, we propose \textbf{TimeClaw}, a benchmark-grounded external learning framework for time-series agents. Rather than treating each benchmark instance as an isolated execution problem, TimeClaw turns exploration-time execution into a reusable source of scientific experience. Its core design is an external learning loop: the agent explores multiple candidate execution paths, compares them under task-valid supervision, distills the resulting experience into editable external artifacts, and reinjects those artifacts into future problem solving. In this way, benchmark execution becomes not only an evaluation target, but also a metric-supervised learning process. Crucially, TimeClaw separates learning from inference: learning happens during a dedicated exploration phase, while downstream inference reuses distilled artifacts without further online adaptation.

TimeClaw is built around three key ideas. First, we introduce \textbf{metric-supervised path learning} for numeric and verifiable time-series tasks, enabling the agent to learn not only whether a task is solvable, but which candidate path achieves better quantitative outcomes. Second, to mitigate tool-prior collapse, we propose a \textbf{forced exploration protocol} with \textbf{scope-aware tool dropout}, which improves exploratory tool learning by reducing overreliance on dominant tools and increasing candidate-path diversity. Third, we externalize learning through a hierarchical artifact stack that stores reusable experience outside model weights, including path-level lessons, tool guidance, and higher-level reusable skills. In this sense, TimeClaw intersects with tool learning, but goes beyond tool-use correctness by learning from comparative path quality and reusing distilled experience across tasks~\cite{schick2023toolformer,patil2023gorilla,qin2023toolllm,memskill2026,asda2026,skillsd2026}.

More broadly, TimeClaw can be viewed as a scientific-domain instantiation of the broader move from stateless tool use toward persistent, learning-enabled agent systems. Unlike open-world agent runtimes that emphasize continuous operation, personalization, or computer-use autonomy, TimeClaw specializes this broader systems philosophy to a benchmark-grounded scientific setting, where experience accumulation is comparable, verifiable, and reusable. This positioning allows us to study not only whether a time-series agent can solve benchmark tasks, but whether it can \emph{learn from exploratory execution in a measurable and interpretable way}~\cite{openclawrl2026,cuaskill2026,clawkeeper2026}.

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Positioning TimeClaw against representative neighboring lines of work. TimeClaw is distinguished not by having more components, but by jointly combining benchmark-grounded learning, path-level comparative supervision, external artifact distillation, and inference-time reuse in a time-series agent setting.}
\label{tab:positioning}
\begin{tabular}{lccccc}
\toprule
\textbf{Line of Work} & \textbf{Time-series} & \textbf{Learns from} & \textbf{Path-level} & \textbf{External artifact} & \textbf{Inference reuse} \\
 & \textbf{agent setting} & \textbf{execution} & \textbf{comparison} & \textbf{distillation} & \textbf{without online adaptation} \\
\midrule
MTBench / TemporalBench~\cite{chen2025mtbench,weng2026temporalbench} & \checkmark &  &  &  &  \\
TimeART / TS-Agent / TimeSeriesScientist~\cite{wu2026timeart,liu2025tsagent,zhao2025timeseriesscientist} & \checkmark &  &  &  &  \\
Cast-R1~\cite{castrone2026castr1} & \checkmark & \checkmark &  &  &  \\
MemSkill / ASDA / Skill-SD~\cite{memskill2026,asda2026,skillsd2026} &  & \checkmark &  & \checkmark & \checkmark \\
OpenClaw-style runtimes~\cite{openclawrl2026,cuaskill2026,clawkeeper2026} &  & \checkmark &  & \checkmark & \checkmark \\
\textbf{TimeClaw} & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{table*}

Our contributions are summarized as follows:
\begin{itemize}[leftmargin=1.5em]
    \item We formulate \textbf{benchmark-grounded external learning} for time-series agents, and instantiate it in TimeClaw as a learning architecture that separates exploration-time experience acquisition from inference-time reuse.
    \item We identify \textbf{path-quality supervision} as a more suitable learning signal than completion-only supervision for numeric and verifiable time-series tasks, and operationalize this insight through metric-supervised path learning.
    \item We identify \textbf{tool-prior collapse} in benchmark-grounded time-series agent learning as a major obstacle to exploratory tool learning and propose forced exploration with scope-aware tool dropout to improve candidate-path diversity and supervision quality.
    \item We build a dedicated exploratory learning setup for time-series agents and show how distilled external artifacts can improve benchmark-grounded reasoning without parameter updates or online adaptation.
\end{itemize}


\section{Related Work}

Recent work related to TimeClaw spans three closely connected directions: time-series agent systems, agent learning through tools and external artifacts, and persistent open agent runtimes.

\paragraph{Time-series agent systems.}
Recent work has substantially strengthened time-series agents through tool use, iterative reasoning, forecasting workflow orchestration, and multi-agent collaboration. TimeART learns strategic tool use for time-series reasoning through tool augmentation and expert trajectories~\cite{wu2026timeart}. TS-Agent emphasizes grounded reasoning over raw numeric sequences through iterative evidence gathering~\cite{liu2025tsagent}, while TimeSeriesScientist and other forecasting-oriented frameworks focus on automating forecasting pipelines and improving end-to-end workflow quality~\cite{zhao2025timeseriesscientist,garza2025timecopilot,yeh2025llmagentsforecasting}. In parallel, systems such as TS-Debate and MAS4TS improve performance through debate, modality specialization, or multi-agent decomposition within a single task instance~\cite{trirat2026tsdebate,ruan2026mas4ts}. These approaches primarily optimize execution-time capability. TimeClaw differs in objective: it treats exploratory execution itself as a reusable learning signal. A concise summary is that TimeART teaches a time-series reasoning model to use tools, whereas TimeClaw teaches an executing agent to learn from tool use.

\paragraph{Tool learning, external memory, and skill distillation.}
Beyond the time-series domain, recent work has increasingly treated tool use, external memory, and skill artifacts as explicit learning media for agents. Toolformer, Gorilla, and ToolLLM study how models can invoke external tools more effectively, but mainly focus on tool-use competence and API grounding~\cite{schick2023toolformer,patil2023gorilla,qin2023toolllm}. A complementary line of work studies persistent learning through external artifacts. MemSkill explores evolving memory skills; ASDA distills structured skill files that can be injected at inference time; Skill-SD shows how multi-turn trajectories can be compressed into reusable guidance~\cite{memskill2026,asda2026,skillsd2026}. Cast-R1 brings learning-based tool use closer to time-series forecasting by formulating forecasting as sequential decision making with reflective tool use~\cite{castrone2026castr1}. TimeClaw is complementary to these directions: it is not the first work on tool learning or external artifact learning in general, but it is among the first to combine them into a benchmark-grounded external learning process for time-series agents, where candidate execution paths are explicitly compared, distilled, and reused at inference time without online adaptation.

\paragraph{Persistent open agent runtimes.}
At the systems level, 2026 has also seen the rapid rise of persistent open agent runtimes, exemplified by OpenClaw-style ecosystems and related work on online training, skill development, runtime governance, and safety~\cite{openclawrl2026,cuaskill2026,clawkeeper2026,clawsafety2026,safeclawr2026}. These systems are characterized by long-lived state, reusable skills, plugins, and runtime control layers, but are typically designed for open-world automation rather than benchmark-grounded scientific reasoning. TimeClaw shares the broader systems philosophy of persistence and reusable external intelligence, but specializes it to a verifiable scientific setting, where persistence serves benchmark-grounded experience accumulation rather than open-world autonomy.

Finally, TimeClaw is built on benchmark substrates such as MTBench and TemporalBench~\cite{chen2025mtbench,weng2026temporalbench}, but does not aim to contribute another evaluation benchmark. Instead, it asks how benchmark-grounded execution should be turned into a reusable learning process. Its novelty lies in combining tool-augmented execution, path-level comparative learning, external artifact distillation, and cross-task reuse into a single external learning framework for time-series agents.


\section{Methodology}

\subsection{Problem Setup}

We study \textbf{benchmark-grounded external learning} for time-series agents. Each task instance is written as
\begin{equation}
x = (z, c, \tau, s),
\end{equation}
where $z$ denotes the time-series input, $c$ denotes optional multimodal context such as text, $\tau$ is the task type, and $s$ is the scope under which reusable experience is stored. Rather than committing to a single execution chain, the agent may explore a set of candidate paths
\begin{equation}
\Pi(x) = \{\pi_1,\ldots,\pi_K\},
\end{equation}
where each path $\pi_k$ is a structured trajectory of tool selections, intermediate observations, and a final task output. Let
\begin{equation}
\hat y_k = \mathrm{Exec}(x,\pi_k)
\end{equation}
denote the output produced by path $\pi_k$. A path is \textbf{task-valid} if $\hat y_k$ satisfies the output contract of task $\tau$, e.g. a forecast array for forecasting, a structured JSON object for indicator prediction, or a legal label for trend and correlation tasks.

The key methodological issue is that, for numeric and verifiable time-series tasks, multiple paths may all be task-valid while still differing substantially in quality. We therefore distinguish \emph{completion} from \emph{path quality}. Given task-valid supervision, the quality of a candidate path is written as
\begin{equation}
q(\pi_k; x) = -\mathcal{L}_{\tau}(\hat y_k, y^\star),
\end{equation}
where $y^\star$ is the task target and $\mathcal{L}_{\tau}$ is task-dependent, e.g. MAE or RMSE for continuous outputs and a correctness-based loss for discrete outputs. The preferred path is then
\begin{equation}
\pi^\star(x) = \arg\max_{\pi_k \in \Pi_{\mathrm{valid}}(x)} q(\pi_k; x),
\end{equation}
where $\Pi_{\mathrm{valid}}(x)\subseteq \Pi(x)$ is the set of task-valid candidates. If no candidate is validated, the learning episode records failure evidence rather than forcing an artificial winner.

This formalization clarifies why completion-only supervision is insufficient. If two paths $\pi_a$ and $\pi_b$ both yield task-valid outputs, then a binary success signal maps both to the same supervision value even when
\begin{equation}
\mathcal{L}_{\tau}(\hat y_a, y^\star) \neq \mathcal{L}_{\tau}(\hat y_b, y^\star).
\end{equation}
Thus, completion-only supervision cannot distinguish paths that are all executable but quantitatively unequal. TimeClaw therefore learns from \textbf{path-quality supervision}, not only from completion.

This is not a separate theory claim; it is a methodological justification for the learning signal. For metric-evaluable tasks, any learning rule that aims to preserve stronger future execution must distinguish between valid-but-unequal paths rather than collapsing them into the same binary success label.

TimeClaw separates the agent lifecycle into two phases. In the \textbf{exploration phase}, the agent generates and compares candidate execution paths under task-valid supervision. In the \textbf{inference phase}, the agent does not learn online; instead, it reuses distilled external artifacts accumulated during exploration. This separation keeps the learning process auditable, reusable, and benchmark-grounded.

\subsection{Data Protocol}

TimeClaw adopts a two-corpus protocol. For learning, we construct an \textbf{independent MTBench-compatible corpus} whose task contracts, schemas, and scope structure are aligned with MTBench, but whose samples are disjoint from the official benchmark. This corpus is used only for exploration-time experience acquisition. For evaluation, all reported experiments use a \textbf{TimeART-aligned subset of MTBench}. The role of the independent corpus is to expose whether the agent can acquire reusable execution knowledge; the role of the MTBench subset is to provide a stable and directly comparable evaluation target.

This protocol is preferable to simply splitting MTBench into train and test subsets. A benchmark split is sufficient for measuring temporal generalization, but it is less well suited for isolating whether exploratory agent learning transfers beyond the benchmark samples themselves. By learning on an aligned but disjoint corpus and evaluating on the same MTBench task subset used by TimeART, TimeClaw centers the methodology on reusable learning while preserving direct head-to-head comparability.

\subsection{TimeClaw Learning Framework}

TimeClaw instantiates external learning as a four-stage loop:
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Explore}: generate multiple candidate execution paths for the same task.
    \item \textbf{Compare}: evaluate task-valid candidates under benchmark-grounded supervision.
    \item \textbf{Distill}: compress path-level evidence into reusable external artifacts.
    \item \textbf{Reinject}: reuse the distilled artifacts during later inference without online adaptation.
\end{enumerate}

This \texttt{Explore $\rightarrow$ Compare $\rightarrow$ Distill $\rightarrow$ Reinject} loop is the main methodological object of the paper. Prior time-series agent systems mainly strengthen execution-time capability within a task instance. TimeClaw instead treats execution itself as a reusable learning signal. The key transition is from candidate-path exploration to artifact update. For a single training instance $x$, TimeClaw first constructs a sample-level learning record
\begin{equation}
n_x = \mathrm{Summarize}\!\left(x,\Pi(x), \{q(\pi_k;x)\}_{k=1}^{K}, \pi^\star(x)\right),
\end{equation}
which captures the strongest validated path, decisive evidence, and failure signals from weaker paths. However, the stored learning signal is not the raw reflective text emitted by the agent. TimeClaw further rewrites candidate-level comparison into reusable execution evidence,
\begin{equation}
e_x = \mathrm{Clean}\!\left(n_x\right),
\end{equation}
where $\mathrm{Clean}(\cdot)$ removes task-answer leakage, framework-control language, and orchestration-specific wording that would not transfer to future tasks. The cleaned evidence is then used to update the external artifact state associated with scope $s$:
\begin{equation}
\mathcal{A}_{s}^{(m+1)} = \mathrm{Distill}\!\left(\mathcal{A}_{s}^{(m)}, e_x\right),
\end{equation}
where $\mathcal{A}_{s}$ denotes the current external state for that scope. The update target is not a model parameter vector but a structured artifact state containing notes, reusable rules, tool guidance, and operational procedures. This is precisely why TimeClaw is an \textbf{external learning} framework rather than an online parameter-learning runtime.

\paragraph{Learning protocol.}
\begin{enumerate}[leftmargin=1.5em]
    \item Run the executor in training mode on task $x$.
    \item Generate multiple task-valid candidate paths.
    \item Compare candidate outputs under task-valid supervision.
    \item Select the strongest validated path, or record that no path is validated.
    \item Distill sample-level evidence into external artifacts under scope $s$.
    \item Retrieve artifacts relevant to scope $s$ for later inference.
\end{enumerate}

\subsection{Exploration-Time Agent Learning}

The exploration phase is implemented through a constrained executor that converts autonomous execution into explicit candidate-path learning. In training mode, the main agent is allowed to access exploration-only orchestration tools, including \texttt{spawn\_subagent} for branch generation and \texttt{evaluate\_*} tools for post-hoc comparison against ground truth. However, this is not unconstrained free-form exploration. TimeClaw enforces a structured training contract:
\begin{itemize}[leftmargin=1.5em]
    \item the main agent must produce \textbf{at least two task-valid candidate paths};
    \item these candidates must arise from genuine branch exploration rather than stylistic paraphrases of the same path;
    \item when a prior successful path family already exists for the scope, the first exploration round must include both a prior-guided candidate and an alternative candidate;
    \item candidate comparison must occur before training completion;
    \item \texttt{evaluate\_*} tools are gated on candidate availability, and \texttt{learning\_summary} is gated on completed comparison;
    \item the agent may finish only by returning a \texttt{learning\_summary}, not by treating the training run as ordinary inference.
\end{itemize}

This contract matters because it turns what would otherwise be a loose exploratory run into a controlled learning episode. A trajectory is considered complete only after candidate generation and task-valid comparison are already satisfied. The resulting \texttt{learning\_summary} is not the final task answer. Instead, it records the selected path, the decisive evidence behind that selection, and a concise reusable lesson that can later be distilled into external artifacts.

To keep candidate paths semantically comparable, intermediate tool outputs are also tracked as typed artifacts with coordinate provenance. As a result, downstream operations interpret values relative to the current derived artifact rather than blindly reusing indices from the original series. This constraint is important for slices, rolling windows, and forecast artifacts, where coordinate semantics would otherwise be ambiguous.

\subsection{Tool-Prior Collapse and Scope-Aware Tool Dropout}

A major obstacle to exploratory tool learning is that the agent quickly develops strong preferences for a small subset of familiar tools. We refer to this phenomenon as \textbf{tool-prior collapse}: during benchmark-grounded exploration, historically frequent or often-successful tools can dominate future branch generation, reducing path diversity and weakening subsequent comparison and distillation.

Let $n_i(s)$ denote the historical usage count of tool $i$ under scope $s$, and let $p_i(s)$ be the normalized usage distribution over competing tools in that scope:
\begin{equation}
p_i(s)=\frac{n_i(s)}{\sum_j n_j(s)}.
\end{equation}
Then one operational signature of collapse is decreasing tool-support diversity, which can be tracked by the scope-level entropy
\begin{equation}
\mathcal{H}(s) = - \sum_i p_i(s)\log p_i(s).
\end{equation}
When a small number of tools dominate, $\mathcal{H}(s)$ decreases and the explored candidate set becomes structurally narrower. In that regime, path comparison becomes less informative because multiple branches differ only superficially while relying on the same dominant tool family.

TimeClaw mitigates this problem with \textbf{scope-aware tool dropout}. The mechanism operates during \textbf{training-time sub-agent tool exposure}. For each scope, the system tracks historical tool usage and computes a keep probability for each candidate tool relative to the competing tools in that scope. Frequently used tools receive a lower keep probability, while underused tools remain more visible. Task-critical tools and explicitly hinted tools are protected through a \texttt{never\_drop} rule. The purpose is not regularization for its own sake, but to preserve branch diversity so that the agent can continue to generate meaningfully different candidate paths rather than repeatedly collapsing onto the same familiar execution pattern.

Concretely, for tool $i$ in a candidate pool under scope $s$, the keep probability is
\begin{equation}
p_{\mathrm{keep}}(i \mid s)=
\left(\frac{1+n_{\min}(s)}{1+n_i(s)}\right)^{\alpha},
\end{equation}
where $n_{\min}(s)$ is the minimum usage count among the competing non-protected tools in that scope, and $\alpha > 0$ controls the strength of reweighting. This form has two important properties. First, it is monotone decreasing in $n_i(s)$, so historically dominant tools are more likely to be suppressed. Second, tools at the cold edge of the scope, i.e. $n_i(s)=n_{\min}(s)$, receive keep probability $1$, so the mechanism explicitly protects under-explored tool families rather than merely injecting uniform randomness. Protected tools bypass dropout altogether:
\begin{equation}
p_{\mathrm{keep}}(i \mid s)=1 \quad \text{for } i \in \mathcal{T}_{\mathrm{protected}}(s).
\end{equation}

Moreover, for non-protected tools $i$ and $j$ in the same scope $s$, if $n_i(s) > n_j(s)$, then the keep rule implies
\begin{equation}
p_{\mathrm{keep}}(i \mid s) < p_{\mathrm{keep}}(j \mid s)
\quad \text{for } \alpha > 0.
\end{equation}
The least-used non-protected tools in the scope satisfy $p_{\mathrm{keep}}=1$. Hence the dropout rule is not uniform noise: it is an order-preserving reweighting that suppresses dominant tools while guaranteeing visibility of under-explored ones.

\subsection{Hierarchical Distillation, Conflict-Aware Memory, and Inference-Time Reuse}

TimeClaw externalizes learning into a hierarchical artifact stack rather than updating model parameters. The stack contains several layers with different semantic roles:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{SOUL}: a fixed human-authored identity and behavior layer;
    \item \textbf{Notes}: sample-level experience records produced from individual exploration runs;
    \item \textbf{MEMORY}: scope-local structured reusable rules distilled from repeated notes;
    \item \textbf{TOOLS}: tool-specific usage guidance summarized from scope-local learning;
    \item \textbf{SKILLS}: scope-local operational procedures or task-solving SOPs;
    \item \textbf{cross-domain}: task-bucket abstractions synthesized across multiple scopes.
\end{itemize}

This hierarchy gives TimeClaw a concrete medium for persistent learning. Notes capture raw experience at sample granularity, but they are not the main runtime memory surface. Instead, they serve as distillation sources from which more stable scope-level artifacts are derived. The key runtime layer is \texttt{MEMORY}, whose entries are structured reusable rules rather than free-form reflections. A memory rule can be written abstractly as
\begin{equation}
r = (\kappa,\sigma,\chi,\mathcal{T}^{+},\mathcal{T}^{-},\rho,\mathcal{E},c,\iota),
\end{equation}
where $\kappa$ is the rule kind, $\sigma$ is a concise summary, $\chi$ denotes applicability conditions, $\mathcal{T}^{+}$ and $\mathcal{T}^{-}$ are preferred and avoided tool sets, $\rho$ is the rationale, $\mathcal{E}$ is supporting evidence, $c$ is confidence, and $\iota \in \{0,1\}$ indicates whether the rule is currently safe to inject into the prompt.

This representation matters because TimeClaw does not update memory by naive append. Given cleaned evidence $e_x$, the scope-local memory state is updated through a conflict-aware operator
\begin{equation}
\mathcal{M}_{s}^{(m+1)} = \mathrm{Update}\!\left(\mathcal{M}_{s}^{(m)}, e_x\right),
\end{equation}
where $\mathrm{Update}$ may append a new rule, merge near-duplicate evidence into an existing rule, strengthen a supported rule, or register a conflict relation between competing rules. When an item is not yet stable enough for prompt-time reuse, it can remain stored but be marked non-injectable through $\iota=0$. This separation between storage and injection is the core mechanism by which TimeClaw preserves long-horizon experience without forcing every intermediate belief into the runtime prompt.

The same logic extends beyond \texttt{MEMORY}. \texttt{TOOLS} distills tool-specific usage advice from repeated execution evidence, while \texttt{SKILLS} compresses stronger scope-local procedures into reusable SOP-like guidance. To preserve learning over time, the system also records historical snapshots of notes, current memory state, tool notes, skill text, and cross-domain abstractions. These snapshots provide a long-term audit trail of how reusable experience evolves rather than reducing learning to the latest prompt state.

During inference, the system reuses these artifacts without online learning. Formally, inference on a new instance $x'$ under scope $s$ is written as
\begin{equation}
\hat y = f_{\mathrm{agent}}\!\left(x';\, \mathrm{Retrieve}(\mathcal{A}_{s}), \mathcal{T}(x')\right),
\end{equation}
where $\mathrm{Retrieve}(\mathcal{A}_{s})$ returns the relevant learned artifacts for the scope and $\mathcal{T}(x')$ is the runtime-compatible visible tool set. In the current implementation, retrieval is filtered by scope, applicability, and injection status. Abstractly, the prompt-eligible memory subset is
\begin{equation}
\mathcal{R}_{s}(x')=
\{r \in \mathcal{M}_{s} : \iota(r)=1,\; \mathrm{Match}(\chi(r), \phi(x'))=1\},
\end{equation}
where $\phi(x')$ denotes the sample fingerprint and profiling view of the current instance.

Prompt construction first assembles an official task description, then provides the task boundary and output contract, then injects a sample fingerprint together with profiling signals, and finally reinjects the learned artifact layers. In the present implementation, \texttt{SOUL} and scope-local \texttt{MEMORY} are injected at the system level, while tool notes and skills are injected as reusable support guidance; the visible tool catalog is further filtered by modality and runtime compatibility.

The train-test difference is crucial. Training exposes exploration-only tools such as \texttt{spawn\_subagent} and \texttt{evaluate\_*} because the goal is to compare candidate paths. Inference removes these training-only tools and retains only the tools needed to solve the task itself. Thus, the runtime agent is not continuing to learn; it is solving the task under a prompt enriched by previously distilled experience. The \texttt{cross-domain} layer exists in storage and distillation, but it is not the main runtime injection path in the present implementation. Accordingly, the primary reuse mechanism is best understood as \textbf{scope-local artifact reinjection}, not as a fully cross-domain reasoning controller.

\section{Experiments}

We design the experiments to answer four questions: (1) Does TimeClaw improve benchmark-grounded time-series reasoning on MTBench? (2) Does metric-supervised path learning matter beyond completion-only or single-path execution? (3) Does scope-aware tool dropout mitigate tool-prior collapse and improve transfer? (4) Which external artifacts are responsible for downstream gains?

\subsection{Experimental Setup}

\paragraph{Benchmark.}
We evaluate on the \textbf{TimeART-aligned MTBench subset}~\cite{chen2025mtbench,wu2026timeart}, using the official prompts, task contracts, and task-specific metrics on the shared task family. Concretely, all reported experiments follow the MTBench subsets covered by TimeART: weather forecasting and trend tasks, finance forecasting and trend tasks, finance indicator prediction, and finance QA/correlation tasks. In the implementation, finance indicator evaluation is expanded into \texttt{MACD} and \texttt{upper Bollinger band} prediction, yielding an 18-scope evaluation suite. This gives a like-for-like comparison target while remaining inside the official MTBench task family.

\paragraph{Learning protocol.}
TimeClaw learns on an independently constructed MTBench-compatible exploration corpus whose schemas and task contracts are aligned with MTBench but whose samples are disjoint from the official benchmark. No online adaptation is allowed at test time. During evaluation, the agent only reuses the distilled external artifacts produced during the exploration phase.

\paragraph{Task metrics.}
We report the official or task-natural metrics for each task family:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Forecasting}: MAE, RMSE, and MSE.
    \item \textbf{Trend classification}: accuracy.
    \item \textbf{Indicator prediction}: MAE over target indicator values or sequences.
    \item \textbf{Correlation reasoning}: accuracy on the canonical correlation label set.
    \item \textbf{MCQA}: accuracy.
\end{itemize}
For overall comparison, we report a macro-average across task families, together with per-task breakdowns.

\paragraph{Baselines.}
We compare against the following representative baselines:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Direct-LLM}: the same backbone model prompted with the official MTBench prompt and no tool-augmented agent pipeline.
    \item \textbf{LLM-Reasoning}: a reasoning-only baseline that retains structured answer generation but removes TimeClaw's exploration-and-distillation loop.
    \item \textbf{TimeART}: the strongest directly comparable prior time-series agent under the same MTBench task family and evaluation setting.
    \item \textbf{TimeClaw w/o learning}: the TimeClaw runtime with no distilled artifacts injected at inference time.
    \item \textbf{TimeClaw}: the full method.
\end{itemize}

\paragraph{Implementation notes.}
All methods use the same base LLM backbone whenever possible. We keep the official MTBench prompts and parsing contracts fixed across methods, so the comparison focuses on learning protocol and agent design rather than prompt engineering. For TimeClaw, the only train-time capabilities unavailable at test time are the exploration-only orchestration tools such as \texttt{spawn\_subagent} and \texttt{evaluate\_*}.

\subsection{Main Results}

Table~\ref{tab:main-mtbench} will report the overall MTBench comparison. The main claim to validate is that TimeClaw improves both aggregate benchmark performance and metric-sensitive task quality relative to strong tool-augmented baselines.

\begin{table*}[t]
\centering
\small
\caption{Main comparison on the TimeART-aligned MTBench subset. Lower is better for forecasting and indicator errors; higher is better for accuracy metrics. Results are placeholders and should be filled after final runs.}
\label{tab:main-mtbench}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Forecast MAE}$\downarrow$ & \textbf{Forecast RMSE}$\downarrow$ & \textbf{Trend Acc.}$\uparrow$ & \textbf{Indicator Error}$\downarrow$ & \textbf{QA/Correlation Acc.}$\uparrow$ & \textbf{Macro Avg.}$\uparrow$ \\
\midrule
Direct-LLM & XX & XX & XX & XX & XX & XX \\
LLM-Reasoning & XX & XX & XX & XX & XX & XX \\
TimeART & XX & XX & XX & XX & XX & XX \\
TimeClaw w/o learning & XX & XX & XX & XX & XX & XX \\
\textbf{TimeClaw} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:task-breakdown} will provide a finer task breakdown. This table is important because TimeClaw's motivation is strongest on metric-evaluable tasks, where path quality matters even when multiple candidate paths are task-valid.

\begin{table*}[t]
\centering
\small
\caption{Per-task breakdown on MTBench. The table should separate weather and finance forecasting, trend classification, indicator prediction, and finance QA/correlation tasks.}
\label{tab:task-breakdown}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Weather Forecast} & \textbf{Finance Forecast} & \textbf{Weather Trend} & \textbf{Finance Trend} & \textbf{Finance Indicator} & \textbf{Finance QA/Corr.} \\
\midrule
Direct-LLM & XX & XX & XX & XX & XX & XX \\
TimeART & XX & XX & XX & XX & XX & XX \\
\textbf{TimeClaw} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} & \textbf{XX} \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Ablation Studies}

We use ablations to isolate the contribution of each core methodological claim.

\paragraph{Ablation A: completion-only vs. path-quality supervision.}
This ablation removes candidate-path comparison and trains the agent with single-path or completion-only supervision. Its purpose is to test whether TimeClaw's gains come specifically from metric-supervised path selection rather than simply from additional execution traces.

\paragraph{Ablation B: without scope-aware tool dropout.}
This ablation keeps the exploration pipeline but disables scope-aware tool dropout. It tests whether preventing tool-prior collapse is necessary for preserving branch diversity and improving downstream learning quality.

\paragraph{Ablation C: without artifact reinjection.}
This ablation runs the same trained system but removes learned artifact injection at inference time. It isolates whether the gains truly come from reusable external learning rather than from train-time exploration alone.

\paragraph{Ablation D: artifact-type breakdown.}
We further remove one artifact family at a time, including \texttt{MEMORY}, tool notes, and \texttt{SKILLS}. This reveals which external artifact layer contributes most to different MTBench task types.

\begin{table}[t]
\centering
\small
\caption{Core ablations for TimeClaw.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Macro Avg.}$\uparrow$ & \textbf{Forecast MAE}$\downarrow$ & \textbf{Trend Acc.}$\uparrow$ & \textbf{QA/Corr. Acc.}$\uparrow$ \\
\midrule
Full TimeClaw & XX & XX & XX & XX \\
\quad w/o path comparison & XX & XX & XX & XX \\
\quad w/o tool dropout & XX & XX & XX & XX \\
\quad w/o artifact reinjection & XX & XX & XX & XX \\
\quad w/o MEMORY & XX & XX & XX & XX \\
\quad w/o tool notes & XX & XX & XX & XX \\
\quad w/o SKILLS & XX & XX & XX & XX \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Analysis of Learning Dynamics}

Beyond end-task metrics, we analyze whether TimeClaw learns in the way the methodology claims.

\paragraph{Path-quality gap.}
We measure the metric gap between the winning candidate path and alternative task-valid candidates. If these gaps are consistently non-trivial, they provide direct evidence that completion-only supervision is too weak for numeric tasks.

\paragraph{Tool diversity and collapse.}
We track scope-level tool usage distributions and report entropy before and after applying scope-aware tool dropout. If dropout works as intended, dominant-tool concentration should decrease and candidate-path diversity should increase.

\paragraph{Artifact utilization.}
We analyze retrieval frequency and downstream usefulness of different artifact layers. In particular, we test whether forecast tasks rely more heavily on scope-local rules, while QA or correlation tasks benefit more from operational tool guidance.

\paragraph{Case studies.}
We include a small number of MTBench case studies that show: (1) two task-valid paths with different quantitative quality, (2) a failure mode caused by dominant-tool overuse, and (3) a successful correction after artifact reinjection.

\subsection{Discussion of Comparison Scope}

Our main experimental goal is not merely to show that TimeClaw is another strong solver on a benchmark subset. Rather, it is to test whether a time-series agent can \emph{learn from exploratory execution} in a way that transfers to benchmark evaluation. For this reason, the experiments should be interpreted along two axes simultaneously: head-to-head competitiveness with TimeART on the same MTBench subset, and internal validation that TimeClaw's external learning loop is responsible for the gains.
```

---

## Suggested Bibkey Mapping

后续你们可以按自己的 `.bib` 命名统一替换。当前这版正文里用到的 bibkey 建议如下：

```text
chen2025mtbench            = MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering
weng2026temporalbench      = TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks
wu2026timeart              = TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation
liu2025tsagent             = TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering
zhao2025timeseriesscientist= TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis
garza2025timecopilot       = TimeCopilot
yeh2025llmagentsforecasting= Empowering Time Series Forecasting with LLM-Agents
trirat2026tsdebate         = TS-Debate: Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning
ruan2026mas4ts             = Visual Reasoning over Time Series via Multi-Agent System
castrone2026castr1         = Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting
schick2023toolformer       = Toolformer: Language Models Can Teach Themselves to Use Tools
patil2023gorilla           = Gorilla: Large Language Model Connected with Massive APIs
qin2023toolllm             = ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
memskill2026               = MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents
asda2026                   = ASDA: Automated Skill Distillation and Adaptation
skillsd2026                = Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents
openclawrl2026             = OpenClaw-RL: Train Any Agent Simply by Talking
cuaskill2026               = CUA-Skill: Develop Skills for Computer Using Agent
clawkeeper2026             = ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers
clawsafety2026             = ClawSafety: "Safe" LLMs, Unsafe Agents
safeclawr2026              = SafeClaw-R: Towards Safe and Secure Multi-Agent Personal Assistants
```

---

## Notes

- 这一版 `Experiments` 没有虚构结果，所有数值位置都保留为 `XX`。
- 现在这版已经统一为：`所有报告实验 = TimeART 对齐的 MTBench 子集`。
- 这份文档目前是“正文可贴版”，不是 appendix；如果你后面要，我可以继续把 `Prompt Assembly`、`Artifact Rendering`、`Workflow Figure` 也整理成一份独立 appendix LaTeX 稿。
