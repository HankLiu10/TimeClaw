# Tool Dropout 最终方案报告

**版本**：定稿（§3.2 为最终采用的概率形式）  
**目的**：以训练期随机「隐藏高频工具」替代独立 Warmup 阶段，在**不增加额外 API 阶段**的前提下提升工具探索多样性，并与 per-scope 记忆蒸馏对齐。  
**核心公式**：\(P_{\text{keep}}(i)=\bigl((1+n_{\min})/(1+n_i)\bigr)^{\alpha}\)，\(n_{\min}=\min_j n_j\)；均匀探索后 \(n_{\min}\approx n_i\) 时 dropout 自然减弱。  
**状态**：若仓库中实现被回滚，应以本文为恢复规格说明。

---

## 1. 背景与目标

### 1.1 问题

- Agent 在训练期容易反复选择少数「顺手」工具（如 `ets_forecast`、`frequency_forecast`），长尾工具（尤其多类时序大模型）长期零调用。
- 独立 **Warmup**（每工具强制 K 次）在工具数量 × scope 数量较大时 API 与时间成本过高。

### 1.2 目标

1. **调用次数越多的工具，越有可能从当前 subagent 的可见工具列表中被隐藏**（dropout），从而鼓励尝试低频工具。
2. **每个 subagent 独立采样** mask，使并行分支自然看到不同工具子集（多样性优于仅靠 prompt 要求「路径不同」）。
3. **计数必须 per-scope**（如 `weather_forecast` 与 `finance_forecast` 独立），与 `ToolsStore` / 蒸馏使用的 scope 一致。
4. **测试阶段不启用**：仅 `train` 模式、且在 `spawn_subagent` 派生子执行器时生效。

---

## 2. 超参与永不 Drop 集合

| 符号 / 常量 | 说明 |
|-------------|------|
| \(\alpha > 0\) | 幂指数；越大则「相对最冷门工具」与「相对热门工具」的保留率差距越明显 |
| `_NEVER_DROP` | 见下；**不参与**下文的 \(n_{\min}\)、\(n_i\) 比较时应对其强制 \(P_{\text{keep}}=1\) |

**`_NEVER_DROP`（必须始终对子 agent 可见）**：

- `evaluate_against_gt`
- `evaluate_batch_against_gt`（若注册名含 `batch_evaluate_against_gt` 等，以实际注册名为准）
- `spawn_subagent`
- **`llm_reasoning`**（训练流程写死必须可走通）

对 \(t \in \text{\_NEVER\_DROP}\)：不应用随机 mask，等价于 **\(P_{\text{keep}}(t)=1\)**（或从不加入待采样列表）。

---

## 3. Dropout 概率：相对最小计数的幂律保留率（最终实现）

本节为**最终实现采用的公式**（替代讨论阶段曾尝试的 Sigmoid+z-score，见附录 A）。

### 3.1 记号

- 在当前 scope、当前 `candidates` 下，仅对**参与 dropout 竞争**的工具建索引 \(i\)（与实现一致：一般为模态兼容且不在 `_NEVER_DROP` 中的工具）。
- \(n_i \ge 0\)：该工具在本 scope 下的累计调用次数。
- \[
  n_{\min} = \min_j n_j
  \]
  其中 \(j\) 与 \(i\) 遍历**同一套**参与统计的工具集合（保证大家用同一标尺）。

### 3.2 保留率与 dropout 率

定义 **保留**（可见）概率：

\[
P_{\text{keep}}(i) = \left(\frac{1 + n_{\min}}{1 + n_i}\right)^{\alpha}.
\]

单次 mask 中：以概率 \(P_{\text{keep}}(i)\) **保留**工具 \(i\)，以概率 \(1 - P_{\text{keep}}(i)\) **隐藏**（dropout）工具 \(i\)（可与 Bernoulli 采样等价实现）。

### 3.3 性质（设计动机）

1. **最冷门的一批工具**：\(n_i = n_{\min}\) 时，
   \[
   P_{\text{keep}}(i) = \left(\frac{1+n_{\min}}{1+n_{\min}}\right)^{\alpha} = 1,
   \]
   即**当前最不被偏爱的工具永远不会被本公式主动 mask 掉**，与「先保证可见性、再谈探索」一致。

2. **调用越多，越容易被 drop**：\(n_i > n_{\min}\) 时 \((1+n_{\min})/(1+n_i) < 1\)，故 \(P_{\text{keep}}(i) < 1\)；\(n_i\) 越大，\(P_{\text{keep}}\) 越小。

3. **均匀探索后自动减弱正则**：当各工具计数拉平、\(n_{\min} \approx n_i\) 时，比值 \(\approx 1\)，故 \(P_{\text{keep}}(i) \approx 1\)，**dropout 强度随训练自发减弱**，无需手工切换阶段。

4. **全零计数**：若所有参与统计的 \(n_i=0\)，则 \(n_{\min}=0\)，对每个 \(i\) 有 \(P_{\text{keep}}(i) = (1/1)^\alpha = 1\)，训练初期**不 dropout**，与「冷启动自由探索」一致。

5. **+1 平滑**：避免 \(n_i=0\) 与分母为 0 的数值问题，并在极少次调用时使曲线平滑。

### 3.4 与采样、保底的关系

- §4 中「以概率保留」的 Bernoulli 步骤，其 **保留概率即为本节的 \(P_{\text{keep}}(i)\)**（对 `_NEVER_DROP` 固定为 1）。
- 保底 A（sub-0 全量）与保底 B（按类别补回）仍在该概率采样**之后**或与之配合，以硬约束防止无解。

---

## 4. 采样规则与保底机制

### 4.1 每个 subagent 独立采样

对 subagent \(k\) 及工具 \(t \in \mathcal{C}\)（`_NEVER_DROP` 强制保留，\(P_{\text{keep}}=1\)）：

- 以概率 \(P_{\text{keep}}(t)\)（§3.2）**保留** \(t\)，以概率 \(1 - P_{\text{keep}}(t)\) **移除** \(t\)。
- 等价实现：若 `u ~ Uniform(0,1)` 且 `u < P_keep(t)` 则保留（或 `u >= P_drop` 则保留，二者一致即可）。

各 subagent 使用**独立**随机数流，使并行路径工具组合不同。

### 4.2 保底 A：首个 subagent 不 dropout

- **Sub-agent 0**：`protect_first=True`，`visible = candidates`（完整列表）。
- **Sub-agent 1, 2, …**：应用上述采样 + 保底 B。

保证每次 spawn 至少有一条分支不受 mask 影响，降低「全体走错工具集」对 Daily Notes 质量的冲击。

### 4.3 保底 B：按工具类别至少保留一个

- 提供 `category_map: tool_name -> category`（来自 `ClawTool.category.value` 等）。
- 采样后若某类别在 `visible` 中**没有任何**工具，则从该类的 `candidates` 中 **随机恢复一个**。

避免某一大类（如 prediction）被整体清空导致任务不可解。

---

## 5. 与流水线的衔接

### 5.1 注入点

1. **`ExecutorAgent._spawn_subagents`**（仅 `train`）  
   - 读取当前 `plan` 中的 `scope_counts`、`category_map`（及当前 scope 键）。  
   - 为每个子任务调用 `generate_dropout_candidates`（或等价逻辑），得到 `override_candidates`。  
   - **`_run_sub_executor(..., override_candidates=...)`**。

2. **`PipelineRunner.run()`（训练模式）**  
   - 将当前 scope 的 `scope_counts` 与全局/静态 `category_map` 写入 `envelope.task.metadata`，供 `perceive` → `plan` 传入 executor。

### 5.2 计数更新（定稿）

- `scope_counts` 必须**随训练实时更新**（例如每次 run 结束后根据 `execution_log` 递增），**不能**仅依赖「每 N 条 note 才蒸馏一次」时才刷新，否则小样本训练全程 dropout 不生效。
- 推荐在 `ToolsStore` 上提供 `increment_scope_usage(scope, tool_names)`（或等价 API），由 `runner` 在持久化 learning outcome 后调用。

### 5.3 与 Warmup 的关系

- **最终方案**：用 Tool Dropout **替代**独立 Warmup 阶段；删除 `--warmup-k`、`_run_warmup` 等与强制 K 次探索绑定的逻辑。
- Dropout 超参初期可硬编码；后续可迁入 `TimeClawConfig`。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 某样本最优工具被 mask，轨迹变差 | 保底 A（sub-0 全量）+ 保底 B（类内至少一个） |
| 幂律下 \(\alpha\) 过大导致高频工具几乎总被 hide | 调小 \(\alpha\)；或 cap \(P_{\text{keep}}\) 下限（若需保留探索噪声） |
| `_NEVER_DROP` 工具被误算进 \(n_{\min}\) 导致标尺偏移 | 实现上 \(n_{\min},n_i\) 仅对「可参与 dropout 的工具集」计算；基础设施工具单独强制保留 |
| 不保证有限样本内每个工具都被调用 | 接受为概率探索；可调 \(\alpha\) 或训练量；均匀后 \(P_{\text{keep}}\to 1\) 会自然放松 |

---

## 7. 实现清单（恢复代码时自检）

- [ ] 新建或恢复 `timeclaw/agents/tool_dropout.py`：按 §3.2 计算 \(P_{\text{keep}}(i)\)，再 `apply_dropout` / `generate_dropout_candidates`。
- [ ] `executor.py`：`train` + spawn 路径注入 per-subagent `override_candidates`。
- [ ] `runner.py`：训练 metadata 注入 `scope_counts`、`category_map`；每步更新 scope 工具计数。
- [ ] 确认测试路径不传入 dropout 相关字段或显式关闭。
- [ ] 单元测试：全零计数（全体 \(P_{\text{keep}}=1\)）、\(n_i=n_{\min}\)、仅少数工具 \(n_i\gg n_{\min}\)、类别保底、`_NEVER_DROP`。

---

## 8. 文档与溯源

- **概率形式以 §3.2 幂律为准**（用户确认的最终实现）；附录 A 为早期讨论中的 Sigmoid 备选，**非**当前定稿公式。
- 设计讨论与实现迭代记录见 Cursor 会话 transcript（例如 uuid `4f3aa96a-84eb-4558-8206-166452f584c6`）；plan 文档若与 §3 冲突，以本文 §3 为准。

---

## 附录 A：讨论阶段备选（Sigmoid + z-score，非最终实现）

曾考虑用 per-scope 计数的标准化分数控制 dropout：

\[
z_i = \frac{n_i - \bar n}{\sigma_n},\quad
p_{\text{drop}}(i) = P_{\max}\cdot \sigma(\text{STEEPNESS}\cdot z_i),
\]

并配合「\(n_i=0\) 不 drop」「\(\bar n,\sigma_n\) 仅对 \(n_i>0\) 计算」等修补。该路线在实现与调试中暴露出对「大量零计数工具」标尺敏感等问题，**最终被 §3.2 的 \((1+n_{\min})/(1+n_i)\) 幂律替代**，因其直接对齐「相对最冷工具」且均匀探索后自动减弱。

---

*报告结束。*
