# TimeClaw Sub-Agent 机制技术报告

> 版本: v1.0 | 日期: 2026-03-13

---

## 1. 背景与动机

### 1.1 问题

TimeClaw 当前采用单 Executor Agent（mini-ReAct 循环）处理所有任务。虽然单 agent 在大多数场景下足够高效，但某些复杂场景存在局限：

- **多变量预测**：未来涉及多个变量的联合预测时，不同变量可能需要独立的分析和预测流程
- **异构子任务**：一个复杂任务可能自然分解为多个独立子问题（如同时分析趋势和季节性并产出独立结论）
- **并行加速**：独立子任务可以并行执行，减少整体耗时

### 1.2 设计目标

1. **主 agent 自主决策**：是否 spawn sub-agent 由主 agent 根据任务复杂度自行判断
2. **完整经验继承**：sub-agent 继承全部层次化记忆（SOUL + PRINCIPLES + MEMORY + TOOLS + PLAYBOOKS），确保精度不降
3. **向后兼容**：不影响现有单 agent 路径，spawn 机制为可选能力
4. **防递归**：sub-agent 不能再次 spawn sub-agent

### 1.3 与 OpenClaw 的关键差异

| 维度 | OpenClaw | TimeClaw |
|------|----------|----------|
| sub-agent 上下文 | 精简提示（仅任务描述） | 完整经验注入（SOUL + PRINCIPLES + MEMORY + tool_notes） |
| 适用场景 | 通用任务（文件操作、代码等） | 精度敏感的数值预测任务 |
| 启用策略 | 框架自动拆分 | 主 agent 自主判断 |

---

## 2. 架构设计

### 2.1 整体流程

```
主 Agent (ExecutorAgent, ReAct 循环)
|
+-- Step 1: 分析任务 -> 调用普通工具 (basic_stats, detect_trend, ...)
+-- Step 2: 继续分析 -> 调用更多工具
+-- Step N: 决定需要并行处理 -> 调用 spawn_subagent
|   |
|   +-- Sub-Agent 0 (独立 ReAct 循环, 最多 5 步)
|   |   +-- 继承: system_memory (SOUL+PRINCIPLES+MEMORY)
|   |   +-- 继承: data_context (原始数据)
|   |   +-- 继承: tool_guide (全部工具)
|   |   +-- 继承: l1_section + skills_section (Daily Notes + Playbooks)
|   |   +-- 返回: {answer, answer_type, confidence, reasoning}
|   |
|   +-- Sub-Agent 1 (独立 ReAct 循环)
|   |   +-- ... (同上)
|   |
|   +-- Sub-Agent 2-3 (最多 4 个)
|       +-- ...
|
+-- Step N+1: 收到所有 sub-agent 结果 -> 综合推理
+-- Step N+2: done=true, 输出最终 answer
```

### 2.2 核心常量

```python
MAX_REACT_STEPS = 10           # 主 agent 最大步数
MAX_SUB_REACT_STEPS = 5        # sub-agent 最大步数
MAX_CONCURRENT_SUBAGENTS = 4   # 最大并发 sub-agent 数
SUBAGENT_TIMEOUT_SECONDS = 120 # 超时时间（预留）
```

### 2.3 Prompt 设计

#### 主 Agent Prompt 扩展

在 REACT_SYSTEM_PROMPT 中新增 spawn_subagent 工具的 JSON 输出格式：

```json
{
  "thought": "why this task benefits from parallel sub-agents",
  "done": false,
  "tool": "spawn_subagent",
  "args": {
    "tasks": [
      {"task": "sub-task description 1", "tools_hint": "optional tool names"},
      {"task": "sub-task description 2", "tools_hint": "optional tool names"}
    ]
  }
}
```

#### Sub-Agent 专用 Prompt

新增 SUBAGENT_SYSTEM_PROMPT：
- 明确身份："你是 TimeClaw 的 sub-agent，负责特定子任务"
- 约束范围："只关注分配的子任务，不要尝试解决整体问题"
- 步数限制：<= 5 步
- 输出格式与主 agent 一致

新增 SUBAGENT_FIRST_TURN：
- 子任务描述
- 数据指纹
- 记忆部分（Daily Notes + Playbooks）
- 完整工具列表

---

## 3. 实现细节

### 3.1 文件变更

| 文件 | 变更内容 |
|------|----------|
| `timeclaw/agents/prompts/executor_prompt.py` | 新增 spawn_subagent JSON 格式说明、SUBAGENT_SYSTEM_PROMPT、SUBAGENT_FIRST_TURN |
| `timeclaw/agents/executor.py` | 新增 _spawn_subagents()、_run_sub_executor() 方法；在 act() 中增加 spawn_subagent 路由 |

### 3.2 _spawn_subagents() 方法

```
输入: tool_args (包含 tasks 列表), plan, envelope
流程:
  1. 取 tasks 列表，截断到 MAX_CONCURRENT_SUBAGENTS
  2. 为每个 task 创建协程 _run_sub_executor()
  3. asyncio.gather() 并发执行
  4. 收集结果，构建 observation 字符串返回给主 agent
输出: {n_tasks, n_success, results, observation}
```

### 3.3 _run_sub_executor() 方法

```
输入: idx, task_description, tools_hint, plan, envelope
流程:
  1. 构建 system_prompt = SUBAGENT_SYSTEM_PROMPT + system_memory (SOUL+PRINCIPLES+MEMORY)
  2. 构建 first_user = SUBAGENT_FIRST_TURN + data_context
  3. 初始化 conversation 列表
  4. 进入 mini-ReAct 循环 (最多 MAX_SUB_REACT_STEPS 步):
     a. LLM 推理
     b. 如果 done -> 返回结果
     c. 如果 tool=spawn_subagent -> 拒绝（防递归）
     d. 如果 tool 不在候选列表 -> 报错
     e. 执行工具 -> 结果写入 conversation
  5. 返回 {answer, answer_type, confidence, reasoning}
```

### 3.4 act() 中的路由

在主 agent 的 ReAct 循环中，当 tool_name == "spawn_subagent" 时：
1. 调用 _spawn_subagents()
2. 将观察结果（所有 sub-agent 的结果摘要）写入 conversation
3. 记录到 execution_log
4. continue 让主 agent 继续下一步（综合 sub-agent 结果）

### 3.5 记忆注入路径

| 记忆组件 | 注入方式 | 注入位置 |
|----------|----------|----------|
| SOUL.md | system_memory -> system_prompt | Sub-agent system prompt |
| PRINCIPLES.md | system_memory -> system_prompt | Sub-agent system prompt |
| MEMORY.md (L2) | system_memory -> system_prompt | Sub-agent system prompt |
| Daily Notes (L1) | memory_section -> first_user | Sub-agent first turn message |
| PLAYBOOKS | memory_section -> first_user | Sub-agent first turn message |
| TOOLS.md | tool_guide (experience_insights) | Sub-agent first turn message |

---

## 4. 安全机制

### 4.1 防递归
Sub-agent 如果尝试调用 spawn_subagent，会收到拒绝消息："Sub-agents cannot spawn further sub-agents."

### 4.2 并发限制
tasks 列表被截断到 MAX_CONCURRENT_SUBAGENTS = 4。

### 4.3 步数限制
Sub-agent 最多执行 MAX_SUB_REACT_STEPS = 5 步（主 agent 为 10 步）。

### 4.4 异常隔离
使用 asyncio.gather(*coros, return_exceptions=True) 确保单个 sub-agent 的失败不影响其他 sub-agent。

---

## 5. 使用场景示例

### 5.1 多变量预测（未来）

```
主 Agent:
  Step 1: basic_stats -> 发现数据有温度和湿度两个维度
  Step 2: spawn_subagent
    - Sub-Agent 0: "预测未来 24 小时温度变化" -> 独立 ReAct -> 温度预测
    - Sub-Agent 1: "预测未来 24 小时湿度变化" -> 独立 ReAct -> 湿度预测
  Step 3: 综合两个预测结果 -> 输出联合预测
```

### 5.2 多模态分析

```
主 Agent:
  Step 1: 分析指纹 -> 发现同时有时序数据和文本新闻
  Step 2: spawn_subagent
    - Sub-Agent 0: "分析时序数据趋势和关键指标"
    - Sub-Agent 1: "分析新闻文本的情感和事件"
  Step 3: 综合时序分析和文本分析 -> 输出最终判断
```

### 5.3 竞争性模型选择

```
主 Agent:
  Step 1: detect_seasonality -> 发现周期性
  Step 2: spawn_subagent
    - Sub-Agent 0: "使用 prophet_forecast 预测"
    - Sub-Agent 1: "使用 chronos_forecast 预测"
  Step 3: 比较两个预测结果 -> 选择更优的
```

---

## 6. 未来工程化方向

### 6.1 超时机制
当前 SUBAGENT_TIMEOUT_SECONDS = 120 已定义但未启用。未来可用 asyncio.wait_for() 包装每个 sub-agent 协程。

### 6.2 资源隔离
当前 sub-agent 与主 agent 共享 LLM 连接和 MCP manager。高并发场景下可能需要独立的 LLM 客户端实例。

### 6.3 sub-agent 间通信
当前 sub-agent 之间完全隔离。如果未来需要协作（如一个 sub-agent 的输出作为另一个的输入），可以引入依赖关系图。

### 6.4 经验记录
当前 sub-agent 的执行日志不单独记录到 experience_pool。未来可以将 sub-agent 的工具链也纳入学习。

### 6.5 动态 sub-agent 数量
可以根据任务复杂度动态调整 MAX_SUB_REACT_STEPS 和允许的 sub-agent 数量。

### 6.6 差异化 sub-agent
未来可支持不同 sub-agent 使用不同模型（如一个用 GPT-5.1，另一个用 Claude）。

### 6.7 结果投票/集成
对于预测类 sub-agent，可以引入 ensemble 机制（加权平均、投票等）而非由主 agent 手动综合。

---

## 7. 测试验证

### 7.1 导入验证

```
Constants: MAX_REACT_STEPS=10, MAX_SUB_REACT_STEPS=5, 
           MAX_CONCURRENT_SUBAGENTS=4, SUBAGENT_TIMEOUT_SECONDS=120
Methods: _spawn_subagents OK, _run_sub_executor OK
```

### 7.2 Prompt 验证

```
SUBAGENT_SYSTEM_PROMPT length: 845 chars
SUBAGENT_FIRST_TURN length: 143 chars
"spawn_subagent" in REACT_SYSTEM_PROMPT: True
```

### 7.3 Lint 验证
executor.py 和 executor_prompt.py 无 linter 错误。

---

## 附录 A: 相关文件清单

| 文件路径 | 说明 |
|----------|------|
| `timeclaw/agents/executor.py` | 主 executor，包含 spawn 逻辑 |
| `timeclaw/agents/prompts/executor_prompt.py` | 所有 prompt 模板 |
| `timeclaw/memory/SOUL.md` | 固定的 agent 身份定义 |
| `timeclaw/memory/hierarchical.py` | 层次化记忆管理器 |
| `timeclaw/pipeline/runner.py` | 管线编排器 |
