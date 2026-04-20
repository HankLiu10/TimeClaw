# TimeClaw 流程讲解版

更新时间：2026-04-16

这份文档的目标不是展示全部原始 trace，而是用**一条可读性极高的代表性样本**，讲清楚：

- 主 agent 在做什么
- 子 agent 在做什么
- 为什么要分支探索
- 为什么要比较路径
- `notes / memory / skills` 分别是什么
- 这些产物是怎么一步一步得到的

为了便于理解，我会：

- 只保留最关键字段
- 用中文解释每一步在做什么
- 用一个统一例子贯穿全流程

---

## 1. 一句话先讲清楚 TimeClaw 在做什么

TimeClaw 不是让 agent “只做一道题”，而是让 agent：

1. 对同一道题探索多条候选路径
2. 比较哪条路径更优
3. 把这次学到的经验沉淀成外部知识
4. 以后再遇到类似题时复用这些知识

所以它的主线是：

**做题 -> 比较 -> 学习 -> 复用**

---

## 2. 我们用什么例子来讲

这里用一个最典型的例子：

- 任务：金融短期预测 `forecast`
- 输入：最近 48 个时间点的价格序列
- 输出：未来 6 个时间点的预测数组

这个例子很适合说明 TimeClaw 的核心思想，因为：

- 不止一条路径能给出合法答案
- 但不同路径的误差会差很多
- 所以关键不是“能不能做出来”，而是“哪条路径更好”

---

## 3. 角色分工：主 Agent 和子 Agent 到底分别干什么

### 3.1 主 Agent

主 agent 负责：

- 理解当前任务
- 决定要不要分支探索
- 指派多个子 agent 去走不同路径
- 收集所有候选答案
- 调用评估工具比较这些候选
- 最后输出一条 `learning_summary`

可以把主 agent 理解成：

**总指挥 + 比较者 + 经验总结者**

### 3.2 子 Agent

子 agent 负责：

- 沿着主 agent 指派的某一条路径去真正做题
- 调用真实工具得到结果
- 输出一个 task-valid candidate

可以把子 agent 理解成：

**具体执行某一条候选方案的小执行者**

### 3.3 为什么要分成主 agent 和子 agent

因为如果只有一个 agent，它很容易：

- 只走一条最熟悉的路径
- 做完就结束
- 没法比较“其实还有没有更优的路径”

而 TimeClaw 的学习恰恰来自：

- 多条路径的并行探索
- 再基于指标做比较

---

## 4. 整个流程总览

先看一遍最简流程：

```text
输入任务
  ↓
主 agent 判断：需要探索多条路径
  ↓
spawn 多个子 agent
  ↓
每个子 agent 各走一条路径并产出合法答案
  ↓
主 agent 收集所有候选答案
  ↓
evaluate_* 比较路径质量
  ↓
主 agent 输出 learning_summary
  ↓
系统把这次经验写成 notes
  ↓
notes 蒸馏成 memory / tools / skills
  ↓
以后推理时把这些 artifact 再注入 prompt
```

---

## 5. 第一步：主 Agent 先决定要探索哪些路径

在这个例子里，主 agent 看完样本后，决定探索 3 条 forecasting 路径：

- `chronos2_forecast`
- `ets_forecast`
- `arima_forecast`

它不会自己把三条都跑完，而是通过 `spawn_subagent` 把任务分给三个子 agent。

下面是一个简化后的主 agent 决策结果。

```json
{
  "tool": "spawn_subagent",
  "args": {
    "tasks": [
      {
        "slot_id": 0,
        "task": "用 Chronos-2 做未来 6 步预测",
        "tools_hint": ["chronos2_forecast"],
        "branch_metadata": {"branch_name": "子agent1"}
      },
      {
        "slot_id": 1,
        "task": "用 ETS 做未来 6 步预测",
        "tools_hint": ["ets_forecast"],
        "branch_metadata": {"branch_name": "子agent2"}
      },
      {
        "slot_id": 2,
        "task": "用 ARIMA 做未来 6 步预测",
        "tools_hint": ["arima_forecast"],
        "branch_metadata": {"branch_name": "子agent3"}
      }
    ]
  }
}
```

### 这一步的本质是什么

主 agent 在做的不是“直接预测”，而是：

**把一个任务拆成多个可比较的候选执行路径**

---

## 6. 第二步：子 Agent 真正去走各自的路径

每个子 agent 会拿到：

- 同一个原始任务
- 同一份样本数据
- 当前 scope 对应的 memory / skills / tool notes
- 一条明确的路径提示，比如“优先用 ets_forecast”

但子 agent 有两个重要限制：

1. 它不能再 `spawn_subagent`
2. 它不能调用 `evaluate_*`

也就是说，子 agent 不是来学习和比较的，它只是：

**认真把自己这一条路径做出来**

---

## 7. 子 Agent 的结果长什么样

我们只看最关键的返回结果。

### 子agent1：Chronos-2 路径

```json
{
  "candidate_name": "chronos2_forecast",
  "tool_chain": ["chronos2_forecast"],
  "answer": [102.4, 103.1, 103.7, 104.2, 104.6, 105.0],
  "answer_type": "forecast"
}
```

### 子agent2：ETS 路径

```json
{
  "candidate_name": "ets_forecast",
  "tool_chain": ["ets_forecast"],
  "answer": [101.8, 102.0, 102.3, 102.5, 102.8, 103.0],
  "answer_type": "forecast"
}
```

### 子agent3：ARIMA 路径

```json
{
  "candidate_name": "arima_forecast",
  "tool_chain": ["arima_forecast"],
  "answer": [101.6, 101.7, 101.8, 101.8, 101.9, 101.9],
  "answer_type": "forecast"
}
```

### 这一阶段最关键的点

三条路径都输出了合法 forecast 数组。  
也就是说，它们都是 **task-valid candidates**。

这时候问题就变成：

**既然都合法，那到底哪条更值得学？**

---

## 8. 第三步：主 Agent 比较这些候选路径

这一步是 TimeClaw 非常核心的地方。

主 agent 不会只看“哪条路径完成了任务”，因为三条都完成了。  
它会调用 `evaluate_batch_against_gt`，拿真实指标去比较。

简化后的评估结果如下：

```json
{
  "results": [
    {
      "candidate_index": 0,
      "metrics": {"mae": 2.410, "rmse": 2.882}
    },
    {
      "candidate_index": 1,
      "metrics": {"mae": 1.120, "rmse": 1.392}
    },
    {
      "candidate_index": 2,
      "metrics": {"mae": 2.980, "rmse": 3.344}
    }
  ],
  "winner_index": 1
}
```

### 这一步说明了什么

- `chronos2_forecast`：能做出来，但误差偏大
- `ets_forecast`：也能做出来，而且误差最小
- `arima_forecast`：也合法，但偏平，误差最大

所以真正值得学习的不是：

“forecast 任务可以完成”

而是：

**在这种样本条件下，ETS 这条路径比另外两条更优**

---

## 9. 第四步：主 Agent 输出学习总结

当主 agent 已经：

- 看到了多条候选路径
- 做了真实指标比较
- 知道谁赢了

它才可以输出最后的 `learning_summary`。

下面只看最关键部分：

```json
{
  "answer_type": "learning_summary",
  "answer": {
    "insight": "ets_forecast 路径胜出，因为它保留了局部平稳后缓升结构，在 MAE 和 RMSE 上都优于更激进或更平坦的路径。",
    "recommendation": "当短期序列局部平稳、仅有轻微上行且没有明显尖峰时，优先尝试 ets_forecast 这类保守平滑路径。"
  }
}
```

### 这一步的本质是什么

主 agent 最终输出的不是“题目的答案”，而是：

- 哪条路径赢了
- 为什么赢
- 下次遇到类似样本时该怎么做

这正是 TimeClaw 的学习信号来源。

---

## 10. 第五步：这次经验先变成 Notes

`Notes` 可以理解为：

**一次训练样本的原始经验记录**

它保留的是：

- 当前任务是什么
- 哪条路径赢了
- 候选路径有哪些
- 指标怎么样
- 当前样本是什么特征

一个简化后的 note 长这样：

```md
### [finance_short_horizon_forecast] SCORED

Scope:
finance_short_horizon_forecast

Winner Tools:
- ets_forecast

Prediction:
[101.8, 102.0, 102.3, 102.5, 102.8, 103.0]

Metrics:
- mae: 1.120
- rmse: 1.392

Fingerprint:
{"recent_trend":"flat_to_slight_up","volatility":"medium","spike":"none"}

Approaches:
- ets_forecast(MAE=1.120, RMSE=1.392)
- chronos2_forecast(MAE=2.410, RMSE=2.882)
- arima_forecast(MAE=2.980, RMSE=3.344)

Analysis:
ETS 路径在当前样本上胜出。Chronos-2 过冲，ARIMA 偏平，ETS 最贴近局部结构。
```

### Notes 的定位

它不是最终给模型直接看的主 memory。  
它更像：

**原始学习证据**

后面还要继续蒸馏。

---

## 11. 第六步：Notes 蒸馏成 MEMORY

`MEMORY` 不是流水账，而是结构化规则。

它会把多条相似的 notes 归纳成：

- 什么时候适用
- 优先用什么工具
- 避开什么工具
- 为什么
- 这条规则有多少证据支持

一个典型的 `MEMORY` 条目是这样：

```json
{
  "scope": "finance_short_horizon_forecast",
  "rule_id": "finance_short_horizon_forecast::007",
  "kind": "tool_selection",
  "summary": "当短期序列局部平稳、仅有轻微上行时，优先使用 ets_forecast。",
  "conditions": [
    "recent_trend=flat_to_slight_up",
    "volatility=low_or_medium",
    "no_large_spike=true"
  ],
  "preferred_tools": ["ets_forecast"],
  "avoid_tools": ["chronos2_forecast"],
  "rationale": "激进路径更容易在短窗口里过冲。",
  "evidence": {
    "support_count": 4
  },
  "confidence": 0.86,
  "injectable": true
}
```

### 这说明 MEMORY 是什么

它不是一句“经验总结”，而是：

**结构化、可筛选、可注入、可累积证据的规则对象**

---

## 12. 为什么 MEMORY 不是“越存越多”就行

TimeClaw 的 memory 不是简单 append-only。

系统还会做这些事：

- 合并近似重复规则
- 给已有规则增加支持证据
- 发现规则冲突
- 把不稳定的规则标成 `injectable=false`

比如下面这条规则：

```json
{
  "rule_id": "finance_short_horizon_forecast::010",
  "summary": "当局部证据不足时，不要只凭 basic_stats 直接决定未来路径。",
  "injectable": false
}
```

它的意思不是这条规则没价值，而是：

**先保留，但现在还不适合直接注入推理 prompt**

这正是 TimeClaw 一个很重要的点：

**存储 和 注入 是分开的**

---

## 13. 第七步：MEMORY 再蒸馏成 SKILLS

`SKILLS` 可以理解为：

**面向后续推理的可执行工作流**

也就是说，`MEMORY` 更像规则库，`SKILLS` 更像 SOP。

一个简化后的 `SKILL` 长这样：

```md
### Skill 1
Condition:
近期窗口整体平稳，仅在尾部出现轻微抬升，没有明显尖峰。

Why:
这类样本更适合保守平滑路径，过激预测更容易把短期尾部抬得过高。

Workflow:
1. 先调用 ets_forecast 生成基线预测。
2. 检查结果是否保持“平稳后缓升”的局部形态。
3. 如果没有明显过冲，则优先采用该结果。

Guardrails:
- 如果尾部突然明显变陡，不要直接接受。
```

### 这说明 SKILLS 是什么

它不是单条规则，而是：

**一组可操作的、面向测试时复用的工作流指导**

---

## 14. 这些东西在推理时怎么用

到了测试/推理阶段，系统不会再继续学习。  
它做的是把已经学到的 artifact 注入给 agent。

也就是说，推理时 agent 会看到：

- 当前任务描述
- 当前样本的 fingerprint / profiling
- 相关的 `MEMORY`
- 相关的 `tool notes`
- 相关的 `SKILLS`

然后在这些外部知识的帮助下完成当前任务。

所以推理阶段的逻辑是：

**不是重新学，而是复用以前学到的东西**

---

## 15. 最终你可以怎么理解这几个产物

### 主 Agent 的结果是什么

主 agent 的核心结果是：

- 多条候选路径
- 路径比较结果
- 最终 `learning_summary`

### 子 Agent 的结果是什么

子 agent 的核心结果是：

- 某一条路径上的真实工具执行结果
- 一个 task-valid candidate answer

### Notes 是什么

`Notes` 是：

**单样本经验记录**

### MEMORY 是什么

`MEMORY` 是：

**从多条 notes 蒸馏出来的结构化规则**

### SKILLS 是什么

`SKILLS` 是：

**从 MEMORY 再整理出来的可执行工作流**

---

## 16. 用一句话收束整套机制

如果你要给别人讲整个流程，我建议最后就用这句话：

**主 agent 负责“组织探索和比较学习”，子 agent 负责“把每条候选路径真正做出来”，然后系统把一次次样本级经验先记成 notes，再蒸馏成 memory 和 skills，最终在推理阶段复用这些外部知识。**

---

## 17. 最适合口头展示的版本

你也可以直接这么讲：

1. 给定一个时序任务，主 agent 不会马上只走一条路径。
2. 它会派多个子 agent 去探索不同工具路径。
3. 每个子 agent 都给出一个合法答案。
4. 主 agent 再用真实指标比较这些答案，选出最优路径。
5. 这次比较学习的结果先写成 notes。
6. 多条 notes 再蒸馏成结构化 memory。
7. memory 进一步整理成更适合测试时复用的 skills。
8. 以后推理时，agent 不是重新学习，而是把这些 artifact 再拿出来用。

这 8 句话基本就能把 TimeClaw 的主线讲清楚。

