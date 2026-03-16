# TimeClaw 完整架构报告

> 版本: v1.0 | 日期: 2026-03-13

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构总览](#2-系统架构总览)
3. [核心模块详解](#3-核心模块详解)
4. [工具系统 ClawHub](#4-工具系统-clawhub)
5. [MCP 协议集成](#5-mcp-协议集成)
6. [层次化记忆系统](#6-层次化记忆系统)
7. [Executor Agent 与 ReAct 循环](#7-executor-agent-与-react-循环)
8. [Sub-Agent 机制](#8-sub-agent-机制)
9. [Benchmark 与评估](#9-benchmark-与评估)
10. [训练与学习流程](#10-训练与学习流程)
11. [数据集与适配](#11-数据集与适配)
12. [LLM 路由与模型配置](#12-llm-路由与模型配置)
13. [其他模块](#13-其他模块)
14. [目录结构全览](#14-目录结构全览)
15. [设计决策与 OpenClaw 对比](#15-设计决策与-openclaw-对比)
16. [已知问题与未来方向](#16-已知问题与未来方向)

---

## 1. 项目概述

### 1.1 定位

TimeClaw 是一个原生多模态时序自主 agent 平台。它能处理包括时序数据、文本、图像在内的多模态数据，完成预测 (forecast)、指标提取 (indicator)、趋势分类 (trend)、多选问答 (MCQA)、相关性分析 (correlation) 等任务。

### 1.2 核心理念

- **自主工具选择**：agent 根据任务和数据特征自行选择工具链，不预设固定流程
- **追求最优解**：与 OpenClaw 追求"可行解"不同，TimeClaw 追求最精准的数值结果
- **自我进化**：通过层次化记忆系统从训练数据中积累经验，持续提升性能
- **统一管线**：所有模态和任务类型走同一个 agent 路径，无硬编码分支

### 1.3 技术栈

| 层面 | 技术 |
|------|------|
| 推理引擎 | GPT-5.1 (Azure OpenAI) / Claude / Qwen |
| 工具暴露 | MCP (Model Context Protocol, stdio 传输) |
| 时序基础模型 | TimesFM, Chronos, Timer, Moirai (GPU 服务化) |
| 统计模型 | Prophet, ARIMA, ETS, Ensemble |
| 向量存储 | SQLite + TF-IDF |
| 配置管理 | Pydantic + YAML |
| 并发 | asyncio |

---

## 2. 系统架构总览

### 2.1 单 Agent 架构

TimeClaw 的核心是一个单 Executor Agent，采用 mini-ReAct（Think -> Act -> Observe）循环。这是经过多次架构迭代后的最终方案——从最初的 4-agent 管线（Manager -> Analyzer -> Executor -> Reasoner）简化而来。

```
                    +-------------------------------------+
                    |           PipelineRunner             |
                    |  (预处理 -> 记忆注入 -> 执行 -> 后处理) |
                    +--------------+----------------------+
                                   |
                    +--------------v----------------------+
                    |         ExecutorAgent                |
                    |       (mini-ReAct Loop)              |
                    |                                      |
                    |  Step 1: Think -> 选择工具            |
                    |  Step 2: Act -> 执行工具              |
                    |  Step 3: Observe -> 分析结果          |
                    |  ...                                 |
                    |  Step N: Done -> 输出最终结果         |
                    |                                      |
                    |  [可选] spawn_subagent -> 并行子任务   |
                    +--------------+----------------------+
                                   |
           +-----------------------+------------------------+
           |                       |                        |
    +------v------+   +------------v----------+   +---------v-----+
    |  ClawHub    |   |   MCP Servers (9)     |   |  SBM Service  |
    |  (64 tools) |   |   (stdio transport)   |   |  (4 models)   |
    +-------------+   +-----------------------+   +---------------+
```

### 2.2 数据流

```
输入 (时序 + 文本 + 图像)
    |
PipelineRunner.run()
    +-- 1. 模态检测 (ModalityDetector)
    +-- 2. 指纹提取 (FingerprintExtractor)
    +-- 3. 策略匹配 (FingerprintMatcher)
    +-- 4. 记忆注入 (HierarchicalMemory -> envelope.metadata)
    |       +-- SOUL -> system_prompt
    |       +-- PRINCIPLES -> system_prompt
    |       +-- MEMORY (L2) -> system_prompt
    |       +-- Daily Notes (L1) -> first_user
    |       +-- PLAYBOOKS -> first_user
    |       +-- TOOLS.md -> tool_guide
    +-- 5. 构建 TaskEnvelope (TEA Protocol)
    +-- 6. Executor.perceive() -> plan() -> act()
    |       +-- mini-ReAct: 最多 10 步工具调用
    +-- 7. 后处理 (QualityGate, 反归一化)
    +-- 8. 输出 + 经验记录
```

---

## 3. 核心模块详解

### 3.1 TEA Protocol (timeclaw/core/tea_protocol.py)

Task-Environment-Agent 三元组：

| 组件 | 类名 | 内容 |
|------|------|------|
| Task | TaskSpec | task_type, description, horizon, constraints, metadata |
| Environment | EnvironmentSnapshot | data (ndarray), text_data, images, fingerprint, data_shape, modalities |
| Envelope | TaskEnvelope | 封装 Task + Environment，在 agent 间传递 |

### 3.2 Agent 基类 (timeclaw/core/agent_base.py)

定义四阶段生命周期：

```
perceive(envelope) -> 感知环境和任务
    |
plan(perception) -> 规划（准备工具列表、记忆注入）
    |
act(plan, envelope) -> 执行（ReAct 循环）
    |
reflect(action_result) -> 反思（评估结果）
```

### 3.3 配置系统 (timeclaw/core/config.py)

Pydantic 模型，YAML 驱动：

| 配置项 | 说明 |
|--------|------|
| LLMModelConfig | model, provider, api_base, api_key, temperature |
| PipelineConfig | enable_mcp, disable_memory, max_react_steps |
| MCPConfig | transport, servers (per-category) |
| ReasoningConfig | max_depth, confidence_threshold |

### 3.4 证据系统 (timeclaw/core/evidence.py)

- EvidenceEntry：记录每步的 thought, tool_name, tool_args, observation, critic_feedback
- EvidenceLog：有序日志，支持质量门控和约束检查
- QualityGate：边界检查、一致性校验

### 3.5 上下文管理 (timeclaw/core/context_manager.py)

管理 LLM 上下文窗口，处理超长对话的截断策略。

---

## 4. 工具系统 ClawHub

### 4.1 基类 (timeclaw/clawhub/tool_base.py)

```python
class ClawTool:
    name: str
    description: str
    category: ToolCategory
    io_type: ToolIOType
    parameters_schema: dict
    
    async def execute(self, data, text, images, llm, **kwargs) -> ToolResult
```

ToolResult 包含 success, data, metadata, error。

### 4.2 工具注册 (timeclaw/clawhub/tool_registry.py)

- discover_and_register()：自动发现所有子包中的工具实例
- register_tool_instance()：手动注册
- get_tool_instance() / list_tool_instances()：查询

### 4.3 全部 64 个工具

#### 处理 (Processing) - 6 个

| 工具 | 功能 |
|------|------|
| normalize | Z-score / Min-max 归一化 |
| handle_missing | 缺失值处理（插值/填充） |
| select_channels | 多变量通道选择 |
| segment | 时序分段 |
| resample | 重采样 |
| slice | 切片 |

#### 分析 (Profiling) - 8 个

| 工具 | 功能 |
|------|------|
| basic_stats | 均值、标准差、偏度、峰度等 |
| autocorrelation | 自相关分析 |
| stationarity_test | ADF + KPSS 平稳性检验 |
| catch22_features | 22 维时序特征 |
| distribution_fit | 分布拟合 |
| entropy | 信息熵 |
| hurst_exponent | Hurst 指数 |
| spectral_density | 频谱密度 |

#### 检测 (Detection) - 7 个

| 工具 | 功能 |
|------|------|
| detect_anomaly | Z-score 异常检测 |
| classify_noise_type | 噪声类型分类 |
| detect_trend | 线性回归趋势检测 |
| detect_regime | 制度转换检测 |
| detect_seasonality | 季节性周期检测 |
| detect_drift | 分布漂移检测 (CUSUM/PSI) |
| detect_changepoint | 变点检测 |

#### 分解 (Decomposition) - 6 个

| 工具 | 功能 |
|------|------|
| stl_decompose | STL 分解（趋势+季节+残差） |
| fft_spectrum | FFT 频谱分析 |
| emd_decompose | 经验模态分解 |
| ssa_decompose | 奇异谱分析 |
| variance_decompose | 方差分解 |
| wavelet_transform | 小波变换 |

#### 关系 (Relation) - 8 个

| 工具 | 功能 |
|------|------|
| correlation_matrix | 相关性矩阵 |
| granger_causality | Granger 因果检验 |
| dtw_distance | 动态时间规整距离 |
| heterogeneity_score | 异质性评分 |
| transfer_entropy | 传递熵 |
| lagged_correlation | 滞后相关 |
| mutual_information | 互信息 |
| cointegration_test | 协整检验 |

#### 预测 (Prediction) - 10 个

| 工具 | 功能 |
|------|------|
| arima_forecast | ARIMA 模型预测 |
| ets_forecast | 指数平滑预测 |
| prophet_forecast | Prophet 模型预测 |
| ensemble_forecast | 集成预测 |
| checkpoint_forecast | 领域特定检查点预测 |
| llm_indicator | LLM 推理型指标/预测工具 |
| timesfm_forecast | TimesFM 基础模型预测 |
| chronos_forecast | Chronos 基础模型预测 |
| timer_forecast | Timer 基础模型预测 |
| moirai_forecast | Moirai 基础模型预测 |

#### 验证 (Validation) - 5 个

| 工具 | 功能 |
|------|------|
| bound_check | 边界检查 |
| consistency_check | 一致性校验 |
| fol_check | 一阶逻辑检查 |
| uncertainty_quantify | 不确定性量化 |

#### 文本 (Text) - 8 个

| 工具 | 功能 |
|------|------|
| summarize | 文本摘要 |
| sentiment | 情感分析 |
| ner | 命名实体识别 |
| keyword_extract | 关键词提取 |
| text_to_event | 文本转结构化事件 |
| text_embed | 文本向量化 |
| text_similarity | 文本相似度 |
| temporal_align_text | 文本时序对齐 |

#### 图像 (Image) - 6 个

| 工具 | 功能 |
|------|------|
| image_load | 加载/预处理图像 |
| image_feature | 图像特征提取 (LLM) |
| image_ocr | OCR 文字识别 |
| image_classify | 图像分类 |
| chart_to_data | 图表转数值数据 |
| temporal_align_image | 图像时序对齐 |

---

## 5. MCP 协议集成

### 5.1 架构

采用 Anthropic 的 Model Context Protocol (MCP) 暴露工具，按类别分为 9 个独立服务器：

```
MCPManager
+-- timeclaw-processing    (6 tools, stdio)
+-- timeclaw-profiling     (8 tools, stdio)
+-- timeclaw-detection     (7 tools, stdio)
+-- timeclaw-decomposition (6 tools, stdio)
+-- timeclaw-relation      (8 tools, stdio)
+-- timeclaw-prediction    (10 tools, stdio)
+-- timeclaw-validation    (5 tools, stdio)
+-- timeclaw-text          (8 tools, stdio)
+-- timeclaw-image         (6 tools, stdio)
```

### 5.2 核心文件

| 文件 | 职责 |
|------|------|
| timeclaw/mcp/manager.py | 管理所有 MCP 服务器的生命周期和工具调度 |
| timeclaw/mcp/server.py | 将 ClawHub 工具暴露为 MCP 工具 |
| timeclaw/mcp/server_factory.py | 工厂模式：ClawTool -> FastMCP 工具注册 |
| timeclaw/mcp/client.py | MCP 客户端，消费外部 MCP 工具 |
| timeclaw/mcp/servers/*.py | 9 个类别服务器的启动脚本 |

### 5.3 传输协议

采用 stdio 传输：
- 每个类别服务器作为独立子进程运行
- 通过 JSON-RPC 2.0 通信
- 支持 tools/list 和 tools/call 方法

### 5.4 工具调用流程

```
Executor -> MCPManager.call_tool(name, args)
    -> MCPManager 查找工具所属服务器
    -> 通过 stdio 发送 JSON-RPC 请求
    -> 服务器内 server_factory 的 handler 执行工具
    -> 返回 ToolResult
```

### 5.5 类型安全

server_factory.py 中的 _handler 函数包含通用类型转换逻辑：如果参数声明为 string 但收到 dict/list，自动 json.dumps() 序列化。

---

## 6. 层次化记忆系统

### 6.1 六组件架构

| 组件 | 类型 | 存储位置 | 注入位置 |
|------|------|----------|----------|
| SOUL.md | 固定（手写） | timeclaw/memory/SOUL.md | Executor system prompt |
| PRINCIPLES.md | 学习（跨域元策略） | .timeclaw/hierarchical_memory/principles/ | Executor system prompt |
| MEMORY.md | 学习（因果洞察） | .timeclaw/hierarchical_memory/memory/ | Executor system prompt |
| TOOLS.md | 学习（工具使用笔记） | .timeclaw/hierarchical_memory/TOOLS.md | Tool guide 中的 NOTE |
| PLAYBOOKS/ | 学习（场景 SOP） | .timeclaw/hierarchical_memory/playbooks/ | Executor first-turn message |
| Daily Notes | 原始记录 | .timeclaw/hierarchical_memory/daily/ | Executor first-turn message |

### 6.2 蒸馏流程

```
Daily Notes (L1, 原始运行记录)
    | 每 30 条触发 LLM 蒸馏
    v
MEMORY (L2, 因果洞察)  <-- 按 (domain, task_type) 分组
    | 每 30 条 L2 触发
    v
PRINCIPLES (L3, 跨域元策略)

同时:
Daily Notes -> TOOLS.md (工具使用经验)
MEMORY -> PLAYBOOKS/ (场景化 SOP)
```

### 6.3 蒸馏 Agent (DistillationAgent)

- 位置：timeclaw/memory/hierarchical.py
- 触发条件：count % 30 == 0
- 蒸馏方向：
  - Notes -> MEMORY：提取因果规则
  - Notes -> TOOLS：总结工具使用心得
  - MEMORY -> PRINCIPLES：提炼跨域知识
  - MEMORY -> PLAYBOOKS：生成操作手册
- 冲突解决：LLM 驱动的知识合并和修订

### 6.4 语义检索

- 向量存储：SQLite + TF-IDF (timeclaw/memory/vector_store.py)
- 检索时机：查询 Daily Notes 时按指纹/关键词语义匹配
- 注入策略：同时注入三层记忆（SOUL 在 system prompt，其余在 user message）

### 6.5 训练 vs 测试阶段

| 阶段 | 记忆读取 | 记忆写入 |
|------|----------|----------|
| 训练 | 是 | 是（记录经验 + 触发蒸馏） |
| 测试 | 是（使用学到的知识） | 否 |

---

## 7. Executor Agent 与 ReAct 循环

### 7.1 核心文件

| 文件 | 说明 |
|------|------|
| timeclaw/agents/executor.py | ExecutorAgent 实现 |
| timeclaw/agents/prompts/executor_prompt.py | Prompt 模板 |

### 7.2 ReAct 循环流程

```python
for step in range(1, MAX_REACT_STEPS + 1):
    # 1. 构建/追加 conversation 消息
    # 2. LLM 推理 (JSON format)
    # 3. 解析 decision = {thought, done, tool, args, answer, ...}
    # 4. if done:
    #    a. 提取 answer
    #    b. 自我纠正：answer_type != task_type -> 追加纠正消息 -> continue
    #    c. 输出最终结果
    # 5. if tool == "spawn_subagent":
    #    a. 启动并行 sub-agent
    #    b. 收集结果 -> 写入 conversation
    #    c. continue
    # 6. 执行普通工具
    # 7. 结果写入 conversation
```

### 7.3 上下文管理

采用累积式对话历史：

```python
conversation = [
    {"role": "system", "content": system_prompt},   # 含 SOUL+PRINCIPLES+MEMORY
    {"role": "user",   "content": first_user},       # 含任务描述+数据+工具列表
    # 后续每步追加 assistant reply + tool observation
]
```

每次 llm.complete(conversation) 传入完整历史，确保所有上下文（包括原始文本数据、工具调用结果）在整个 ReAct 循环中持续可见。

### 7.4 自我纠正机制

当 agent 宣布 done=true 但 answer_type 不等于 task_type 时：
1. 追加纠正消息到 conversation
2. 继续 ReAct 循环
3. agent 利用已有上下文重新推理

### 7.5 Horizon 强制

对于 *_forecast 工具，管线指定的 horizon 会强制覆盖 agent 传入的值：
```python
if horizon and tool_name.endswith("_forecast"):
    tool_args["horizon"] = horizon
```

### 7.6 反归一化

如果 ReAct 过程中使用了 normalize 工具，最终预测结果会自动反归一化。

---

## 8. Sub-Agent 机制

### 8.1 概述

主 Executor Agent 可通过 spawn_subagent 工具启动最多 4 个并行 sub-agent，每个 sub-agent 是独立的 mini-ReAct 循环（最多 5 步）。

### 8.2 核心设计

- Sub-agent 继承完整记忆：SOUL + PRINCIPLES + MEMORY + TOOLS + Daily Notes + PLAYBOOKS
- Sub-agent 共享数据和工具：与主 agent 看到相同的 data、text_data、images
- Sub-agent 不能递归 spawn
- 结果以文本摘要形式返回给主 agent 的 conversation

### 8.3 详细技术说明

参见 reports/01_sub_agent_technical_report.md。

---

## 9. Benchmark 与评估

### 9.1 评估框架

| 文件 | 说明 |
|------|------|
| timeclaw/benchmark/pipeline_eval.py | TimeClaw 评估（训练/测试） |
| timeclaw/benchmark/baseline_eval.py | 纯 LLM baseline |
| timeclaw/benchmark/metrics.py | MAE, RMSE, MAPE, SMAPE, MASE, Accuracy, F1 |
| scripts/run_full_benchmark.py | 完整评估脚本 |
| scripts/eval_comparison.py | TimeClaw vs Baseline 比较 |

### 9.2 评估任务

| 任务类型 | 指标 | 数据集 |
|----------|------|--------|
| forecast | MAE, RMSE, MAPE | weather_aligned_short/long, finance_aligned_short/long |
| indicator | MAE (max/min/diff) | weather_aligned_short/long, finance_aligned_short/long |
| trend | Accuracy, F1 (3-way) | weather_aligned_short/long, finance_aligned_short/long |
| mcqa | Accuracy | weather_qa_short/long, finance_qa_short/long |
| correlation | Accuracy (3-way) | finance_qa_short/long |

### 9.3 Baseline 对比

Baseline 使用 MTBench 官方 prompt 直接调用 LLM（无工具），与 TimeClaw 使用相同的输入和输出格式。

### 9.4 评估流程

```
1. 清空记忆和技能
2. 数据集 70/30 时间切分
3. 对每个数据集:
   a. TimeClaw 在训练集上运行（记录经验 + 触发蒸馏）
   b. TimeClaw 在测试集上评估（使用学到的记忆，不写入）
   c. Baseline 在测试集上评估
   d. 对比报告
```

---

## 10. 训练与学习流程

### 10.1 渐进式工具暴露 (Warmup)

训练前 50% 的样本，对同质化工具组（如所有预测工具）逐步增加暴露数量：

```
训练进度:   0%     25%     50%     75%    100%
可见工具:   1个    2个     3个      全部    全部
```

避免 LLM 对第一个工具（如 Prophet）产生偏好偏置。

### 10.2 竞争性工具选择

从 2 个工具可见开始，强制 agent 额外选一个工具进行竞争：
- 主选工具 + 竞争工具都运行
- 以 ground truth 比较 MAE
- 最优工具记录到 ExperiencePool

### 10.3 经验记录

每个训练样本运行后记录：
- 工具调用链
- 各工具的成功/失败
- 预测精度 (MAE/accuracy)
- 数据指纹
- 推理过程

### 10.4 知识蒸馏

每 30 条经验触发一次 LLM 蒸馏：
- L1 (Daily Notes) -> L2 (MEMORY): 提取因果洞察
- L1 -> TOOLS.md: 工具使用心得
- L2 -> L3 (PRINCIPLES): 跨域元知识
- L2 -> PLAYBOOKS: 场景 SOP

### 10.5 两轮学习（规划中）

- 第一轮：初始探索，每个样本跑一遍
- 第二轮：带着第一轮蒸馏出的记忆再跑一遍，对比改进

---

## 11. 数据集与适配

### 11.1 MTBench 数据集

| 数据集 | 领域 | 输入长度 | 输出长度 | 任务 |
|--------|------|----------|----------|------|
| weather_aligned_short | 天气 | 7 天 | 1 天 | forecast, trend, indicator |
| weather_aligned_long | 天气 | 14 天 | 3 天 | forecast, trend, indicator |
| weather_qa_short | 天气 | - | - | mcqa |
| weather_qa_long | 天气 | - | - | forecast, mcqa |
| finance_aligned_short | 金融 | 7 天 | 1 天 | forecast, trend, indicator |
| finance_aligned_long | 金融 | 30 天 | 7 天 | forecast, trend, indicator |
| finance_qa_short | 金融 | - | - | mcqa, correlation |
| finance_qa_long | 金融 | - | - | mcqa, correlation |

### 11.2 数据切分

- 切分策略：按时间切割（而非随机），避免分布漂移
- 比例：约 70% 训练，30% 测试
- 保障：训练集和测试集无时间重叠

### 11.3 统一适配 (MTBenchSample)

所有数据集被转换为统一的 MTBenchSample 数据类：
- input_window：输入时序 (numpy array)
- output_window：真实值 (numpy array)
- text：文本信息
- question/answer：问答对
- labels：额外标签（趋势、相关性等）

---

## 12. LLM 路由与模型配置

### 12.1 模型路由 (timeclaw/llm/router.py)

| Agent/用途 | 默认模型 | 说明 |
|------------|----------|------|
| Executor | GPT-5.1 (Azure) | 主推理 |
| 工具内 LLM | GPT-5.1 | llm_indicator 等工具 |
| 蒸馏 Agent | GPT-5.1 | 知识蒸馏 |
| Baseline | GPT-5.1 | 对照实验 |

### 12.2 配置文件

```yaml
# configs/gpt51.yaml
llm:
  executor:
    model: gpt-5.1
    provider: azure
    api_base: https://asure-test-openai-004.openai.azure.com/
    api_key: ...
    api_version: 2024-12-01-preview
```

### 12.3 SBM 服务

4 个时序基础模型通过 HTTP 微服务部署：

| 模型 | 适配器 | 运行设备 |
|------|--------|----------|
| TimesFM | timeclaw/sbm/timesfm_adapter.py | GPU |
| Chronos | timeclaw/sbm/chronos_adapter.py | GPU |
| Timer | timeclaw/sbm/timer_adapter.py | GPU |
| Moirai | timeclaw/sbm/moirai_adapter.py | GPU |

通过 timeclaw/sbm/server.py 暴露 HTTP 接口，timeclaw/sbm/http_client.py 调用。

---

## 13. 其他模块

### 13.1 指纹系统 (timeclaw/fingerprint/)

- 提取：对输入数据运行 profiling 工具，生成 DataFingerprint
- 匹配：根据指纹特征匹配推荐策略（如哪种预测方法可能更好）
- 用途：传递给 Executor 作为数据上下文，帮助工具选择

### 13.2 多模态系统 (timeclaw/multimodal/)

- 模态检测器：自动识别输入中的时序/文本/图像
- 时序对齐器：将文本/图像按时间戳对齐到时序窗口
- 融合器：支持多种融合策略（固定/任务自适应/置信度加权/LLM 仲裁）
- 置信度评估：每个模态独立评估可信度

### 13.3 TokenCast (timeclaw/tokencast/)

信号令牌化引擎，将连续时序信号转换为离散 token 序列：
- encoder.py：信号 -> token 映射
- decoder.py：token -> 信号重建
- quantizer.py：量化策略
- vocabulary.py：时序词汇表

### 13.4 调度器 (timeclaw/scheduler/)

- 心跳监控
- Cron 式定时任务
- 内置维护任务

---

## 14. 目录结构全览

```
TimeClaw/
+-- timeclaw/                          # 主包
|   +-- agents/                        # Agent 实现
|   |   +-- executor.py               # 核心 Executor (ReAct + Sub-Agent)
|   |   +-- manager.py                # Manager (legacy)
|   |   +-- reasoner.py               # Reasoner (legacy)
|   |   +-- analyzer.py               # Analyzer (legacy)
|   |   +-- prompts/                  # Prompt 模板
|   +-- benchmark/                     # 评估与训练
|   |   +-- pipeline_eval.py          # TimeClaw 评估
|   |   +-- baseline_eval.py          # Baseline 评估
|   |   +-- mtbench_adapter.py        # 数据集适配
|   |   +-- dataset_split.py          # 时间切分
|   |   +-- train_loop.py            # 训练循环
|   |   +-- skill_distill.py         # 技能蒸馏
|   |   +-- metrics.py               # 评估指标
|   |   +-- tasks/                    # 任务定义
|   +-- clawhub/                       # 工具池 (64 tools)
|   |   +-- tool_base.py              # 基类
|   |   +-- tool_registry.py          # 注册中心
|   |   +-- processing/               # 6 个处理工具
|   |   +-- profiling/                # 8 个分析工具
|   |   +-- detection/                # 7 个检测工具
|   |   +-- decomposition/            # 6 个分解工具
|   |   +-- relation/                 # 8 个关系工具
|   |   +-- prediction/               # 10 个预测工具
|   |   +-- validation/               # 5 个验证工具
|   |   +-- text/                     # 8 个文本工具
|   |   +-- image/                    # 6 个图像工具
|   +-- core/                          # 核心抽象
|   |   +-- agent_base.py             # Agent 基类
|   |   +-- config.py                 # 配置系统
|   |   +-- tea_protocol.py           # TEA 协议
|   |   +-- evidence.py               # 证据系统
|   |   +-- context_manager.py        # 上下文管理
|   |   +-- message.py                # 消息协议
|   |   +-- trace_logger.py           # 追踪日志
|   |   +-- reasoning.py              # 推理引擎
|   |   +-- registry.py               # 组件注册
|   |   +-- task_graph.py             # 任务图
|   +-- fingerprint/                   # 数据指纹
|   |   +-- extractor.py              # 指纹提取
|   |   +-- matcher.py                # 策略匹配
|   |   +-- schema.py                 # 指纹模式
|   +-- llm/                          # LLM 路由
|   |   +-- base.py                   # LLM 基类
|   |   +-- router.py                 # 模型路由
|   +-- mcp/                          # MCP 协议 (9 个类别服务器)
|   |   +-- manager.py                # 服务器管理
|   |   +-- server.py                 # MCP 服务器
|   |   +-- server_factory.py         # 工厂
|   |   +-- client.py                 # 客户端
|   |   +-- servers/                  # 9 个类别子服务器
|   +-- memory/                        # 层次化记忆 (6 组件)
|   |   +-- hierarchical.py           # 主管理器
|   |   +-- experience_pool.py        # 经验池
|   |   +-- vector_store.py           # 向量存储
|   |   +-- task_strategy.py          # 任务策略
|   |   +-- domain_knowledge.py       # 领域知识
|   |   +-- SOUL.md                   # Agent 身份
|   +-- multimodal/                    # 多模态融合
|   |   +-- detector.py               # 模态检测
|   |   +-- aligner.py                # 时序对齐
|   |   +-- fusion.py                 # 融合策略
|   |   +-- confidence.py             # 置信度
|   +-- pipeline/                      # 管线编排
|   |   +-- runner.py                 # PipelineRunner
|   +-- sbm/                          # 时序基础模型 (4 个)
|   |   +-- server.py                 # HTTP 服务
|   |   +-- http_client.py            # HTTP 客户端
|   |   +-- timesfm_adapter.py        # TimesFM
|   |   +-- chronos_adapter.py        # Chronos
|   |   +-- timer_adapter.py          # Timer
|   |   +-- moirai_adapter.py         # Moirai
|   +-- scheduler/                     # 调度器
|   +-- tokencast/                     # 信号令牌化
+-- configs/                           # YAML 配置
|   +-- default.yaml
|   +-- gpt5.yaml
|   +-- gpt51.yaml
|   +-- qwen25_32b.yaml
|   +-- cloud.yaml
|   +-- agents/                        # Agent 配置
+-- scripts/                           # 脚本工具
|   +-- run_full_benchmark.py          # 完整 benchmark
|   +-- download_mtbench.py            # 下载数据
|   +-- eval_comparison.py             # 评估对比
|   +-- test_single_agent_demo.py      # 单 agent demo
+-- tests/                             # 测试套件
+-- data/                              # 数据集
|   +-- processed/                    # 预处理数据
|   +-- splits/                       # 训练/测试切分
+-- mtbench_official/                  # MTBench 官方评估
+-- reports/                           # 报告
+-- .timeclaw/                         # 运行时状态
|   +-- hierarchical_memory/          # 记忆存储
|   +-- experiences/                  # 经验池
|   +-- eval_results/                 # 评估结果
|   +-- test_results/                 # 测试结果
+-- pyproject.toml                     # 项目配置
+-- README.md
```

---

## 15. 设计决策与 OpenClaw 对比

### 15.1 架构对比

| 维度 | OpenClaw | TimeClaw |
|------|----------|----------|
| Agent 架构 | 单 agent，自动 spawn sub-agents | 单 Executor agent + 可选 sub-agent |
| 目标 | 可行解 (feasible) | 最优解 (optimal) |
| 工具暴露 | MCP (stdio) | MCP (stdio)，与 OpenClaw 一致 |
| 记忆系统 | 5 组件 (SOUL, PRINCIPLES, MEMORY, TOOLS, PLAYBOOKS) | 6 组件 (+ Daily Notes) |
| Sub-agent 上下文 | 精简（仅任务描述） | 完整经验（SOUL + PRINCIPLES + MEMORY + tool_notes） |
| 工具管理 | 工具不重叠 | 同质化工具组 + 渐进暴露 + 竞争选择 |
| 记忆蒸馏 | 基于文件的人工/半自动管理 | LLM 驱动的三层自动蒸馏 |

### 15.2 关键设计决策

1. **单 Agent 而非多 Agent**：减少 agent 间通信开销和信息丢失
2. **累积式对话历史**：确保全部上下文在 ReAct 循环中持续可见
3. **Sub-agent 完整经验继承**：牺牲部分效率换取预测精度
4. **渐进式工具暴露**：解决 LLM 对首个同质化工具的偏好偏置
5. **竞争性工具选择**：训练阶段通过 ground truth 发现最优工具
6. **自我纠正机制**：answer_type 不匹配时自动重试

### 15.3 为什么单 Agent 而非多 Agent

经过多次迭代，从 4-agent (Manager -> Analyzer -> Executor -> Reasoner) 简化为单 Executor：

| 原因 | 说明 |
|------|------|
| 上下文丢失 | 多 agent 间的信息传递不可避免地丢失细节 |
| 延迟 | 每个 agent 都需要一次 LLM 调用，4-agent 意味着至少 4 次 |
| OpenClaw 验证 | OpenClaw 证明了单 agent + ReAct 在大多数场景下足够 |
| 灵活性 | 单 agent 可以根据需要动态决定步骤数量和工具组合 |

---

## 16. 已知问题与未来方向

### 16.1 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| 上下文超长时的截断 | 待优化 | GPT-5.1 400K 窗口下暂不严重 |
| Sub-agent 超时未启用 | 已定义常量 | 需增加 asyncio.wait_for() |
| 同质化工具竞争仍存偏好 | 改善中 | 渐进暴露 + 竞争选择已实施 |
| 多变量预测未实测 | 待验证 | Sub-agent 机制为此设计 |

### 16.2 未来方向

1. **多变量联合预测**：利用 sub-agent 并行预测不同变量
2. **两轮学习**：初始探索 + 带记忆的精细化学习
3. **差异化 sub-agent 模型**：不同 sub-agent 使用不同 LLM
4. **工具自进化**：从经验中自动生成新工具的 wrapper
5. **在线学习**：部署后持续从新数据中学习
6. **跨数据集迁移学习**：天气领域的经验迁移到金融领域

---

## 附录 A: 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_REACT_STEPS | 10 | 主 agent 最大 ReAct 步数 |
| MAX_SUB_REACT_STEPS | 5 | sub-agent 最大步数 |
| MAX_CONCURRENT_SUBAGENTS | 4 | 最大并发 sub-agent 数 |
| _DISTILL_BATCH | 30 | 每 30 条经验触发一次蒸馏 |
| 训练/测试比 | 70/30 | 按时间切分 |
| 工具总数 | 64 | 9 个类别 |

## 附录 B: 历史架构变迁

| 阶段 | 架构 | 说明 |
|------|------|------|
| v0.1 | 4-Agent 管线 | Manager -> Analyzer -> Executor -> Reasoner |
| v0.2 | 简化管线 | Manager -> Executor (Analyzer/Reasoner 内化) |
| v0.3 | 单 Agent | 仅 Executor (mini-ReAct) |
| v0.4 (当前) | 单 Agent + Sub-Agent | Executor + 可选 spawn_subagent |

## 附录 C: 相关报告

- reports/01_sub_agent_technical_report.md — Sub-Agent 机制技术细节
