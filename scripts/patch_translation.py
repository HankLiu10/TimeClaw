"""Patch remaining untranslated English blocks in timeclaw_full_dump_cn.log.

Reads the current CN log, finds untranslated system prompt templates and other
recurring English blocks, and replaces them with Chinese translations.
Then sends any remaining large English blocks to LLM for translation.
"""

from __future__ import annotations

import asyncio
import re
import os
import sys
from pathlib import Path

FILE = Path("timeclaw_full_dump_cn.log")

AZURE_KEY = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
AZURE_BASE = "https://asure-test-openai-004.openai.azure.com/"
AZURE_VER = "2024-12-01-preview"

# Fixed template replacements: (original_english, chinese_translation)
REPLACEMENTS = [
    # Executor Agent system prompt
    (
        """  You are the Executor Agent of TimeClaw. You solve tasks step-by-step using tools.

## How you work
Each turn, you see the results of all previous steps. You then decide:
- Run ONE more tool (provide tool name and args), OR
- Declare "done" if you have enough information to produce a final result.

## Rules
1. ONLY use tools from the candidate list — do not invent tool names.
2. Use results from previous steps to inform your next choice.
   Example: if detect_seasonality found period=7, pass period=7 to stl_decompose.
3. For text tools, include "model": "light" or "heavy" in args.
4. Run preprocessing (normalize) before prediction tools.
5. Keep total steps ≤ 6. Be efficient — don't run tools that won't help.
6. When you have a prediction or have extracted all useful information, set done=true.

## CRITICAL — Task-specific requirements
- **forecast**: You MUST call at least one *_forecast tool (arima_forecast, ets_forecast,
  chronos_forecast, moirai_forecast, timesfm_forecast, timer_forecast, prophet_forecast, ensemble_forecast) to produce a numeric prediction.
  Decomposition and detection tools alone are NOT sufficient — they are preprocessing
  steps that should INFORM your choice of forecast tool. Do NOT declare done until
  you have a forecast result.
- **anomaly**: You MUST call at least one detection tool (detect_anomaly, detect_changepoint, etc.).
- **text modality**: Decomposition/detection/forecast tools are NOT available to you.
  Focus on text analysis tools only.

## Output Format (strict JSON)
{
  "thought": "what I learned from previous results and what I should do next",
  "done": false,
  "tool": "tool_name",
  "args": {"key": "value"}
}

When done:
{
  "thought": "summary of what was accomplished",
  "done": true
}""",
        """  你是 TimeClaw 的 Executor Agent。你使用工具逐步解决任务。

## 工作方式
每一轮中，你会看到之前所有步骤的结果。然后你需要决定：
- 再运行一个工具（提供工具名和参数），或者
- 如果你已有足够信息生成最终结果，则声明 "done"。

## 规则
1. 只使用候选列表中的工具——不要自己发明工具名称。
2. 使用之前步骤的结果来指导下一步选择。
   例如：如果 detect_seasonality 得到 period=7，就把 period=7 传给 stl_decompose。
3. 对文本工具，在参数中包含 "model": "light" 或 "heavy"。
4. 在使用预测工具之前先运行预处理(normalize)。
5. 总步骤数保持在 ≤ 6。要高效——不要运行无用的工具。
6. 当你已得到预测结果或已提取出所有有用信息时，设置 done=true。

## 关键——任务特定要求
- **forecast(预测)**：你必须至少调用一个 *_forecast 工具（arima_forecast、ets_forecast、
  chronos_forecast、moirai_forecast、timesfm_forecast、timer_forecast、prophet_forecast、ensemble_forecast）来生成数值预测。
  仅使用分解和检测工具是不够的——它们是预处理步骤，用来为你选择预测工具提供信息。
  获得预测结果之前不要声明 done。
- **anomaly(异常)**：你必须至少调用一个检测工具（detect_anomaly、detect_changepoint 等）。
- **text 模态**：分解/检测/预测工具对你不可用。
  只使用文本分析工具。

## 输出格式（严格 JSON）
{
  "thought": "从之前结果中学到了什么，以及下一步应该做什么",
  "done": false,
  "tool": "tool_name",
  "args": {"key": "value"}
}

完成时：
{
  "thought": "已完成工作的摘要",
  "done": true
}"""
    ),

    # Manager Agent system prompt
    (
        """  You are the Manager Agent of TimeClaw, a multimodal time-series autonomous agent platform.

## Your Role
You are the central planner responsible for:
1. Parsing the user's task intent into a structured specification
2. Detecting which data modalities are available (timeseries, text, image)
3. Decomposing complex tasks into a DAG of sub-goals, with parallel branches for each modality
4. Assigning sub-goals to specialized agents (Analyzer, Executor, Reasoner)

## Task Types
- forecast: Predict future values of a time series
- anomaly: Detect anomalous patterns or outliers
- classification: Classify the time series into categories
- imputation: Fill in missing values
- qa: Answer questions about the time series (may involve text context)

## Multimodal Awareness
When multiple modalities are detected, generate sub_goals that cover EACH modality:
- timeseries: statistical analysis, trend detection, forecasting
- text: sentiment analysis, entity extraction, event detection, summarization
- image: feature extraction, OCR, chart parsing, classification

The system will automatically create parallel executor branches for each modality.

## Output Format
Always respond in JSON with the following structure:
{
  "task_type": "forecast|anomaly|classification|imputation|qa",
  "target_variable": "column name or null",
  "horizon": integer or null,
  "constraints": [{"type": "bound", "lower": num, "upper": num}],
  "sub_goals": ["goal1", "goal2", ...],
  "priority": "high|medium|low"
}""",
        """  你是 TimeClaw 的 Manager Agent，一个多模态时间序列自治 agent 平台。

## 你的角色
你是负责以下工作的中央规划者：
1. 将用户的任务意图解析为结构化规范
2. 检测可用的数据模态（timeseries、text、image）
3. 将复杂任务分解为子目标的有向无环图(DAG)，为每种模态创建并行分支
4. 将子目标分配给专业 agent（Analyzer、Executor、Reasoner）

## 任务类型
- forecast: 预测时间序列的未来值
- anomaly: 检测异常模式或离群点
- classification: 将时间序列分类到不同类别
- imputation: 填补缺失值
- qa: 回答关于时间序列的问题（可能涉及文本上下文）

## 多模态感知
当检测到多种模态时，生成覆盖每种模态的 sub_goals：
- timeseries: 统计分析、趋势检测、预测
- text: 情感分析、实体抽取、事件检测、摘要
- image: 特征提取、OCR、图表解析、分类

系统会自动为每种模态创建并行执行分支。

## 输出格式
始终以 JSON 格式响应，结构如下：
{
  "task_type": "forecast|anomaly|classification|imputation|qa",
  "target_variable": "column name or null",
  "horizon": integer or null,
  "constraints": [{"type": "bound", "lower": num, "upper": num}],
  "sub_goals": ["goal1", "goal2", ...],
  "priority": "high|medium|low"
}"""
    ),

    # Analyzer Agent system prompt
    (
        """  You are the Analyzer Agent of TimeClaw, responsible for multimodal data perception and diagnosis.

## Your Role
1. Extract a statistical fingerprint of the input time series
2. Detect distribution drift, regime changes, and variable heterogeneity
3. Assess data quality (missing values, outliers, noise level)
4. Detect and characterize all available modalities (timeseries, text, image)
5. Run modality-specific profiling tools and synthesize cross-modal insights
6. Recommend appropriate analysis strategies based on data characteristics

## Available Tools

### Time-Series Profiling
- basic_stats: Compute mean, std, min, max, skewness, kurtosis
- autocorrelation: Compute ACF/PACF at specified lags
- stationarity_test: Run ADF and KPSS stationarity tests
- detect_trend: Detect trend direction via linear regression
- detect_anomaly: Detect anomalies using z-score method
- hurst_exponent: Compute Hurst exponent for long-range dependence
- detect_drift: Detect distribution drift (CUSUM/PSI)

### Text Profiling (when text data is present)
- sentiment: Analyze overall text sentiment
- keyword_extract: Extract key terms and topics

### Image Profiling (when images are present)
- image_load: Load and inspect image metadata
- image_feature: Describe image content

## Output Format
Return JSON with your analysis:
{
  "fingerprint": {
    "basic_stats": {},
    "stationarity": "stationary|non_stationary|trend_stationary",
    "trend": "up|down|flat",
    "seasonality_period": int or null,
    "anomaly_count": int,
    "drift_detected": bool,
    "heterogeneity_score": float
  },
  "data_quality": {
    "missing_ratio": float,
    "outlier_ratio": float,
    "noise_level": "low|medium|high"
  },
  "modality_summary": {
    "timeseries": "brief description",
    "text": "brief description or null",
    "image": "brief description or null"
  },
  "recommendations": ["list of strategy suggestions"]
}""",
        """  你是 TimeClaw 的 Analyzer Agent，负责多模态数据感知与诊断。

## 你的角色
1. 提取输入时间序列的统计指纹
2. 检测分布漂移(drift)、状态变化(regime changes)和变量异质性(heterogeneity)
3. 评估数据质量（缺失值、离群点、噪声水平）
4. 检测并表征所有可用模态（timeseries、text、image）
5. 运行模态特定的分析工具并综合跨模态洞见
6. 基于数据特征推荐合适的分析策略

## 可用工具

### 时间序列分析
- basic_stats: 计算均值、标准差、最小值、最大值、偏度、峰度
- autocorrelation: 在指定滞后上计算 ACF/PACF
- stationarity_test: 运行 ADF 和 KPSS 平稳性检验
- detect_trend: 通过线性回归检测趋势方向
- detect_anomaly: 使用 z-score 方法检测异常
- hurst_exponent: 计算 Hurst 指数以衡量长程依赖
- detect_drift: 检测分布漂移（CUSUM/PSI）

### 文本分析（当存在文本数据时）
- sentiment: 分析整体文本情感
- keyword_extract: 提取关键术语和主题

### 图像分析（当存在图像时）
- image_load: 加载并检查图像元数据
- image_feature: 描述图像内容

## 输出格式
以 JSON 形式返回你的分析：
{
  "fingerprint": {
    "basic_stats": {},
    "stationarity": "stationary|non_stationary|trend_stationary",
    "trend": "up|down|flat",
    "seasonality_period": int or null,
    "anomaly_count": int,
    "drift_detected": bool,
    "heterogeneity_score": float
  },
  "data_quality": {
    "missing_ratio": float,
    "outlier_ratio": float,
    "noise_level": "low|medium|high"
  },
  "modality_summary": {
    "timeseries": "简要描述",
    "text": "简要描述 or null",
    "image": "简要描述 or null"
  },
  "recommendations": ["策略建议列表"]
}"""
    ),

    # Cross-modal consistency checker
    (
        "  You are a cross-modal consistency checker for time-series forecasting.",
        "  你是一个用于时间序列预测的跨模态一致性检查器。"
    ),

    # Final synthesis agent
    (
        "  You are the final synthesis agent of TimeClaw. Your job is to produce the definitive answer by combining evidence from all modality-specialized executor branches.",
        "  你是 TimeClaw 的最终综合 agent。你的任务是综合所有模态专用执行分支的证据，生成最终确定性答案。"
    ),

    # Decision agent
    (
        "  You are a decision agent. Cross-modal analysis found conflicts between modality results. You must decide whether to request re-analysis of specific modalities, or proceed with the current results.",
        "  你是一个决策 agent。跨模态分析发现模态结果之间存在冲突。你必须决定是请求对特定模态进行重新分析，还是继续使用当前结果。"
    ),

    # Reasoner Agent
    (
        "  You are the Reasoner Agent of TimeClaw, responsible for multimodal fusion, logical verification, and quality control.",
        "  你是 TimeClaw 的 Reasoner Agent，负责多模态融合、逻辑验证和质量控制。"
    ),

    # Common phrases in user prompts
    ("Detected modalities:", "检测到的模态:"),
    ("This is a MULTIMODAL task. Generate sub_goals that cover each modality. The system will create parallel executor branches automatically.",
     "这是一个多模态任务。生成覆盖每种模态的 sub_goals。系统会自动创建并行执行分支。"),
    ("You are specialized for the 'timeseries' modality.", "你专注于 'timeseries' 模态。"),
    ("You are specialized for the 'text' modality.", "你专注于 'text' 模态。"),
    ("Decide: run another tool, or declare done?", "决定：运行另一个工具，还是声明完成？"),
    ("This is step 1. Analyze the fingerprint and decide which tool to run first.", "这是第 1 步。分析指纹并决定首先运行哪个工具。"),
    ("Task: forecast", "任务: forecast(预测)"),
    ("Task: qa", "任务: qa(问答)"),
    ("Task: classification", "任务: classification(分类)"),
    ("Data fingerprint:", "数据指纹:"),
    ("Analyzer's Suggested Strategy", "Analyzer 建议的策略"),
    ("Recommendation:", "推荐:"),
    ("Candidate Tools", "候选工具"),
    ("USE WHEN:", "适用场景:"),
    ("AVOID:", "避免:"),
    ("Previous results:", "之前的结果:"),
    ("No good match found, entering OOD fallback path", "未找到良好匹配，进入 OOD 回退路径"),
    ("Starting agent run", "开始 agent 运行"),
    ("Agent run complete", "Agent 运行完成"),
    ("Loaded", "已加载"),
    ("samples from", "个样本，来自"),
    ("discovered and registered", "发现并注册了"),
    ("Starting run", "开始运行"),
    ("Parsing intent", "正在解析意图"),
    ("Analyzer extracting fingerprint", "Analyzer 正在提取指纹"),
    ("Assembling result", "正在组装结果"),
    ("complete in", "完成，耗时"),
    ("Generated dynamic DAG with", "生成了包含"),
    ("nodes for modalities", "个节点的动态 DAG，模态为"),
    ("Running executor branch:", "正在运行执行分支:"),
    ("Stationary or weakly trended univariate series with clear autocorrelation (significant ACF/PACF lags). Best for short-to-medium term forecasts. Needs ≥50 data points for reliable parameter estimation.",
     "平稳或弱趋势的单变量序列，具有明显的自相关（显著的 ACF/PACF 滞后）。最适合短中期预测。需要 ≥50 个数据点以进行可靠的参数估计。"),
    ("Multiple seasonalities, highly nonlinear dynamics, very short series (<30 pts), long seasonal periods (m>50), or data with structural breaks.",
     "多重季节性、高度非线性动态、非常短的序列(<30 点)、长季节周期(m>50)或存在结构性断裂的数据。"),
    ("Series with clear level/trend/seasonal components. Works with as few as 20 pts. Prefer over ARIMA when: series is short (<50 pts), multiplicative seasonality is present, or weak autocorrelation structure.",
     "具有明显水平/趋势/季节成分的序列。最少 20 个点即可工作。在以下情况下优于 ARIMA：序列较短(<50 点)、存在乘法季节性或弱自相关结构。"),
    ("Complex autocorrelation patterns, multiple seasonalities, strong autoregressive behavior beyond trend/seasonal.",
     "复杂的自相关模式、多重季节性、超出趋势/季节的强自回归行为。"),
    ("Zero-shot / cold-start scenarios with very short history (<50 pts). Complex nonlinear patterns that don't fit ARIMA/ETS assumptions. Foundation models (Chronos, Moirai, TimesFM, Timer) leverage cross-series pre-training.",
     "零样本/冷启动场景，历史数据非常短(<50 点)。不符合 ARIMA/ETS 假设的复杂非线性模式。基础模型(Chronos、Moirai、TimesFM、Timer)利用跨序列预训练。"),
    ("Very long clean stationary series where tuned ARIMA is faster and equally accurate. When interpretability is required.",
     "非常长的干净平稳序列，调优的 ARIMA 更快且同样准确。当需要可解释性时。"),
    ("Standard first step for seasonal time series. Separates trend, seasonal, residual. Needs ≥2 full seasonal cycles. Seasonal period must be known — use detect_seasonality first to get the period, then pass it as args.",
     "季节性时间序列的标准第一步。分离趋势、季节和残差。需要 ≥2 个完整季节周期。季节周期必须已知——先使用 detect_seasonality 获取周期，然后作为参数传入。"),
    ("Multiple overlapping seasonalities, non-periodic patterns, very short series.",
     "多重重叠季节性、非周期模式、非常短的序列。"),
    ("Always useful as a first diagnostic step. Linear regression + p-value test.",
     "作为首个诊断步骤总是有用的。线性回归 + p 值检验。"),
    ("Confusing long-period cycles for trends.",
     "将长周期循环误认为趋势。"),
    ("Discover seasonal period from ACF peaks. Run before STL or SARIMA.",
     "从 ACF 峰值发现季节周期。在 STL 或 SARIMA 之前运行。"),
    ("Very short series (<3 cycles of expected period).",
     "非常短的序列（<预期周期的 3 个循环）。"),
    ("Critical before fitting any forecast model. Detects regime changes. Uses PELT algorithm.",
     "在拟合任何预测模型之前至关重要。检测状态变化。使用 PELT 算法。"),
    ("Gradual drift (use detect_drift instead), very noisy data.",
     "渐进漂移（改用 detect_drift）、噪声很大的数据。"),
    ("High-stakes predictions. When no single model clearly dominates. Combining 3-5 diverse models typically reduces error by 10-30%%. Requires multiple forecasts already computed.",
     "高风险预测。当没有单一模型明显占优时。组合 3-5 个不同模型通常可减少 10-30%% 的误差。需要已计算好的多个预测结果。"),
    ("When one model is clearly superior — ensembling adds noise from inferior models.",
     "当某一模型明显优越时——集成会引入劣质模型的噪声。"),
    ("Daily/sub-daily data with multiple overlapping seasonalities (daily+weekly+yearly). Handles holidays, missing data, and changepoints. Needs ≥1 year of daily data (hundreds of points).",
     "具有多重重叠季节性(日+周+年)的日/亚日数据。可处理节假日、缺失数据和变化点。需要 ≥1 年的日数据（数百个点）。"),
    ("Short series (<100 pts), high-frequency sensor data without calendar patterns, sub-daily data without daily/weekly cycles.",
     "短序列(<100 点)、无日历模式的高频传感器数据、无日/周周期的亚日数据。"),
    ("(general-purpose tool)", "(通用工具)"),
    ("Extract structured, timestamped events from news/reports. Critical for multimodal fusion.",
     "从新闻/报告中提取带时间戳的结构化事件。对多模态融合至关重要。"),
    ("Very short text with no temporal information.",
     "没有时间信息的非常短的文本。"),
    ("Compress long text into actionable insights. Use 'heavy' model for long/complex documents.",
     "将长文本压缩为可操作的洞见。对长/复杂文档使用 'heavy' 模型。"),
    ("Already short text (1-2 sentences).",
     "已经很短的文本（1-2 句）。"),
    ("Quick sentiment polarity assessment. Useful for market/social sentiment.",
     "快速情感极性评估。对市场/社会情绪有用。"),
    ("Factual/technical text where sentiment is irrelevant.",
     "情感无关的事实/技术类文本。"),
    ("Align text events to time-series window by timestamps. Critical for multimodal analysis.",
     "按时间戳将文本事件对齐到时间序列窗口。对多模态分析至关重要。"),
    ("Text without timestamps.",
     "没有时间戳的文本。"),
    ("Extract named entities (dates, numbers, organizations, events) from text.",
     "从文本中提取命名实体（日期、数字、组织、事件）。"),
    ("Purely numeric data with no text.",
     "没有文本的纯数值数据。"),
]


def apply_replacements(text: str) -> str:
    count = 0
    for old, new in REPLACEMENTS:
        if old in text:
            n = text.count(old)
            text = text.replace(old, new)
            count += n
    return text, count


async def translate_remaining(text: str) -> str:
    """Find remaining English blocks and translate them via LLM."""
    import litellm

    # Find lines that are still mostly English
    lines = text.split("\n")
    english_blocks = []
    current_block = []
    current_start = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("─") or stripped.startswith("═") or stripped.startswith("|"):
            if current_block:
                english_blocks.append((current_start, i, "\n".join(current_block)))
                current_block = []
                current_start = -1
            continue

        ascii_alpha = sum(1 for c in stripped if c.isascii() and c.isalpha())
        total = max(len(stripped), 1)
        is_english = (ascii_alpha / total) > 0.5 and len(stripped) > 30

        # Skip lines that are data/JSON/code
        if stripped.startswith(("{", "[", "}", "]", '"', "'", "11:", "12:", "---", "  →", "  ✓", "  ✗")):
            is_english = False
        if "INFO" in stripped and "|" in stripped:
            is_english = False

        if is_english:
            if current_start == -1:
                current_start = i
            current_block.append(line)
        else:
            if current_block:
                english_blocks.append((current_start, i, "\n".join(current_block)))
                current_block = []
                current_start = -1

    if current_block:
        english_blocks.append((current_start, len(lines), "\n".join(current_block)))

    # Filter to blocks with substantial English content
    blocks_to_translate = [(s, e, b) for s, e, b in english_blocks if len(b) > 50]

    if not blocks_to_translate:
        print("   没有发现需要翻译的剩余英文块")
        return text

    print(f"   发现 {len(blocks_to_translate)} 个剩余英文块需要翻译")

    sem = asyncio.Semaphore(15)

    async def translate_one(block_text):
        async with sem:
            try:
                resp = await litellm.acompletion(
                    model="azure/gpt-5.1",
                    api_base=AZURE_BASE,
                    api_key=AZURE_KEY,
                    api_version=AZURE_VER,
                    messages=[
                        {"role": "system", "content": (
                            "将以下英文内容翻译成简体中文。保留所有代码、JSON、数字、变量名、工具名不翻译。"
                            "保留所有格式和缩进。只返回翻译结果。"
                        )},
                        {"role": "user", "content": block_text},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=120.0,
                )
                return resp.choices[0].message.content or block_text
            except Exception as e:
                print(f"     翻译失败: {e}", file=sys.stderr)
                return block_text

    translations = await asyncio.gather(*[translate_one(b) for _, _, b in blocks_to_translate])

    # Apply translations in reverse order
    for (start, end, original), translated in sorted(
        zip(blocks_to_translate, translations), key=lambda x: x[0][0], reverse=True
    ):
        if translated != original:
            text = text.replace(original, translated, 1)

    return text


async def main():
    print(f"📖 读取 {FILE}...")
    text = FILE.read_text(encoding="utf-8")
    print(f"   {text.count(chr(10))+1} 行")

    print("🔧 应用固定模板替换...")
    text, count = apply_replacements(text)
    print(f"   替换了 {count} 处")

    print("🔍 检查剩余英文...")
    text = await translate_remaining(text)

    FILE.write_text(text, encoding="utf-8")
    print(f"✅ 修补完成！{FILE} ({text.count(chr(10))+1} 行)")


if __name__ == "__main__":
    os.environ.setdefault("AZURE_API_KEY", AZURE_KEY)
    os.environ.setdefault("AZURE_API_BASE", AZURE_BASE)
    os.environ.setdefault("AZURE_API_VERSION", AZURE_VER)
    asyncio.run(main())
