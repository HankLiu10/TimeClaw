"""Prompt templates for the Executor Agent — mini-ReAct architecture.

The Executor uses a step-by-step Think→Act→Observe loop instead of
planning the entire tool chain upfront. Each step, the LLM sees the
results of previous tools and decides what to do next.

Tool selection guidelines are derived from time-series research.
"""

# ── Tool Selection Guide ─────────────────────────────────────────────

TOOL_GUIDE = {
    # ── PREDICTION ──
    "arima_forecast": {
        "when": (
            "Stationary or weakly trended univariate series with clear autocorrelation "
            "(significant ACF/PACF lags). Best for short-to-medium term forecasts. "
            "Needs ≥50 data points for reliable parameter estimation."
        ),
        "avoid": (
            "Multiple seasonalities, highly nonlinear dynamics, very short series (<30 pts), "
            "long seasonal periods (m>50), or data with structural breaks."
        ),
    },
    "ets_forecast": {
        "when": (
            "Series with clear level/trend/seasonal components. Works with as few as 20 pts. "
            "Prefer over ARIMA when: series is short (<50 pts), multiplicative seasonality "
            "is present, or weak autocorrelation structure."
        ),
        "avoid": (
            "Complex autocorrelation patterns, multiple seasonalities, "
            "strong autoregressive behavior beyond trend/seasonal."
        ),
    },
    "prophet_forecast": {
        "when": (
            "Daily/sub-daily data with multiple overlapping seasonalities "
            "(daily+weekly+yearly). Handles holidays, missing data, and changepoints. "
            "Needs ≥1 year of daily data (hundreds of points)."
        ),
        "avoid": (
            "Short series (<100 pts), high-frequency sensor data without calendar patterns, "
            "sub-daily data without daily/weekly cycles."
        ),
    },
    "sbm_forecast": {
        "when": (
            "Zero-shot / cold-start scenarios with very short history (<50 pts). "
            "Complex nonlinear patterns that don't fit ARIMA/ETS assumptions. "
            "Foundation models (Chronos, Moirai, TimesFM, Timer) leverage cross-series pre-training."
        ),
        "avoid": (
            "Very long clean stationary series where tuned ARIMA is faster and equally accurate. "
            "When interpretability is required."
        ),
    },
    "ensemble_forecast": {
        "when": (
            "High-stakes predictions. When no single model clearly dominates. "
            "Combining 3-5 diverse models typically reduces error by 10-30%%. "
            "Requires multiple forecasts already computed."
        ),
        "avoid": "When one model is clearly superior — ensembling adds noise from inferior models.",
    },
    "checkpoint_forecast": {
        "when": "Domain-specific fine-tuned model checkpoint is available on disk.",
        "avoid": "No checkpoint available. General-purpose tasks.",
    },

    # ── DECOMPOSITION ──
    "stl_decompose": {
        "when": (
            "Standard first step for seasonal time series. Separates trend, seasonal, residual. "
            "Needs ≥2 full seasonal cycles. Seasonal period must be known — "
            "use detect_seasonality first to get the period, then pass it as args."
        ),
        "avoid": "Multiple overlapping seasonalities, non-periodic patterns, very short series.",
    },
    "fft_spectrum": {
        "when": (
            "Discover dominant frequencies/periods in the data. Detect hidden periodicities. "
            "Needs ≥4-5 complete cycles of the longest period of interest."
        ),
        "avoid": "Non-stationary data (use wavelets), very short series, irregularly sampled data.",
    },
    "ssa_decompose": {
        "when": (
            "Non-parametric decomposition without specifying periods in advance. "
            "Discovers trend and oscillatory components automatically. Works with ≥50 pts."
        ),
        "avoid": "Very short series (<30 pts), components with very similar frequencies.",
    },
    "emd_decompose": {
        "when": (
            "Nonlinear, non-stationary data where frequency content changes over time. "
            "Fully adaptive — no parameters needed."
        ),
        "avoid": "Short series (<100 pts), when reproducibility is critical.",
    },
    "wavelet_transform": {
        "when": (
            "Time-frequency analysis — need to know WHEN frequencies appear/disappear. "
            "Non-stationary data with transient events or changing periodicities."
        ),
        "avoid": "Stationary data (use FFT instead, simpler).",
    },

    # ── DETECTION ──
    "detect_trend": {
        "when": "Always useful as a first diagnostic step. Linear regression + p-value test.",
        "avoid": "Confusing long-period cycles for trends.",
    },
    "detect_anomaly": {
        "when": "Quick z-score based outlier detection. Works on roughly Gaussian, stationary data.",
        "avoid": "Seasonal data (peaks get flagged), non-Gaussian distributions, trended data.",
    },
    "detect_changepoint": {
        "when": (
            "Critical before fitting any forecast model. Detects regime changes. "
            "Uses PELT algorithm."
        ),
        "avoid": "Gradual drift (use detect_drift instead), very noisy data.",
    },
    "detect_seasonality": {
        "when": "Discover seasonal period from ACF peaks. Run before STL or SARIMA.",
        "avoid": "Very short series (<3 cycles of expected period).",
    },
    "detect_drift": {
        "when": "Check if data distribution has shifted (CUSUM + PSI).",
        "avoid": "Very short series where split halves are too small.",
    },

    # ── TEXT ──
    "text_to_event": {
        "when": "Extract structured events with timestamps from news/reports. Critical for multimodal fusion.",
        "avoid": "Very short text without temporal information.",
    },
    "summarize": {
        "when": "Condense long text into actionable insights. Use 'heavy' model for long/complex documents.",
        "avoid": "Already short text (1-2 sentences).",
    },
    "sentiment": {
        "when": "Quick emotional tone assessment of text. Useful for market/social sentiment.",
        "avoid": "Factual/technical text where sentiment is irrelevant.",
    },
    "ner": {
        "when": "Extract named entities (dates, numbers, organizations, events) from text.",
        "avoid": "Pure numerical data without text.",
    },
    "keyword_extract": {
        "when": "Extract key terms and topics from text. Quick overview of text content.",
        "avoid": "When full event extraction is needed (use text_to_event instead).",
    },
    "temporal_align_text": {
        "when": "Align text events to time-series windows by timestamp. Essential for multimodal analysis.",
        "avoid": "Text without timestamps.",
    },
}

# ── Candidate tool sets per modality × task_type ──────────────────────

CANDIDATE_TOOLS = {
    ("timeseries", "forecast"): [
        "normalize", "arima_forecast", "ets_forecast", "sbm_forecast",
        "stl_decompose", "detect_trend", "detect_seasonality", "detect_changepoint",
        "ensemble_forecast", "prophet_forecast",
    ],
    ("timeseries", "anomaly"): [
        "normalize", "detect_anomaly", "detect_changepoint", "detect_drift",
        "detect_regime", "stl_decompose", "basic_stats",
    ],
    ("timeseries", "classification"): [
        "basic_stats", "autocorrelation", "stationarity_test", "entropy",
        "catch22_features", "hurst_exponent", "detect_trend", "fft_spectrum",
    ],
    ("timeseries", "imputation"): [
        "handle_missing", "normalize", "detect_trend", "stl_decompose",
        "basic_stats",
    ],
    ("timeseries", "qa"): [
        "basic_stats", "detect_trend", "autocorrelation", "stationarity_test",
        "detect_anomaly", "detect_seasonality", "stl_decompose",
    ],
    ("text", "forecast"): [
        "text_to_event", "summarize", "sentiment", "temporal_align_text",
        "ner", "keyword_extract",
    ],
    ("text", "qa"): [
        "summarize", "keyword_extract", "ner", "sentiment",
        "text_to_event", "text_similarity",
    ],
    ("text", "anomaly"): [
        "text_to_event", "sentiment", "ner", "keyword_extract",
        "temporal_align_text",
    ],
    ("text", "classification"): [
        "sentiment", "keyword_extract", "ner", "text_embed",
        "summarize",
    ],
    ("image", "forecast"): [
        "image_load", "image_feature", "chart_to_data", "temporal_align_image",
    ],
    ("image", "qa"): [
        "image_load", "image_feature", "image_ocr", "image_classify",
    ],
    ("image", "anomaly"): [
        "image_load", "image_feature", "image_classify",
    ],
    ("image", "classification"): [
        "image_load", "image_feature", "image_classify",
    ],
}

_DEFAULT_TS = ["normalize", "arima_forecast", "basic_stats", "detect_trend"]
_DEFAULT_TEXT = ["summarize", "sentiment", "keyword_extract", "text_to_event"]
_DEFAULT_IMAGE = ["image_load", "image_feature"]


def get_candidate_tools(modality: str, task_type: str) -> list[str]:
    """Return the reduced candidate tool set for a given modality and task type."""
    key = (modality, task_type)
    if key in CANDIDATE_TOOLS:
        return CANDIDATE_TOOLS[key]
    if modality == "timeseries":
        return _DEFAULT_TS
    if modality == "text":
        return _DEFAULT_TEXT
    if modality == "image":
        return _DEFAULT_IMAGE
    return _DEFAULT_TS


def build_tool_guide_text(
    tool_names: list[str],
    experience_insights: dict[str, str] | None = None,
) -> str:
    """Build a concise tool selection guide for the given candidate tools."""
    lines = []
    for name in tool_names:
        guide = TOOL_GUIDE.get(name)
        if guide:
            lines.append(f"- **{name}**: USE WHEN: {guide['when']}")
            if guide.get("avoid"):
                lines.append(f"  AVOID: {guide['avoid']}")
        else:
            lines.append(f"- **{name}**: (general-purpose tool)")
        if experience_insights and name in experience_insights:
            lines.append(f"  LEARNED: {experience_insights[name]}")
    return "\n".join(lines)


# ── Mini-ReAct Prompt Templates ──────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are the Executor Agent of TimeClaw. You solve tasks step-by-step using tools.

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
5. Keep total steps ≤ {max_steps}. Be efficient — don't run tools that won't help.
6. When you have a prediction or have extracted all useful information, set done=true.

## CRITICAL — Task-specific requirements
- **forecast**: You MUST call at least one *_forecast tool (arima_forecast, ets_forecast,
  sbm_forecast, prophet_forecast, ensemble_forecast) to produce a numeric prediction.
  Decomposition and detection tools alone are NOT sufficient — they are preprocessing
  steps that should INFORM your choice of forecast tool. Do NOT declare done until
  you have a forecast result.
- **anomaly**: You MUST call at least one detection tool (detect_anomaly, detect_changepoint, etc.).
- **text modality**: Decomposition/detection/forecast tools are NOT available to you.
  Focus on text analysis tools only.

## Output Format (strict JSON)
{{
  "thought": "what I learned from previous results and what I should do next",
  "done": false,
  "tool": "tool_name",
  "args": {{"key": "value"}}
}}

When done:
{{
  "thought": "summary of what was accomplished",
  "done": true
}}
"""

REACT_FIRST_TURN = """\
Task: {task_type}
Data fingerprint: {fingerprint}

{strategy_section}\
{experience_section}\
## Candidate Tools
{tool_guide}

This is step 1. Analyze the fingerprint and decide which tool to run first.
"""

REACT_NEXT_TURN = """\
Step {step}/{max_steps}. Previous results:

{history}

Decide: run another tool, or declare done?
"""
