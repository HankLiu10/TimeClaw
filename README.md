# TimeClaw

**Native Multimodal Time-Series Autonomous Agent Platform**

TimeClaw is a time-series domain autonomous agent operating system that shifts from
"visual-assisted" to "signal-native", from "model-centric" to "governance-centric",
and from "form simulation" to "logical self-consistency".

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────┐
│  Manager Agent  (Hierarchical Planning)     │
│  ── Parse intent → Generate Task DAG ──     │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌─────────┐ ┌────────┐ ┌─────────┐
   │Analyzer │ │Executor│ │Reasoner │
   │(Diagnose)│ │(Tools) │ │(Verify) │
   └────┬────┘ └───┬────┘ └────┬────┘
        │          │           │
        └──────────┼───────────┘
                   ▼
        ┌─────────────────────┐
        │  ClawHub Tool Pool  │
        │  48 Atomic Tools    │
        └─────────────────────┘
```

## Key Features

- **Signal-Native (TokenCast)**: Direct symbolic discretization of raw signals into
  semantic tokens, preserving full precision without "time-series-to-image" conversion.
- **Evidence-Grounded ReflAct**: Three-layer nested reasoning (Hierarchical Planning +
  Goal Reflection + Evidence Iteration) with explicit evidence logs and quality gates.
- **Unified LLM Routing**: LiteLLM-based backend supporting both cloud APIs
  (OpenAI/Anthropic/Google/Qwen) and local models (Ollama/vLLM).
- **48 Atomic Tools**: Seven categories covering profiling, detection, decomposition,
  relation analysis, prediction, and validation.
- **SafeAuto**: First-order logic and physical constraint verification to intercept
  numerical hallucinations.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run with local Ollama model
export TIMECLAW_CONFIG=configs/default.yaml
python examples/quickstart_local.py

# Run with cloud API
export OPENAI_API_KEY=sk-...
python examples/quickstart.py
```

## Configuration

TimeClaw supports per-agent model assignment. Edit `configs/default.yaml`:

```yaml
llm:
  models:
    manager: "ollama/qwen3:8b"
    analyzer: "ollama/qwen3:1.7b"
    executor: "ollama/qwen3:1.7b"
    reasoner: "ollama/qwen3:8b"
  fallback: "gpt-4o-mini"
```

## Project Structure

```
timeclaw/
├── core/          # Core abstractions (Agent, TEA protocol, Evidence, DAG)
├── llm/           # Unified LLM routing (LiteLLM wrapper)
├── agents/        # Four-layer agents (Manager/Analyzer/Executor/Reasoner)
├── tokencast/     # Signal-native encoding
├── clawhub/       # 48 atomic tools in 7 categories
├── fingerprint/   # Data fingerprint system
├── memory/        # Experience pool and domain knowledge
├── benchmark/     # MTBench evaluation framework
└── pipeline/      # End-to-end runner
```

## License

MIT
