#!/usr/bin/env bash
# 从零构建 data_preparation/independent 下 MTBench 兼容训练集（仓库根目录执行）。
# 用法：
#   ./reports/trainingset_generation/run_from_scratch.sh
# 环境变量（可选）：
#   SKIP_FINANCE_DOWNLOAD=1   — 等价 build_finance --skip_download
#   SKIP_WEATHER_DOWNLOAD=1   — 等价 build_weather --skip_download
#   SKIP_RSS_FULLTEXT=1       — 不推荐：仅 RSS 标题，不利 MTBench 正文对齐
#   SKIP_QA=1                 — 仅 PART1 aligned，不调用 LLM
#   FORCE_ALIGNED=1           — 强制重建四份 aligned
#   FINANCE_QA_OVERSAMPLE=18  — 传给 generate_tasks --finance-qa-oversample
#   CONCURRENCY=50
#   CONFIG=configs/claude_proxy.yaml  — TimeClaw YAML（api_base/api_key）

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BASE="data_preparation/independent"
PY="${PYTHON:-python3}"
SCRIPTS="$BASE/scripts"

echo "==> Repo root: $ROOT"
echo "==> Base dir:  $BASE"

FIN_ARGS=(--base_dir "$BASE")
[[ "${SKIP_FINANCE_DOWNLOAD:-}" == "1" ]] && FIN_ARGS+=(--skip_download)
[[ "${SKIP_RSS_FULLTEXT:-}" == "1" ]] && FIN_ARGS+=(--skip_rss_fulltext)

echo ""
echo "=== [1/4] build_finance.py ==="
"$PY" "$SCRIPTS/build_finance.py" "${FIN_ARGS[@]}"

WX_ARGS=(--base_dir "$BASE")
[[ "${SKIP_WEATHER_DOWNLOAD:-}" == "1" ]] && WX_ARGS+=(--skip_download)

echo ""
echo "=== [2/4] build_weather.py ==="
"$PY" "$SCRIPTS/build_weather.py" "${WX_ARGS[@]}"

GT_ARGS=(--base_dir "$BASE" --concurrency "${CONCURRENCY:-50}")
[[ "${SKIP_QA:-}" == "1" ]] && GT_ARGS+=(--skip_qa)
[[ "${FORCE_ALIGNED:-}" == "1" ]] && GT_ARGS+=(--force_aligned)
[[ -n "${CONFIG:-}" ]] && GT_ARGS+=(--config "$CONFIG")
[[ -n "${FINANCE_QA_OVERSAMPLE:-}" ]] && GT_ARGS+=(--finance-qa-oversample "$FINANCE_QA_OVERSAMPLE")

echo ""
echo "=== [3/4] generate_tasks.py ==="
echo "    args: ${GT_ARGS[*]}"
"$PY" "$SCRIPTS/generate_tasks.py" "${GT_ARGS[@]}"

echo ""
echo "=== [4/4] validate.py ==="
"$PY" "$SCRIPTS/validate.py" --base_dir "$BASE"

echo ""
echo "ALL STEPS DONE. Outputs under $BASE/splits/"
