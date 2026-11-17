# Paper task and metric inventory

The single owner is `timeclaw/benchmark/task_registry.py`. The default
`paper30_original` matches the public v1 main tables: 17 tasks / 30 metric cells.
A cell is one reported metric for a task. Ground truth is used only by the training
evaluator or by the external test scorer, never as an inference input.

| Task key | Dataset | Task | Primary metrics |
|---|---|---|---|
| `finance_pred_time_short` | `finance_aligned_short` | `forecast` | mae, mape |
| `finance_pred_time_long` | `finance_aligned_long` | `forecast` | mae, mape |
| `finance_pred_macd_short` | `finance_aligned_short` | `indicator_macd` | macd_mse |
| `finance_pred_macd_long` | `finance_aligned_long` | `indicator_macd` | macd_mse |
| `finance_trend_short` | `finance_aligned_short` | `trend` | accuracy, accuracy_3way |
| `finance_trend_long` | `finance_aligned_long` | `trend` | accuracy, accuracy_3way |
| `finance_correlation_short` | `finance_qa_short` | `correlation` | exact_accuracy, brief_accuracy |
| `finance_correlation_long` | `finance_qa_long` | `correlation` | exact_accuracy, brief_accuracy |
| `finance_mcqa_short` | `finance_qa_short` | `mcqa` | accuracy |
| `finance_mcqa_long` | `finance_qa_long` | `mcqa` | accuracy |
| `weather_forecast_short` | `weather_aligned_short` | `forecast` | mse, mae |
| `weather_forecast_long` | `weather_aligned_long` | `forecast` | mse, mae |
| `weather_trend_future_short` | `weather_aligned_short` | `trend` | accuracy |
| `weather_trend_past_short` | `weather_aligned_short` | `trend_past` | accuracy |
| `weather_mcqa_short` | `weather_qa_short` | `mcqa` | accuracy |
| `weather_mcqa_long` | `weather_qa_long` | `mcqa` | accuracy |
| `weather_indicator_short` | `weather_aligned_short` | `indicator` | max_mse, max_mae, min_mse, min_mae, diff_mse, diff_mae |

Weather indicators predict the highest temperature, lowest temperature and their
difference over the first 24 future hours. Each has MSE and MAE (six cells).
The native tools and direct baseline use this same target and scorer.

## Explicit extension

`timeclaw18` retains the previous 18-task / 26-cell software inventory. It removes
`weather_indicator_short` and adds `weather_trend_future_long` and
`weather_trend_past_long`. It is not the paper inventory. Finance Bollinger-band
prediction remains an internal tool capability rather than a benchmark task.

Set `benchmark.task_set: paper30_original` (default) or `timeclaw18` for learning
and direct baselines. Native benchmark inference uses that config unless an explicit
`--task-set` is supplied. Source lineage and output identities reject cross-preset reuse.

Paper roster SHA-256: `4f37fb76974ce4dbe7760e6cd8d11a922c668bf953f0daf705fd136d382213eb`.
Extension roster SHA-256: `fa83a9e3cd26f7f2e81fd4bfe68b84cf88cd21feda0c2a90bd8351147f1c7dae`.
