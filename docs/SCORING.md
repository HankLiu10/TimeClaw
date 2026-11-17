# Main experiment scoring

The agent and the direct-LLM baseline share
`benchmark/mtbench_support/task_scoring.py`. Supervision is used by the training
evaluator or by offline scoring after inference; it is not an inference input.
Raw predictions, failed outcomes and per-row metrics remain available.

## Numerical rules

Predictions are linearly interpolated to the known target length using NumPy,
as in the pinned MTBench scripts. One predicted value is repeated across the
public horizon. Empty/unparseable predictions are errors, not guessed forecasts.
MAE, MSE and RMSE retain full floating-point precision until summary display.
Rounding before a threshold test would change admission near that threshold.

For benchmark MAPE, only nonzero targets enter the percentage-error mean,
matching upstream `evaluation/utils.py:calculate_mape`. An all-zero target has
undefined MAPE: its field is null, with `mape_defined: false`, while defined
MAE/MSE/RMSE remain. This is not assigned zero error. The generic library's
standalone epsilon-based `metrics.mape` is not the benchmark scorer.

The inherited numeric-target normalizer can interpolate NaN gaps with at least
two finite target points; empty, infinite and insufficiently supported targets
are invalid. That normalization is a disclosed maintenance difference from the
upstream scripts, not evidence that all pathological inputs yield identical
historical scores. Never alter test targets to tune a prediction policy.

## Two explicitly labeled summaries

The upstream scripts exclude extreme numerical errors from cumulative means:

| Domain and task | Excluded from upstream-style mean |
|---|---|
| Finance forecast | MSE > 100 |
| Weather forecast | MSE > 100 |
| Finance MACD | MACD MSE > 10 |

These constants come from the pinned benchmark, not learned memory. The release
keeps that reporting view and also emits an **all-valid** companion: every
completed, scoreable prediction contributes even if its error exceeds the
threshold. Both views preserve total, completed, failed/invalid and included/
filtered counts. Failed predictions have no invented numerical error.

Inference uses `scopes` for the upstream-filtered task view and `all_valid` for
the companion. Direct baselines use `official_filtered` and `all_valid` inside
their summaries. These views are produced by the same shared summary owner.
A mean over valid predictions is not a failure-inclusive success rate; always
report its included count and execution coverage.

## Labels and the 30 paper metric cells

Finance trend/correlation report fine 5-way and coarse 3-way accuracy. MCQA and
weather trends report accuracy. MCQA/trend parsing failures remain errors.
Unrecognized correlation predictions receive incorrect fine/coarse scores,
without guessing a canonical label. These decoder rules can differ from
permissive upstream scripts. For the default `paper30_original` view:

| Task family (short/long except where stated) | Primary metrics per setting |
|---|---|
| Finance forecast | MAE, MAPE |
| Finance MACD | MSE (`macd_mse`) |
| Finance trend | accuracy, accuracy_3way |
| Finance correlation | exact_accuracy, brief_accuracy |
| Finance MCQA | accuracy |
| Weather forecast | MSE, MAE |
| Weather indicators, short | max_mse, max_mae, min_mse, min_mae, diff_mse, diff_mae |
| Weather future trend, short | accuracy |
| Weather past trend, short | accuracy |
| Weather MCQA | accuracy |

This is 17 tasks and 30 primary metric cells. The explicit `timeclaw18` extension
omits the six indicator cells and adds two long trend cells (18 tasks / 26 cells).
Indicator MSE/MAE is computed before display rounding, using scalar squared/absolute
error; summaries average each field across included samples. Auxiliary errors do not create
extra tasks or paper columns. All displayed metrics come from saved predictions;
there is no score-driven per-sample choice between model runs.

## Upstream weather-long ambiguity

The pinned weather future-trend prompt says to compare the last observed day
with the **first predicted day**. The upstream script instead passes the entire
output window to `compute_temperature_trend`; these differ for a three-day
forecast. The release retains TimeClaw's prompt-defined first-24-hour label.
For example, with a last observed mean of 20, a first future day of 21 and later
days of 18, the prompt-defined trend is increasing; the all-three-day average
can be decreasing. Do not claim exact upstream-script score equivalence for
this extension-only long-trend setting. The paper short trend uses a 24-hour future
window, so this counterexample does not establish a short-task mismatch. Resolve the evaluation protocol explicitly before reproducing
or comparing historical tables; no label was chosen based on model outcomes.
