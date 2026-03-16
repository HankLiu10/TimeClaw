# Baseline vs TimeClaw vs LLM-Reasoning Quick Comparison

Samples per task: 3
Baseline time: 56.6s
LLM-Reasoning time: 38.5s
TimeClaw time: 168.0s


## weather_aligned_short
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| forecast | mae | 5.1424 | 3.7618 | 3.5216 | TimeClaw |
| forecast | mape | 40.4917 | 29.5361 | 20.6589 | TimeClaw |
| forecast | mse | 37.9151 | 20.6905 | 22.6091 | LLM-R |
| forecast | rmse | 5.9750 | 4.3951 | 4.5321 | LLM-R |
| trend | accuracy | 0.3333 | 0.6667 | 0.0000 | LLM-R |
| indicator | diff_mae | 1.2667 | 1.2667 | 5.6877 | Baseline |
| indicator | max_mae | 1.8333 | 1.9000 | 5.3429 | Baseline |
| indicator | min_mae | 2.2333 | 1.9667 | 4.0781 | LLM-R |

## weather_aligned_long
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| forecast | mae | 5.3172 | 4.2129 | 23.5319 | LLM-R |
| forecast | mape | 37.7697 | 27.3753 | 171.9046 | LLM-R |
| forecast | mse | 34.9224 | 28.3917 | 775.0954 | LLM-R |
| forecast | rmse | 5.9068 | 5.2401 | 26.1642 | LLM-R |
| trend | accuracy | 0.3333 | 0.3333 | 0.0000 | Baseline |
| indicator | diff_mae | 4.8667 | 2.4000 | 22.6156 | LLM-R |
| indicator | max_mae | 4.3333 | 1.6333 | 34.6033 | LLM-R |
| indicator | min_mae | 3.0667 | 2.2333 | 11.9878 | LLM-R |

## weather_qa_short
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| mcqa | accuracy | 0.6667 | 1.0000 | 0.6667 | LLM-R |

## weather_qa_long
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| forecast | mae | 3.2904 | 5.8466 | 30.3379 | Baseline |
| forecast | mape | 20.0908 | 32.1177 | 168.5403 | Baseline |
| forecast | mse | 18.3504 | 48.5952 | 1163.2594 | Baseline |
| forecast | rmse | 4.0372 | 6.8945 | 33.8779 | Baseline |
| mcqa | accuracy | 0.3333 | 0.3333 | 0.3333 | TimeClaw |

## finance_aligned_short
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| forecast | mae | 0.0689 | 0.1616 | 0.2946 | Baseline |
| forecast | mape | 0.8155 | 1.6207 | 3.0056 | Baseline |
| forecast | mse | 0.0096 | 0.0503 | 0.2396 | Baseline |
| forecast | rmse | 0.0835 | 0.1907 | 0.3432 | Baseline |
| trend | accuracy | 0.3333 | 0.3333 | 0.6667 | TimeClaw |
| indicator | macd_mae | 0.1103 | N/A | 0.1101 | TimeClaw |
| indicator | upper_bb_mae | 0.6245 | N/A | 0.6508 | Baseline |

## finance_aligned_long
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| forecast | mae | 2.4093 | 1.0713 | 2.8648 | LLM-R |
| forecast | mape | 3.6973 | 1.9686 | 5.2148 | LLM-R |
| forecast | mse | 10.0225 | 2.8168 | 13.8300 | LLM-R |
| forecast | rmse | 2.6753 | 1.3029 | 3.0816 | LLM-R |
| trend | accuracy | 0.6667 | 1.0000 | 0.6667 | LLM-R |
| indicator | macd_mae | 0.3887 | N/A | 0.1972 | TimeClaw |
| indicator | upper_bb_mae | 8.3146 | N/A | 2.4312 | TimeClaw |

## finance_qa_short
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| mcqa | accuracy | 0.6667 | 0.6667 | 1.0000 | TimeClaw |
| correlation | accuracy | 0.0000 | N/A | 0.0000 | TimeClaw |

## finance_qa_long
| Task | Metric | Baseline | LLM-Reasoning | TimeClaw | Best |
|------|--------|----------|---------------|----------|------|
| mcqa | accuracy | 1.0000 | 1.0000 | 1.0000 | TimeClaw |
| correlation | accuracy | 0.3333 | N/A | 0.6667 | TimeClaw |