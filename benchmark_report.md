# TimeClaw vs Baseline — Full Benchmark Report

Generated: 2026-03-16 02:55:20

## Dataset Splits (70/30 by time, no overlap)

| Dataset | Train | Test | Split Year | Train % |
|---------|-------|------|------------|---------|
| finance_aligned_long | 384 | 141 | 2023 | 73% |
| finance_aligned_short | 529 | 221 | 2023 | 71% |
| finance_qa_long | 368 | 148 | 2023 | 71% |
| finance_qa_short | 341 | 150 | 2023 | 69% |
| weather_aligned_long | 1448 | 511 | 2020 | 74% |
| weather_aligned_short | 1445 | 514 | 2020 | 74% |
| weather_qa_long | 537 | 191 | 2020 | 74% |
| weather_qa_short | 468 | 177 | 2020 | 73% |

## Per-Task Comparison

### Overall (aggregated across all datasets)

| Task | Metric | TimeClaw | Baseline | Δ | Winner |
|------|--------|----------|----------|---|--------|
| correlation | accuracy | 0.1443 | 0.1242 | +0.0201 | TimeClaw |
| forecast | mae | 3.7547 | 4.4907 | -0.7360 | TimeClaw |
| forecast | mape | 32.8661 | 38.6622 | -5.7961 | TimeClaw |
| forecast | mse | 23.8835 | 33.2768 | -9.3933 | TimeClaw |
| forecast | rmse | 4.1765 | 5.0405 | -0.8640 | TimeClaw |
| indicator | diff_mae | 5.2142 | 3.3506 | +1.8636 | Baseline |
| indicator | macd_mae | 0.6725 | 0.3934 | +0.2790 | Baseline |
| indicator | max_mae | 4.9988 | 3.3020 | +1.6968 | Baseline |
| indicator | min_mae | 5.5564 | 3.5461 | +2.0103 | Baseline |
| indicator | upper_bb_mae | 2.7375 | 2.5082 | +0.2293 | Baseline |
| mcqa | accuracy | 0.6336 | 0.6862 | -0.0525 | Baseline |
| trend | accuracy | 0.3367 | 0.2862 | +0.0504 | TimeClaw |

### Per-Dataset Breakdown

#### finance_aligned_long

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| forecast | mae | 2.3060 | 3.4876 | TimeClaw |
| forecast | mape | 2.0274 | 3.1220 | TimeClaw |
| forecast | mse | 31.8477 | 55.4210 | TimeClaw |
| forecast | rmse | 2.5579 | 3.9463 | TimeClaw |
| indicator | macd_mae | 1.2374 | 0.6689 | Baseline |
| indicator | upper_bb_mae | 3.4987 | 3.5619 | TimeClaw |
| trend | accuracy | 0.4326 | 0.5248 | Baseline |

#### finance_aligned_short

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| forecast | mae | 1.5464 | 1.7637 | TimeClaw |
| forecast | mape | 2.0760 | 1.8041 | Baseline |
| forecast | mse | 15.5512 | 17.5013 | TimeClaw |
| forecast | rmse | 1.6711 | 1.9168 | TimeClaw |
| indicator | macd_mae | 0.3104 | 0.2169 | Baseline |
| indicator | upper_bb_mae | 2.2496 | 1.8328 | Baseline |
| trend | accuracy | 0.6561 | 0.7557 | Baseline |

#### finance_qa_long

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| correlation | accuracy | 0.1554 | 0.1419 | TimeClaw |
| mcqa | accuracy | 0.7905 | 0.8176 | Baseline |

#### finance_qa_short

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| correlation | accuracy | 0.1333 | 0.1067 | TimeClaw |
| mcqa | accuracy | 0.7867 | 0.8733 | Baseline |

#### weather_aligned_long

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| forecast | mae | 3.3844 | 4.4077 | TimeClaw |
| forecast | mape | 43.5236 | 55.2331 | TimeClaw |
| forecast | mse | 20.9872 | 35.0940 | TimeClaw |
| forecast | rmse | 3.9052 | 5.0883 | TimeClaw |
| indicator | diff_mae | 4.8842 | 3.7760 | Baseline |
| indicator | max_mae | 3.3258 | 3.0918 | Baseline |
| indicator | min_mae | 3.7776 | 4.2404 | TimeClaw |
| trend | accuracy | 0.2661 | 0.1507 | TimeClaw |

#### weather_aligned_short

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| forecast | mae | 3.7120 | 4.1485 | TimeClaw |
| forecast | mape | 39.8813 | 40.4107 | TimeClaw |
| forecast | mse | 27.8343 | 30.4984 | TimeClaw |
| forecast | rmse | 4.1606 | 4.7234 | TimeClaw |
| indicator | diff_mae | 5.4312 | 2.9235 | Baseline |
| indicator | max_mae | 6.0988 | 3.5131 | Baseline |
| indicator | min_mae | 6.7260 | 2.8488 | Baseline |
| trend | accuracy | 0.2432 | 0.1537 | TimeClaw |

#### weather_qa_long

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| forecast | mae | 8.4852 | 9.5577 | TimeClaw |
| forecast | mape | 43.8662 | 58.2335 | TimeClaw |
| forecast | mse | 24.7620 | 38.0061 | TimeClaw |
| forecast | rmse | 9.0391 | 10.2145 | TimeClaw |
| mcqa | accuracy | 0.4974 | 0.5445 | Baseline |

#### weather_qa_short

| Task | Metric | TimeClaw | Baseline | Winner |
|------|--------|----------|----------|--------|
| mcqa | accuracy | 0.5198 | 0.5706 | Baseline |
