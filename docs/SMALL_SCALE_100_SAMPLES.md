# 小规模实验：每数据集 100 个训练样本 + 100 个测试样本（做法说明）

本文档根据会话 [基准训练与实验监控](d262efb1-cc92-48fe-99ac-a086cbb5beca) 与后续实现讨论整理，用于在你误删编排脚本后**复现同一套协议**。

---

## 1. 目标

- **训练**：8 个 MTBench 数据集，每个数据集在 **train split** 上只用 **100 条样本**（控制成本）。
- **测试**：8 个数据集在 **test split** 上只用 **100 条样本**。
- **公平对比**：Baseline（无工具直出）与 TimeClaw（带记忆）必须在**同一批测试样本**上评估，因此需要**固定测试索引**，而不是「各跑 test 集前 100 条但顺序/抽样不一致」。

---

## 2. 测试集：固定 100 条索引（`data/test_indices_100.json`）

- **文件路径**：`data/test_indices_100.json`
- **格式**：`{ "dataset_name": [i0, i1, ...], ... }`，每个 key 对应 `load_split(dataset, "test")` 返回列表中的 **下标**，长度一般为 100（若全量不足 100 则为全量下标）。
- **生成逻辑**（原 `run_full_experiment.py` 中的 `freeze_test_samples()`）：
  - 若文件**已存在**：直接 `json.load`，保证多次实验**同一批题**。
  - 若**不存在**：对每个数据集从 test split 用 **`_stratified_sample`**（与 `pipeline_eval` 中一致的分层抽样）选出 100 个样本，再反推出对应的 **原始下标列表**，写入 JSON。

仓库中若仍保留该 JSON，可直接复用；若删了，需用同样逻辑**重新生成**一次（否则与历史结果不可比）。

---

## 3. 训练集：100 条样本（**train 前 100 条**）

- **调用**：`timeclaw.benchmark.train_loop.run_train_loop(max_samples=100, ...)`
- **代码语义**（`train_loop.py`）：`load_split(ds, "train")` 后对 `samples = samples[:max_samples]`，即按当前 split 的**顺序取前 100 条**，不是随机 100 条。
- **并发**：历史上多次调整；常见配置包括「全局 semaphore + 多数据集并行、每 task 限流」等，以你当时 `run_full_experiment.py` 或 `train_loop` 参数为准。

训练结束后通常会调用 **`HierarchicalMemory.force_distill()`**（或等价「把 L1 笔记蒸馏进 L2/L3」），再进入测试阶段。

---

## 4. 测试阶段：Baseline + Trained

### Baseline

- `timeclaw.benchmark.baseline_eval.run_baseline(...)`
- 需传入与 TimeClaw **相同的 100 个 test 样本**。  
  **完整做法**：支持按 `sample_indices`（或等价）只对 `indices` 里的行跑 baseline。  
  **简化做法**（当前部分代码）：`max_samples=100` → 使用 **test split 的前 100 条**，与「JSON 固定子集」**不一定一致**。

### Trained（带记忆）

- `timeclaw.benchmark.pipeline_eval.run_pipeline_eval(split="test", label="trained", use_skills=True, ...)`
- **完整做法**：若实现 **`sample_indices: dict[str, list[int]]`**，则对每个 `ds` 执行  
  `samples = [samples[i] for i in idx_list if i < len(samples)]`，与 baseline 完全一致。  
- **简化做法**：仅 `max_samples=100` → test **前 100 条**，与 JSON 子集可能不一致。

---

## 5. 原编排脚本：`scripts/run_full_experiment.py`（你已删除）

典型流程（与 transcript / 后续会话一致）：

1. `freeze_test_samples()` → 读或写 `data/test_indices_100.json`
2. **P1** `phase1_baseline(test_indices)`：全 8 集 baseline，带 `sample_indices`
3. **P3** `phase3_train()`：`run_train_loop(max_samples=100, concurrency=..., per_task_concurrency=..., indicator_concurrency=...)`  
4. 训练结束：`HierarchicalMemory(...).force_distill()`
5. **P4** `phase4_test(test_indices)`：`run_pipeline_eval(..., sample_indices=test_indices, label="trained", split="test", ...)`
6. `compare()`：读 `.timeclaw/test_results/baseline_*.json` 与 `.timeclaw/eval_results/trained_*.json`，按 task 聚合指标对比

常数示例（以最后一次会话为准，可按需改）：

- `TRAIN_MAX_SAMPLES = 100`
- `TEST_MAX_SAMPLES = 100`（与 JSON 中每集长度一致）
- `DATASETS`：8 个 `weather_*` / `finance_*` 数据集名（与 `DATASET_REGISTRY` 一致）

---

## 6. 当前仓库状态（编写此文档时）

- `scripts/run_full_experiment.py`：**不在仓库中**（需从 git 历史恢复或按上文重写）。
- `timeclaw/benchmark/pipeline_eval.py`：若**无** `sample_indices` 参数，则只能用 `max_samples=100` 做**近似**小规模实验，**无法**与 `test_indices_100.json` 严格对齐。
- `data/test_indices_100.json`：若仍存在，可用于恢复「严格 100 子集」评估（需在 `pipeline_eval` / `baseline_eval` 中接回按索引取样本）。

---

## 7. 最小复现命令（当前 API，**不**保证与 JSON 子集一致）

在项目根目录：

```bash
# 训练：每数据集 train 前 100 条
python -c "
import asyncio
from timeclaw.benchmark.train_loop import run_train_loop
asyncio.run(run_train_loop(max_samples=100, concurrency=8))
"

# 测试（带记忆）：test 前 100 条
python -c "
import asyncio
from timeclaw.benchmark.pipeline_eval import run_pipeline_eval
asyncio.run(run_pipeline_eval(max_samples=100, split='test', label='trained', concurrency=20))
"

# Baseline：test 前 100 条
python -c "
import asyncio
from timeclaw.benchmark.baseline_eval import run_baseline
asyncio.run(run_baseline(max_samples=100))
"
```

若要**严格复现**「与历史报告可比的 100 测试子集」，必须恢复 **`sample_indices` + 编排脚本**（或从 `git checkout` / 备份取回旧版 `run_full_experiment.py` 与对应 `pipeline_eval`）。

---

## 8. Transcript 里相关背景（d262）

- 曾从 **10 sample / 全量** 逐步改到 **每任务 100 train + 固定 100 test 与 baseline 对齐**。
- 强调 **train / test 目标一致**、**GT 不泄漏进蒸馏统计**、**评判工具不进 memory** 等（与 `hierarchical.py` 修改配套）。
- 具体并发数字（60/80/100、per-task 等）以当时脚本为准；本文件只固定 **样本数协议**。

---

*若你希望把 `run_full_experiment.py` 与 `sample_indices` 重新加回仓库，可在新会话里说明，按当前 `train_loop` / `pipeline_eval` 签名做最小补丁。*
