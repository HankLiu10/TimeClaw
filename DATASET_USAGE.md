# 数据集使用说明（训练集 / 测试集）

本文供自动化 agent 与人类开发者快速理解：**自研独立训练集**与 **MTBench 官方测试语料**在仓库中的位置、格式约定，以及 TimeClaw 默认评测代码如何加载它们。

---

## 1. 角色划分（推荐实验设定）

| 角色 | 语料 | 典型用途 |
|------|------|----------|
| **训练 / 探索学习** | **独立数据集**（`data_preparation/independent/`） | 与 MTBench **样本不重叠**（不同行情源、气象站点策略等），用于 `train` 模式、记忆蒸馏、策略学习，降低对官方测试集的过拟合。 |
| **测试 / 报告指标** | **MTBench 官方**（`data/processed/` + `data/splits/`） | 标准复现与横向对比；默认 `run_pipeline_eval(..., split="test")` 即在此语料上评测。 |

说明：二者 **schema 与任务类型对齐**（见 `data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md`），但 **文件路径与样本 ID 不同**。

---

## 2. MTBench 官方数据（测试集来源）

### 2.1 获取方式

在仓库根目录执行：

```bash
python scripts/download_mtbench.py
```

从 HuggingFace 拉取各子集到固定本地目录（映射定义在 `scripts/download_mtbench.py` 的 `DATASETS` 字典中）。

### 2.2 处理后文件布局（相对仓库根）

| 逻辑名 `dataset_name` | 目录 `data/processed/...` |
|------------------------|---------------------------|
| `weather_aligned_short` | `weather/aligned_in7days_out1days/`（JSON） |
| `weather_aligned_long` | `weather/aligned_in14days_out3days/`（JSON） |
| `weather_qa_short` | `weather/QAshort/`（JSON） |
| `weather_qa_long` | `weather/QAlong/`（JSON） |
| `finance_aligned_short` | `finance/aligned_in7days_out1days/`（Parquet） |
| `finance_aligned_long` | `finance/aligned_in30days_out7days/`（Parquet） |
| `finance_qa_short` | `finance/QAshort/`（JSON） |
| `finance_qa_long` | `finance/QAlong/`（JSON） |

### 2.3 训练 / 测试划分（官方语料上的时间切分）

- 列表根目录：`data/splits/<dataset_name>/`
- 文件：`train_files.json`、`test_files.json`、`split_meta.json`
- 若缺失：在仓库根目录执行  
  `python -m timeclaw.benchmark.dataset_split`  
  （会调用 `split_all_datasets()`，按年份将官方全量样本切为约 70–75% train / 余下 test）。

### 2.4 代码中的注册表与加载入口

- **路径与任务列表**：`timeclaw/benchmark/mtbench_adapter.py` 内 `DATASET_REGISTRY`
- **统一样本类型**：`MTBenchSample`
- **加载全量某数据集**：`load_dataset(dataset_name)` — 读取 `data/processed/...` 下全部可解析样本
- **按划分加载**：`timeclaw.benchmark.dataset_split.load_split(dataset_name, split)` — `split` 为 `"train"` 或 `"test"`，在 `load_dataset` 结果上按 `data/splits/...` 中的文件名过滤

### 2.5 默认「测试」评测

- **模块**：`timeclaw/benchmark/pipeline_eval.py` → `run_pipeline_eval(..., split="test", ...)`
- **默认行为**：对每个 `DATASET_REGISTRY` 中的数据集，加载 **MTBench 官方** `test` 半（除非 `full_dataset=True`，此时忽略 split 列表、使用全量官方样本）。

同类入口：`timeclaw/benchmark/baseline_eval.py` 中基线评测亦基于 `load_split(..., "test")` 与官方路径。

---

## 3. 独立训练集（自研语料）

### 3.1 根目录

- 基准目录：`data_preparation/independent/`（常被 `.gitignore` 忽略，需在本地生成）
- 划分输出：`data_preparation/independent/splits/<dataset_name>/`
  - 同样包含 `train_files.json`、`test_files.json`（可为空）、`split_meta.json`
- 原始/处理后样本：
  - 金融：`data_preparation/independent/finance/processed/`（如 `train-00000-of-00001.parquet`）
  - 气象：`data_preparation/independent/weather/processed/*.json`

### 3.2 从零构建

完整流程、依赖与脚本索引见：

- `reports/trainingset_generation/README.md`
- 可选一键脚本：`reports/trainingset_generation/run_from_scratch.sh`

核心脚本目录：`data_preparation/independent/scripts/`（`build_finance.py`、`build_weather.py`、`generate_tasks.py`、`validate.py` 等）。

### 3.3 与默认 benchmark 代码的关系（必读）

当前 **`timeclaw.benchmark` 默认只认两套固定相对路径**：

1. 样本文件：`DATASET_REGISTRY` 中的 `data/processed/...`（**仅 MTBench 官方目录**）
2. 划分列表：`dataset_split.SPLITS_DIR = Path("data/splits")`（**仅官方 split**）

因此：

- **`run_pipeline_eval(split="test")`**：正确使用 **MTBench 测试半** ✓  
- **`run_train_loop()`**：内部调用 `load_split(ds_name, "train")`，加载的是 **`data/splits` + `data/processed` 下的官方 train 半**，**不是** `data_preparation/independent/` ✓/✗（与「用独立集训练」的意图不一致时需注意）

若要在 **不修改核心库** 的前提下用独立集驱动与 `MTBenchSample` 相同的训练逻辑，需要自行实现或扩展加载器（例如：从 `data_preparation/independent/splits` 与对应 `processed` 构造 `MTBenchSample` 列表），或经团队约定后调整 `DATASET_REGISTRY` / `SPLITS_DIR` 指向独立路径（需谨慎，避免与官方评测目录混淆）。

---

## 4. 八个子数据集与任务类型（名称对齐）

以下名称在 **官方**与**独立** split 子目录中 **同名**，便于脚本复用 `dataset_name` 字符串：

| `dataset_name` | 领域 | 常见任务类型（以 `DATASET_REGISTRY` 为准；部分天气 QA 任务在注册表中被注释以匹配子集） |
|----------------|------|----------------------------------------------------------------------------------------|
| `finance_aligned_short` / `finance_aligned_long` | 金融 | `forecast`, `trend`, `indicator` |
| `finance_qa_short` / `finance_qa_long` | 金融 | `mcqa`, `correlation` |
| `weather_aligned_short` / `weather_aligned_long` | 气象 | `forecast`, `trend`, `trend_past`（部分配置下 `indicator` 关闭） |
| `weather_qa_short` / `weather_qa_long` | 气象 | 视注册表与 `expand_dataset_tasks` 展开结果而定 |

任务展开与 scope 键：`timeclaw/benchmark/mtbench_support/`。

---

## 5. 工作目录约定

所有上述 **相对路径**均以 **仓库根目录** 为当前工作目录（`cwd`）。Agent 在子目录执行 Python 时，需 `chdir` 到根目录或传入等价绝对路径逻辑。

---

## 6. 相关文件速查

| 用途 | 路径 |
|------|------|
| 官方下载映射 | `scripts/download_mtbench.py` |
| 样本注册与加载 | `timeclaw/benchmark/mtbench_adapter.py` |
| 官方 train/test 列表生成与加载 | `timeclaw/benchmark/dataset_split.py` |
| 流水线评测（默认 test） | `timeclaw/benchmark/pipeline_eval.py` |
| 探索式训练循环（默认官方 train） | `timeclaw/benchmark/train_loop.py` |
| 独立集构建说明 | `reports/trainingset_generation/README.md` |
| 子任务级对齐说明 | `data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md` |

---

## 7. 英文摘要（for English-only agents）

- **Test / metrics**: Use official MTBench under `data/processed/` with file lists in `data/splits/<name>/`. Download via `python scripts/download_mtbench.py`. Evaluate with `run_pipeline_eval(..., split="test")` (default).
- **Train (disjoint corpus)**: Built under `data_preparation/independent/`; see `reports/trainingset_generation/README.md`. Schema aligns with MTBench, but **default** `load_dataset` / `load_split` in `timeclaw.benchmark` point at **official** paths only—custom wiring is required to train on the independent files while keeping official test for evaluation.
