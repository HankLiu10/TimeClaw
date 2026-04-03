# 独立训练集（MTBench 兼容）从零构建指南

本文说明如何在 **TimeClaw 仓库内**，从零生成 `data_preparation/independent/splits/`（历史目录名，存放 **独立语料的样本文件名列表**，与「把 MTBench 一切两半」无关）下 **8 个**与 MTBench **数据契约**对齐的任务数据集，并列出所依赖的 **代码文件** 与 **官方参考数据路径**。

---

## 1. 产出物一览

| 目录（相对 `data_preparation/independent/splits/`，独立侧 **样本列表** 根目录） | 类型 | 目标规模（脚本常量，以实际运行为准） |
|-----------------------------------------------------|------|----------------------------------------|
| `finance_aligned_short/` | Parquet + 列表文件 | 530 |
| `finance_aligned_long/` | Parquet + 列表文件 | 至多 5000（受 1h/5m 覆盖与 MTBench 文本闭区间筛选限制，常为千余级） |
| `weather_aligned_short/` | JSON + 列表 | 与 `weather/processed` 过滤后一致 |
| `weather_aligned_long/` | JSON + 列表 | 能对齐 ISD 重切的条数 |
| `finance_qa_short/` | JSON（MCQA + 五档 correlation） | 350 |
| `finance_qa_long/` | JSON | 350 |
| `weather_qa_short/` | JSON（MCQA） | 400 |
| `weather_qa_long/` | JSON | 400（偶发缺 1 条时需补跑） |

每个数据集含 `train_files.json`、`test_files.json`（可为空）、`split_meta.json`。金融 QA 的 `split_meta.correlation_rebalance` 记录五档 rebalance 是否完整、是否 padding。

---

## 2. 前置条件

### 2.1 仓库与官方参考数据

本流水线在 **统计与筛选** 上会读取仓库内 **MTBench 已处理数据** 与 **`data/splits/` 下的官方文件名列表**（将同一数据集的 `train_files.json` 与 `test_files.json` **合并去重**，仅作 **分布/闭区间统计的参考全集**，不是宣称「官方 train 用来训练本流水线」）：

- `data/processed/finance/`（aligned parquet、QA 子目录等）
- `data/processed/weather/`（aligned、QA 子目录）
- `data/splits/finance_aligned_*`、`finance_qa_*`、`weather_*` 下的 `train_files.json` / `test_files.json`

**若缺失**：部分对齐会退化为启发式默认（脚本内已有 fallback），不建议用于「严格复现 MTBench 对齐」。

### 2.2 Python 与依赖

在仓库根目录执行，需安装 `data_preparation/independent` 流水线所用依赖（含 `pandas`、`pyarrow`、`numpy`、`feedparser`、`yfinance`、`newspaper3k` 等，与 `build_finance.py` / `build_weather.py` / `generate_tasks.py` 一致）。建议在独立 venv 中安装。

### 2.3 LLM（仅生成 QA 与 PART3）

- 环境变量：`OPENAI_API_KEY`（可逗号分隔多 key）；或 `configs/claude_proxy.yaml`（或 `TIMECLAW_CONFIG`）中 `llm.api_base` / `llm.api_key`。
- **仅生成 aligned、不调用 API**：`generate_tasks.py --skip_qa`。

### 2.4 磁盘与网络

- 金融：下载行情 Parquet、RSS、可选全文抓取，体积较大。
- 天气：NOAA ISD、Storm Events 等下载。
- QA：金融每条样本一次（或多次重试）流式调用，**耗时可达数小时**；`FINANCE_QA_GENERATION_OVERSAMPLE` 越大调用越多。

---

## 3. 端到端流程（推荐顺序）

以下路径均以 **仓库根目录** 为当前工作目录，`BASE=data_preparation/independent`。

### 阶段 A：金融原始对与 Parquet

**脚本**：[`data_preparation/independent/scripts/build_finance.py`](../../data_preparation/independent/scripts/build_finance.py)

**作用**：yfinance 拉取 5m（及 1h）、Yahoo RSS、**newspaper3k 全文**（默认），构造 7d/1d 窗内配对，写出：

- `data_preparation/independent/finance/processed/train-00000-of-00001.parquet`
- `data_preparation/independent/finance/raw/` 下缓存与行情文件

**重要**：与 MTBench **正文长度可比**时，**不要**使用 `--skip_rss_fulltext`（否则大量仅标题，后续即使用 MTBench 闭区间过滤也会损失池子）。详见独立目录 [`README.md`](../../data_preparation/independent/README.md)。

示例：

```bash
python3 data_preparation/independent/scripts/build_finance.py --base_dir data_preparation/independent
# 已有一次下载时：
# python3 data_preparation/independent/scripts/build_finance.py --base_dir data_preparation/independent --skip_download
```

### 阶段 B：气象原始 JSON

**脚本**：[`data_preparation/independent/scripts/build_weather.py`](../../data_preparation/independent/scripts/build_weather.py)

**作用**：ISD-Lite + Storm Events（站点与 MTBench WBAN 无交集策略在脚本内），写出：

- `data_preparation/independent/weather/processed/*.json`

```bash
python3 data_preparation/independent/scripts/build_weather.py --base_dir data_preparation/independent
```

### 阶段 C：八数据集生成（aligned + 可选 QA）

**脚本**：[`data_preparation/independent/scripts/generate_tasks.py`](../../data_preparation/independent/scripts/generate_tasks.py)

**PART 1**：从金融 Parquet 与气象 JSON 抽样/重切，写入独立侧 `splits/` 目录下四个 **aligned** 数据集。  
**PART 2**：金融/天气 QA，**并发 LLM**，支持断点续跑（同 stem 跳过）。  
**PART 3**：MCQA **选项 A–D 均匀化 shuffle**（**不**改写 `news_price_correlation`）。

常用命令：

```bash
# 仅重建 aligned（无 API）
python3 data_preparation/independent/scripts/generate_tasks.py \
  --base_dir data_preparation/independent \
  --force_aligned \
  --skip_qa

# 完整跑（含 QA + shuffle），aligned 已存在时可去掉 --force_aligned
python3 data_preparation/independent/scripts/generate_tasks.py \
  --base_dir data_preparation/independent \
  --concurrency 50

# 提高金融 QA 生成池（更多 LLM 调用，利于五档凑满）
python3 data_preparation/independent/scripts/generate_tasks.py \
  --base_dir data_preparation/independent \
  --finance-qa-oversample 18
```

**配置 API**：`--api_key`、`--config`，或环境变量（与脚本内 `apply_timeclaw_yaml_llm_defaults` 一致）。

### 阶段 D：校验

**脚本**：[`data_preparation/independent/scripts/validate.py`](../../data_preparation/independent/scripts/validate.py)

**作用**：与 `data/processed` 的 URL / 站点隔离、schema 抽检等。

```bash
python3 data_preparation/independent/scripts/validate.py --base_dir data_preparation/independent
```

### 一键串联

**脚本**：[`data_preparation/independent/scripts/run_all.py`](../../data_preparation/independent/scripts/run_all.py)

等价于依次执行 `build_finance.py` → `build_weather.py` → `generate_tasks.py`，参数可转发 `--skip_download`、`--skip_qa`、`--force_aligned`、`--finance-qa-oversample`、`--config`、`--api_key` 等。

本目录附带 **[`run_from_scratch.sh`](./run_from_scratch.sh)**：在仓库根目录执行、分步打印命令（可改环境变量控制是否跳过下载/QA）。

---

## 4. 「对齐」策略摘要（与实现位置）

更细的 **23 scope 子任务** 对照见：[`data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md`](../../data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md)。

| 维度 | 做法 | 主要代码位置 |
|------|------|----------------|
| 金融 aligned `(n_in,n_out)` | 软匹配 MTBench 联合直方图；短窗 `pick_rows_soft_mtbench_io_match` | `build_finance.py`、`generate_tasks.build_finance_aligned_short` |
| 金融 aligned 文本 | `content` 长度闭区间（官方 aligned parquet min–max）+ 向官方中位数偏好排序 | `generate_tasks.py`：`mtbench_finance_aligned_parquet_content_bounds`、`_filter_finance_df_by_mtbench_content_bounds` |
| 金融 QA IO 长度 | 按官方 QA JSON 列表统计的直方图分配 | `build_finance.py`：`load_mtbench_finance_io_from_qa_json_split` 等 |
| 金融 QA 五档 | 规则预分层 → LLM 标注 → `rebalance_finance_qa_correlation` 按 MTBench 边际压到 350；**五档优先于文本偏好** | `generate_tasks.py`：`sample_finance_qa_rows_stratified`、`rebalance_finance_qa_correlation`、`_finance_qa_greedy_fill_*` |
| 金融 QA 文本 | 官方 **finance_qa** `len(text)` 闭区间预筛；档内/缺口层内向全局中位数偏好 | `mtbench_finance_qa_text_bounds`、`_finance_qa_text_pref_effective` |
| 气象 past-trend 三档 | 与官方合并列表边际一致的分层抽样 | `select_weather_samples_by_past_label_proportions`、`load_mtbench_weather_past_label_proportions` |
| 气象文本 | 官方 aligned / QA 的 `len(text)` 闭区间 + 中位数偏好 | `mtbench_weather_*_text_bounds`、`_filter_weather_samples_by_mtbench_text_bounds` |
| MCQA 选项 | 均匀 A–D shuffle | `shuffle_all_qa`、`shuffle_qa_file` |

**性能说明**：曾对 `_finance_qa_greedy_fill_rows` 做 **分桶 + 指针** 优化，避免大池子（如 oversample×350）下 **O(池²×重复算相关)** 的「假死」CPU 占满。

---

## 5. 关键常量（以源码为准）

定义于 `generate_tasks.py` 顶部附近：

- `TARGET_FINANCE_ALIGNED_SHORT = 530`
- `TARGET_FINANCE_ALIGNED_LONG = 5000`（实际写出条数受行情与筛选约束）
- `TARGET_FINANCE_QA_SHORT/LONG = 350`
- `FINANCE_QA_GENERATION_OVERSAMPLE = 15.0`（生成池约 `ceil(350 × factor)`）
- `TARGET_WEATHER_QA_SHORT/LONG = 400`
- `SEED = 42`

---

## 6. 从零「干净」重跑 QA 时

若希望 **不 resume** 旧 JSON，需先删除对应目录下除目录本身外的样本文件，例如：

```bash
rm -f data_preparation/independent/splits/finance_qa_short/*.json
rm -f data_preparation/independent/splits/finance_qa_long/*.json
# 天气 QA 同理
```

再运行 `generate_tasks.py`（无需 `--force_aligned` 若 aligned 未改）。

---

## 7. 已知现象与建议

| 现象 | 说明 |
|------|------|
| 五档 `incomplete` / `padded` | LLM 打标后 **Strong Positive** 等档池子不足，rebalance 用其他档凑满 350；可提高 `--finance-qa-oversample` 或改进提示/解析 |
| 文本均值仍低于官方 MTBench | 官方有极长 tail，均值被拉高；中位数差距更宜作对比；源文须足够长 |
| `weather_qa_long` 399 条 | 单次失败未补；可接受或仅补跑 long |
| 日志长时间停在「stratifying」后 | 大池子分层+贪心已优化；若仍慢，检查是否旧版脚本 |

---

## 8. 相关代码文件索引

| 文件 | 角色 |
|------|------|
| [`data_preparation/independent/scripts/build_finance.py`](../../data_preparation/independent/scripts/build_finance.py) | 金融下载、配对、Parquet |
| [`data_preparation/independent/scripts/build_weather.py`](../../data_preparation/independent/scripts/build_weather.py) | 气象下载与 processed JSON |
| [`data_preparation/independent/scripts/generate_tasks.py`](../../data_preparation/independent/scripts/generate_tasks.py) | 八数据集生成、LLM、shuffle |
| [`data_preparation/independent/scripts/run_all.py`](../../data_preparation/independent/scripts/run_all.py) | 一键串联 |
| [`data_preparation/independent/scripts/validate.py`](../../data_preparation/independent/scripts/validate.py) | 与官方隔离及 schema 校验 |
| [`data_preparation/independent/scripts/compare_finance_text_stats.py`](../../data_preparation/independent/scripts/compare_finance_text_stats.py) | 正文长度自检 |
| [`data_preparation/independent/scripts/analyze_mtbench_vs_independent.py`](../../data_preparation/independent/scripts/analyze_mtbench_vs_independent.py) | 可选对比分析 |
| [`data_preparation/independent/README.md`](../../data_preparation/independent/README.md) | 独立目录总览 |
| [`data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md`](../../data_preparation/independent/MTBENCH_TASK_ALIGNMENT.md) | 子任务级对齐说明 |

---

## 9. 文档与脚本维护位置

- **本文档**：`reports/trainingset_generation/README.md`
- **可执行从零脚本**：`reports/trainingset_generation/run_from_scratch.sh`

若流水线常量或行为变更，请同步更新 **本节与 `generate_tasks.py` 注释**，并视情况更新独立目录 `README.md`。
