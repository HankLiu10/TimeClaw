#!/usr/bin/env python3
"""列出「小规模训练」实际用到的样本：与 train_loop 一致，即 train split 顺序下的前 N 条。

`run_train_loop(max_samples=100)` 等价于：
  samples = load_split(ds_name, "train")
  samples = samples[:100]

本脚本不写库代码，只读 split 并打印或导出清单，便于你对照数据文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY


def _sample_record(idx: int, s: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "train_split_index": idx,
        "file": s.file,
        "dataset": s.dataset,
        "domain": s.domain,
        "start_year": s.start_year,
    }
    if s.metadata:
        rec["metadata"] = s.metadata
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description="列出 train split 前 N 条（与 train_loop 一致）")
    p.add_argument("--max-samples", type=int, default=100, help="与 train_loop 的 max_samples 一致")
    p.add_argument("--dataset", type=str, default=None, help="只列某一数据集；默认全部 DATASET_REGISTRY")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="写入 JSON；默认只打印摘要（可用 data/train_first100_manifest.json）",
    )
    p.add_argument("--verbose", action="store_true", help="打印每条 file 与下标")
    args = p.parse_args()

    names = [args.dataset] if args.dataset else list(DATASET_REGISTRY.keys())
    manifest: dict[str, list[dict[str, Any]]] = {}

    for ds in names:
        if ds not in DATASET_REGISTRY:
            print(f"未知数据集: {ds}")
            continue
        try:
            train = load_split(ds, "train")
        except FileNotFoundError:
            print(f"{ds}: 无 train split")
            continue
        n = min(args.max_samples, len(train))
        slice_ = train[:n]
        manifest[ds] = [_sample_record(i, s) for i, s in enumerate(slice_)]
        print(f"{ds}: train 共 {len(train)} 条，取前 {n} 条（下标 0..{n - 1}）")
        if args.verbose:
            for r in manifest[ds]:
                print(f"  [{r['train_split_index']}] {r['file']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n已写入: {args.out}")


if __name__ == "__main__":
    main()
