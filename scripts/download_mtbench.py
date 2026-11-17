"""Download explicitly pinned external MTBench datasets, separately from training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

LOCK = Path(__file__).resolve().parents[1] / "configs/datasets.lock.json"
DATASETS = json.loads(LOCK.read_text())


def _files(directory):
    return sorted(p for p in directory.rglob("*") if p.is_file() and ".cache" not in p.parts)


def download_all(skip_existing=True, *, datasets=None, repo_root=Path(".")):
    # skip_existing means verified reuse, never a heuristic based on file count.
    names = list(datasets) if datasets is not None else list(DATASETS)
    root = Path(repo_root)
    for name in names:
        info = DATASETS[name]
        local = root / info["local"]
        receipt_path = root / "data/download_receipts" / (name + ".json")
        existing = _files(local) if local.exists() else []
        if existing:
            if not receipt_path.is_file():
                raise ValueError(f"Existing data has no pinned download receipt: {local}. Use a new --root.")
            receipt = json.loads(receipt_path.read_text())
            actual = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest() for p in existing}
            if receipt.get("revision") != info["revision"] or receipt.get("files") != actual:
                raise ValueError(f"Dataset differs from its pinned receipt: {local}. Use a new --root.")
            if skip_existing:
                print(f"{name}: verified {len(actual)} files at {info['revision']}")
                continue
        snapshot_download(repo_id=info["repo"], repo_type="dataset", revision=info["revision"], local_dir=str(local))
        files = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _files(local)}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"repository":info["repo"],"revision":info["revision"],"files":files},indent=2)+"\n")
        print(f"{name}: downloaded {len(files)} files at {info['revision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS), help="Repeat to select specific datasets; default is all eight")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root containing data/processed")
    parser.add_argument("--list", action="store_true", help="Print pinned identifiers without downloading")
    args = parser.parse_args()
    if args.list:
        for name, info in DATASETS.items():
            print(f"{name}: {info['repo']} @ {info['revision']}")
        return
    download_all(datasets=args.dataset, repo_root=args.root)


if __name__ == "__main__":
    main()
