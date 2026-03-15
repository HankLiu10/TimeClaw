"""Download all MTBench processed datasets from HuggingFace."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

DATASETS = {
    # Weather
    "weather_aligned_short": {
        "repo": "afeng/MTBench_weather_aligned_pairs_short",
        "local": "data/processed/weather/aligned_in7days_out1days",
    },
    "weather_aligned_long": {
        "repo": "afeng/MTBench_weather_aligned_pairs_long",
        "local": "data/processed/weather/aligned_in14days_out3days",
    },
    "weather_qa_short": {
        "repo": "afeng/MTBench_weather_QA_short",
        "local": "data/processed/weather/QAshort",
    },
    "weather_qa_long": {
        "repo": "afeng/MTBench_weather_QA_long",
        "local": "data/processed/weather/QAlong",
    },
    # Finance
    "finance_aligned_short": {
        "repo": "afeng/MTBench_finance_aligned_pairs_short",
        "local": "data/processed/finance/aligned_in7days_out1days",
    },
    "finance_aligned_long": {
        "repo": "afeng/MTBench_finance_aligned_pairs_long",
        "local": "data/processed/finance/aligned_in30days_out7days",
    },
    "finance_qa_short": {
        "repo": "afeng/MTBench_finance_QA_short",
        "local": "data/processed/finance/QAshort",
    },
    "finance_qa_long": {
        "repo": "afeng/MTBench_finance_QA_long",
        "local": "data/processed/finance/QAlong",
    },
}


def download_all(skip_existing: bool = True) -> None:
    for name, info in DATASETS.items():
        local_dir = Path(info["local"])
        if skip_existing and local_dir.exists():
            jsons = list(local_dir.glob("*.json"))
            if len(jsons) > 10:
                logger.info(f"[{name}] Already exists ({len(jsons)} files), skipping")
                continue
        logger.info(f"[{name}] Downloading {info['repo']} -> {local_dir}")
        snapshot_download(
            info["repo"],
            local_dir=str(local_dir),
            repo_type="dataset",
        )
        jsons = list(local_dir.glob("*.json"))
        logger.info(f"[{name}] Done: {len(jsons)} JSON files")


if __name__ == "__main__":
    download_all()
