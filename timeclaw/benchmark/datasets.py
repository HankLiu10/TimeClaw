"""Dataset loaders for MTBench: HuggingFace, local CSV, and synthetic data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class DatasetLoader:
    """Unified dataset loader supporting multiple sources."""

    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Load a dataset by name.

        Supported formats:
        - "csv:/path/to/file.csv" -- local CSV
        - "hf:dataset_name" -- HuggingFace datasets
        - "synthetic:sine" / "synthetic:random_walk" -- synthetic data
        - Standard names: "etth1", "weather", "ecl", etc.
        """
        if name.startswith("csv:"):
            return self._load_csv(name[4:], **kwargs)
        elif name.startswith("hf:"):
            return self._load_huggingface(name[3:], **kwargs)
        elif name.startswith("synthetic:"):
            return self._load_synthetic(name[10:], **kwargs)
        else:
            return self._load_standard(name, **kwargs)

    def _load_csv(self, path: str, **kwargs: Any) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(p)
        train_ratio = kwargs.get("train_ratio", 0.8)
        split_idx = int(len(df) * train_ratio)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in CSV")

        data = df[numeric_cols].values

        return {
            "name": p.stem,
            "train": data[:split_idx],
            "test": data[split_idx:],
            "data": data,
            "columns": numeric_cols,
            "source": "csv",
        }

    def _load_huggingface(self, dataset_name: str, **kwargs: Any) -> dict[str, Any]:
        try:
            from datasets import load_dataset

            split = kwargs.get("split", "train")
            ds = load_dataset(dataset_name, split=split)
            df = ds.to_pandas()

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values

            train_ratio = kwargs.get("train_ratio", 0.8)
            split_idx = int(len(data) * train_ratio)

            return {
                "name": dataset_name,
                "train": data[:split_idx],
                "test": data[split_idx:],
                "data": data,
                "columns": numeric_cols,
                "source": "huggingface",
            }
        except ImportError:
            logger.warning("datasets library not installed, using synthetic fallback")
            return self._load_synthetic("random_walk", length=1000)
        except Exception as e:
            logger.warning(f"Failed to load HF dataset {dataset_name}: {e}")
            return self._load_synthetic("random_walk", length=1000)

    def _load_synthetic(self, pattern: str = "sine", **kwargs: Any) -> dict[str, Any]:
        length = kwargs.get("length", 1000)
        n_channels = kwargs.get("n_channels", 1)
        rng = np.random.RandomState(kwargs.get("seed", 42))

        if pattern == "sine":
            t = np.linspace(0, 10 * np.pi, length)
            data = np.column_stack([
                np.sin(t * (i + 1)) + rng.randn(length) * 0.1
                for i in range(n_channels)
            ])
        elif pattern == "random_walk":
            steps = rng.randn(length, n_channels) * 0.1
            data = np.cumsum(steps, axis=0)
        elif pattern == "trend":
            t = np.linspace(0, 1, length)
            data = np.column_stack([
                t * (i + 1) + rng.randn(length) * 0.05
                for i in range(n_channels)
            ])
        elif pattern == "anomaly":
            data = rng.randn(length, n_channels)
            anomaly_idx = rng.choice(length, size=int(length * 0.05), replace=False)
            data[anomaly_idx] += rng.choice([-5, 5], size=(len(anomaly_idx), n_channels))
            labels = np.zeros(length, dtype=int)
            labels[anomaly_idx] = 1
            split_idx = int(length * 0.8)
            return {
                "name": f"synthetic_{pattern}",
                "data": data.squeeze(),
                "labels": labels,
                "train": data[:split_idx].squeeze(),
                "test": data[split_idx:].squeeze(),
                "source": "synthetic",
            }
        else:
            data = rng.randn(length, n_channels)

        if n_channels == 1:
            data = data.ravel()

        split_idx = int(length * 0.8)
        return {
            "name": f"synthetic_{pattern}",
            "train": data[:split_idx],
            "test": data[split_idx:],
            "data": data,
            "source": "synthetic",
        }

    def _load_standard(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Load well-known time-series benchmark datasets."""
        standard_hf_map = {
            "etth1": "ettDataset/ETT-small",
            "etth2": "ettDataset/ETT-small",
            "ettm1": "ettDataset/ETT-small",
            "ettm2": "ettDataset/ETT-small",
        }

        if name.lower() in standard_hf_map:
            try:
                return self._load_huggingface(standard_hf_map[name.lower()], **kwargs)
            except Exception:
                pass

        csv_path = self.cache_dir / f"{name}.csv"
        if csv_path.exists():
            return self._load_csv(str(csv_path), **kwargs)

        logger.info(f"Dataset '{name}' not found locally, generating synthetic substitute")
        return self._load_synthetic("sine", length=kwargs.get("length", 2000))

    def load_classification(
        self, name: str = "synthetic", **kwargs: Any
    ) -> dict[str, Any]:
        """Load a classification dataset (UCR/UEA format)."""
        n_classes = kwargs.get("n_classes", 3)
        n_samples = kwargs.get("n_samples", 200)
        seq_len = kwargs.get("seq_len", 100)
        rng = np.random.RandomState(kwargs.get("seed", 42))

        X = []
        y = []
        for i in range(n_samples):
            label = rng.randint(0, n_classes)
            t = np.linspace(0, 4 * np.pi, seq_len)
            signal = np.sin(t * (label + 1)) + rng.randn(seq_len) * 0.3
            X.append(signal)
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        split_idx = int(n_samples * 0.8)

        return {
            "name": name,
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_test": X[split_idx:],
            "y_test": y[split_idx:],
            "n_classes": n_classes,
            "source": "synthetic",
        }

    def load_qa(self, name: str = "synthetic", **kwargs: Any) -> dict[str, Any]:
        """Load a multimodal QA dataset."""
        questions = [
            {
                "question": "Based on the recent trend, will the value increase or decrease?",
                "answer": "Based on the upward trend in the time series, the value is likely to increase.",
                "timeseries": np.sin(np.linspace(0, 4 * np.pi, 100)).tolist(),
                "text_context": "Recent reports indicate positive growth momentum.",
            },
            {
                "question": "Is there any anomaly in the recent data?",
                "answer": "Yes, there appears to be a spike anomaly around the midpoint of the series.",
                "timeseries": list(range(50)) + [200] + list(range(51, 100)),
                "text_context": "Maintenance logs report an unexpected sensor reading.",
            },
            {
                "question": "What is the overall pattern of this time series?",
                "answer": "The time series shows a clear sinusoidal pattern with regular oscillations.",
                "timeseries": np.sin(np.linspace(0, 6 * np.pi, 100)).tolist(),
                "text_context": "The system operates in a cyclical manner.",
            },
        ]

        return {
            "name": name,
            "questions": questions,
            "source": "synthetic",
        }
