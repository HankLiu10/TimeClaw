"""Chronos adapter: Amazon's probabilistic time-series foundation model."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

CHRONOS_DEFAULT_CONFIG = SBMConfig(
    model_name="chronos",
    model_path="amazon/chronos-t5-large",
    context_length=512,
    prediction_length=64,
    num_samples=20,
)


class ChronosAdapter(SBMBase):
    """Adapter for Amazon Chronos probabilistic forecasting model.

    Chronos tokenizes time series into discrete bins and uses a T5-style
    language model for probabilistic forecasting via ancestral sampling.
    """

    name: str = "chronos"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = True

    def load(self) -> None:
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting is required: pip install chronos-forecasting"
            )

        import torch

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self.config.model_path or "amazon/chronos-t5-large"
        logger.info(f"Loading Chronos from {model_path} on {device}")

        self._model = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
        logger.info("Chronos loaded successfully")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        import torch

        pred_len = prediction_length or self.config.prediction_length
        num_samples = kwargs.get("num_samples", self.config.num_samples)

        if context.ndim == 1:
            ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        else:
            ctx_tensor = torch.tensor(context[:, 0], dtype=torch.float32).unsqueeze(0)

        samples = self._model.predict(
            ctx_tensor,
            prediction_length=pred_len,
            num_samples=num_samples,
        )

        samples_np = samples.numpy().squeeze(0)  # (num_samples, pred_len)
        point_forecast = np.median(samples_np, axis=0)

        quantiles = {}
        for q in self.config.quantile_levels:
            quantiles[q] = np.quantile(samples_np, q, axis=0)

        return SBMOutput(
            point_forecast=point_forecast,
            quantile_forecast=quantiles,
            samples=samples_np,
            metadata={"model": "chronos", "prediction_length": pred_len},
        )
