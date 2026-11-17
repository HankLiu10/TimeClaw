"""Chronos adapter backed by Amazon Chronos-2."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

CHRONOS_DEFAULT_CONFIG = SBMConfig(
    model_name="chronos2",
    model_path="amazon/chronos-2",
    context_length=512,
    prediction_length=64,
    num_samples=20,
)


class ChronosAdapter(SBMBase):
    """Adapter for Amazon Chronos-2 forecasting model.

    Chronos-2 produces direct point/quantile forecasts and supports richer
    forecasting interfaces than the original Chronos family.
    """

    name: str = "chronos2"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = True

    def load(self) -> None:
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting>=2.2 is required: pip install 'chronos-forecasting>=2.2'"
            )

        import torch

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self.config.model_path or "amazon/chronos-2"
        logger.info(f"Loading Chronos-2 from {model_path} on {device}")

        self._model = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
        logger.info("Chronos-2 loaded successfully")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        pred_len = prediction_length or self.config.prediction_length

        if context.ndim == 1:
            univariate = np.asarray(context, dtype=np.float32)
        else:
            # Keep the previous TimeClaw behavior for non-1D inputs: use the
            # first variate only, because the public tool contract is still
            # documented as univariate forecasting.
            univariate = np.asarray(context[:, 0], dtype=np.float32)
        model_input = univariate[None, None, :]

        quantile_tensors, mean_tensors = self._model.predict_quantiles(
            model_input,
            prediction_length=pred_len,
            quantile_levels=self.config.quantile_levels,
            limit_prediction_length=False,
        )

        point_forecast = mean_tensors[0].detach().cpu().numpy()
        if point_forecast.ndim > 1 and point_forecast.shape[0] == 1:
            point_forecast = point_forecast.squeeze(0)

        quantiles = {}
        first_quantiles = quantile_tensors[0].detach().cpu().numpy()
        if first_quantiles.ndim > 2 and first_quantiles.shape[0] == 1:
            first_quantiles = first_quantiles.squeeze(0)
        for idx, q in enumerate(self.config.quantile_levels):
            quantile_forecast = first_quantiles[..., idx]
            quantiles[q] = quantile_forecast

        return SBMOutput(
            point_forecast=point_forecast,
            quantile_forecast=quantiles,
            metadata={"model": "chronos-2", "prediction_length": pred_len},
        )
