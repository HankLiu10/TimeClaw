"""TimesFM adapter: Google's time-series foundation model."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

TIMESFM_DEFAULT_CONFIG = SBMConfig(
    model_name="timesfm",
    model_path="google/timesfm-1.0-200m-pytorch",
    context_length=512,
    prediction_length=96,
)


class TimesFMAdapter(SBMBase):
    """Adapter for Google TimesFM foundation model.

    TimesFM is a decoder-only model pre-trained on a large corpus of
    real-world and synthetic time series, supporting zero-shot forecasting.
    """

    name: str = "timesfm"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = False

    def load(self) -> None:
        try:
            import timesfm as tfm
        except ImportError:
            raise ImportError(
                "timesfm is required: pip install timesfm"
            )

        model_path = self.config.model_path or "google/timesfm-1.0-200m-pytorch"
        logger.info(f"Loading TimesFM from {model_path}")

        backend = "gpu" if self.config.device in ("cuda", "auto") else "cpu"

        self._model = tfm.TimesFm(
            hparams=tfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.config.batch_size,
                horizon_len=self.config.prediction_length,
                context_len=self.config.context_length,
            ),
            checkpoint=tfm.TimesFmCheckpoint(
                huggingface_repo_id=model_path,
            ),
        )
        logger.info("TimesFM loaded successfully")

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
            input_data = [context.astype(np.float32)]
        else:
            input_data = [context[:, 0].astype(np.float32)]

        freq = kwargs.get("freq", [0])

        point_forecasts, quantile_forecasts = self._model.forecast(
            input_data,
            freq=freq,
        )

        point = point_forecasts[0][:pred_len]

        quantiles = {}
        if quantile_forecasts is not None and len(quantile_forecasts) > 0:
            qf = quantile_forecasts[0]
            default_qs = [0.1, 0.5, 0.9]
            for i, q in enumerate(default_qs):
                if i < qf.shape[1] if qf.ndim > 1 else False:
                    quantiles[q] = qf[:pred_len, i]

        return SBMOutput(
            point_forecast=point,
            quantile_forecast=quantiles if quantiles else None,
            metadata={"model": "timesfm", "prediction_length": pred_len},
        )
