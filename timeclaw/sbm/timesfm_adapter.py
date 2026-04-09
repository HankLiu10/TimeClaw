"""TimesFM adapter backed by Google TimesFM 2.0."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

TIMESFM_DEFAULT_CONFIG = SBMConfig(
    model_name="timesfm2",
    model_path="google/timesfm-2.0-500m-pytorch",
    context_length=2048,
    prediction_length=128,
)

_TIMESFM2_FIXED_HPARAMS = {
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 50,
    "model_dims": 1280,
    "use_positional_embedding": False,
}


class TimesFMAdapter(SBMBase):
    """Adapter for Google TimesFM 2.0.

    New definition:
        TimeClaw now loads TimesFM through the official ``timesfm`` package and
        the official ``google/timesfm-2.0-500m-pytorch`` checkpoint, instead of
        the previous ad hoc Transformers 2.5 route.

    Why the fixed hparams are required:
        The official 500M checkpoint is not shape-compatible with arbitrary
        defaults. The following parameters must match the model card exactly:
        ``input_patch_len=32``, ``output_patch_len=128``, ``num_layers=50``,
        ``model_dims=1280``, ``use_positional_embedding=False``.
    """

    name: str = "timesfm2"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = True

    def load(self) -> None:
        try:
            import torch
            import timesfm
        except ImportError as e:
            raise ImportError(
                "TimesFM 2.0 requires the official `timesfm` package and torch"
            ) from e

        model_path = self.config.model_path or "google/timesfm-2.0-500m-pytorch"
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        backend = "gpu" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading TimesFM 2.0 from {model_path} on {device} (backend={backend})")

        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                context_len=self.config.context_length,
                horizon_len=self.config.prediction_length,
                per_core_batch_size=max(1, int(self.config.batch_size)),
                **_TIMESFM2_FIXED_HPARAMS,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=model_path,
            ),
        )

        target_device = torch.device(device)
        if model._device != target_device:
            model._model.to(target_device)
            model._device = target_device

        self._model = model
        self._device = target_device
        logger.info("TimesFM 2.0 loaded successfully")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        pred_len = int(prediction_length or self.config.prediction_length)
        max_horizon = int(self.config.prediction_length)
        if pred_len > max_horizon:
            raise ValueError(
                f"TimesFM 2.0 adapter is initialized with horizon_len={max_horizon}; got {pred_len}"
            )

        if context.ndim == 1:
            univariate = np.asarray(context, dtype=np.float32)
        else:
            univariate = np.asarray(context[:, 0], dtype=np.float32)

        ctx_len = min(len(univariate), self.config.context_length)
        raw_freq = kwargs.get("freq", 0)
        if isinstance(raw_freq, (list, tuple)):
            freq = [int(raw_freq[0] if raw_freq else 0)]
        else:
            freq = [int(raw_freq)]

        point, full = self._model.forecast(
            [univariate[-ctx_len:]],
            freq=freq,
            forecast_context_len=ctx_len,
        )

        point_forecast = np.asarray(point[0], dtype=np.float32)[:pred_len]
        quantiles: dict[float, np.ndarray] = {}
        model_quantiles = list(getattr(self._model, "quantiles", []) or [])
        full_forecast = np.asarray(full[0], dtype=np.float32)[:pred_len]
        for idx, q in enumerate(model_quantiles, start=1):
            if idx < full_forecast.shape[-1]:
                quantiles[float(q)] = full_forecast[:, idx]

        return SBMOutput(
            point_forecast=point_forecast,
            quantile_forecast=quantiles if quantiles else None,
            metadata={
                "model": "timesfm-2.0-500m-pytorch",
                "prediction_length": pred_len,
                "context_length": ctx_len,
            },
        )
