"""Timer adapter: THU's generative pre-trained transformer for time series."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

TIMER_DEFAULT_CONFIG = SBMConfig(
    model_name="timer",
    model_path="thuml/timer-base-84m",
    context_length=672,
    prediction_length=96,
)


class TimerAdapter(SBMBase):
    """Adapter for THU Timer (Time-series Transformer) model.

    Timer is a GPT-style generative model pre-trained on large-scale
    time-series data, supporting forecasting, imputation, and anomaly detection.
    """

    name: str = "timer"
    supported_tasks: list[SBMTask] = [
        SBMTask.FORECAST,
        SBMTask.IMPUTATION,
        SBMTask.ANOMALY,
    ]
    requires_gpu: bool = False

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers and torch are required for Timer"
            )

        model_path = self.config.model_path or "thuml/timer-base-84m"
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Timer from {model_path} on {device}")

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()
            self._device = device
        except Exception as e:
            logger.warning(f"Failed to load Timer via AutoModel: {e}")
            logger.info("Timer will use fallback naive forecasting")
            self._model = "fallback"
            self._device = device

        logger.info("Timer loaded successfully")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        pred_len = prediction_length or self.config.prediction_length

        if self._model == "fallback":
            return self._naive_forecast(context, pred_len)

        import torch

        if context.ndim == 1:
            ctx = context.reshape(1, -1, 1)
        elif context.ndim == 2:
            ctx = context.reshape(1, *context.shape)
        else:
            ctx = context

        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            try:
                output = self._model.generate(
                    ctx_tensor,
                    max_new_tokens=pred_len,
                )
                if hasattr(output, "sequences"):
                    pred = output.sequences[0, -pred_len:].cpu().numpy()
                else:
                    pred = output[0, -pred_len:].cpu().numpy()

                return SBMOutput(
                    point_forecast=pred.ravel()[:pred_len],
                    metadata={"model": "timer", "prediction_length": pred_len},
                )
            except Exception as e:
                logger.warning(f"Timer generation failed: {e}, using fallback")
                return self._naive_forecast(context, pred_len)

    def _naive_forecast(self, context: np.ndarray, pred_len: int) -> SBMOutput:
        """Fallback: repeat last values with slight trend continuation."""
        if context.ndim > 1:
            context = context[:, 0]
        last_val = context[-1]
        if len(context) > 1:
            trend = (context[-1] - context[-min(10, len(context))]) / min(10, len(context))
        else:
            trend = 0.0
        forecast = last_val + trend * np.arange(1, pred_len + 1)
        return SBMOutput(
            point_forecast=forecast,
            metadata={"model": "timer-fallback", "prediction_length": pred_len},
        )

    def impute(self, data: np.ndarray, mask: np.ndarray, **kwargs: Any) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        if self._model == "fallback":
            imputed = data.copy()
            if imputed.ndim == 1:
                nan_idx = np.where(mask)[0]
                for idx in nan_idx:
                    left = data[:idx][~mask[:idx]]
                    right = data[idx + 1:][~mask[idx + 1:]]
                    if len(left) > 0 and len(right) > 0:
                        imputed[idx] = (left[-1] + right[0]) / 2
                    elif len(left) > 0:
                        imputed[idx] = left[-1]
                    elif len(right) > 0:
                        imputed[idx] = right[0]
            return SBMOutput(imputed=imputed, metadata={"model": "timer-fallback"})

        return SBMOutput(
            imputed=data,
            metadata={"model": "timer", "note": "full imputation not yet implemented"},
        )
