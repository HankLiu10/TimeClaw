"""Aurora adapter: Decision Intelligence Aurora forecasting model.

New definition:
- this adapter adds Aurora as a new optional local SBM backend
- it does not replace or alter the behavior of existing SBMs
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

AURORA_DEFAULT_CONFIG = SBMConfig(
    model_name="aurora",
    model_path="DecisionIntelligence/Aurora",
    context_length=512,
    prediction_length=96,
    num_samples=20,
    extra={"inference_token_len": 48},
)


class AuroraAdapter(SBMBase):
    """Adapter for the official Aurora forecasting package.

    TimeClaw keeps the first integration scope intentionally small:
    univariate time-series forecasting through the official `load_model`
    + `generate` API.
    """

    name: str = "aurora"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = True

    def load(self) -> None:
        try:
            from aurora import load_model
        except ImportError:
            raise ImportError(
                "aurora-model==0.2.0 is required: pip install 'timeclaw[sbm-aurora]'"
            )

        import torch

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self.config.model_path or "DecisionIntelligence/Aurora"
        logger.info(f"Loading Aurora from {model_path} on {device}")

        self._model = load_model(repo_id=model_path, device=device)
        self._model.eval()
        self._device = device
        logger.info("Aurora loaded successfully")

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
        num_samples = int(kwargs.get("num_samples", self.config.num_samples) or self.config.num_samples)
        inference_token_len = int(
            kwargs.get("inference_token_len", self.config.extra.get("inference_token_len", 48))
            or 48
        )

        if context.ndim == 1:
            univariate = np.asarray(context, dtype=np.float32)
        else:
            # Keep TimeClaw's existing forecasting contract unchanged: public
            # forecasting tools are still univariate by default.
            univariate = np.asarray(context[:, 0], dtype=np.float32)

        inputs = torch.tensor(univariate, dtype=torch.float32, device=self._device).unsqueeze(0)

        predictions = self._model.generate(
            inputs=inputs,
            num_samples=num_samples,
            max_output_length=pred_len,
            inference_token_len=inference_token_len,
        )

        preds_np = predictions.detach().float().cpu().numpy()
        if preds_np.ndim != 3:
            raise RuntimeError(f"Aurora returned unexpected prediction shape: {preds_np.shape}")

        samples_2d = preds_np[0]  # (num_samples, pred_len)
        point_forecast = np.median(samples_2d, axis=0)

        quantiles = {}
        for q in self.config.quantile_levels:
            quantiles[q] = np.quantile(samples_2d, q, axis=0)

        return SBMOutput(
            point_forecast=point_forecast,
            quantile_forecast=quantiles,
            samples=samples_2d,
            metadata={
                "model": "aurora",
                "prediction_length": pred_len,
                "num_samples": num_samples,
                "inference_token_len": inference_token_len,
            },
        )
