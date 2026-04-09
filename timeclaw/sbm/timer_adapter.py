"""Sundial adapter backed by THU Sundial Base 128M."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

TIMER_DEFAULT_CONFIG = SBMConfig(
    model_name="sundial_base_128m",
    model_path="thuml/sundial-base-128m",
    context_length=672,
    prediction_length=96,
)


class TimerAdapter(SBMBase):
    """Adapter for THU Sundial Base 128M.

    Sundial is the newer THU forecasting model line.
    """

    name: str = "sundial_base_128m"
    supported_tasks: list[SBMTask] = [
        SBMTask.FORECAST,
        SBMTask.IMPUTATION,
        SBMTask.ANOMALY,
    ]
    requires_gpu: bool = True

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers and torch are required for Sundial"
            )

        model_path = self.config.model_path or "thuml/sundial-base-128m"
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Sundial from {model_path} on {device}")

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()
            self._device = device
        except Exception as e:
            raise RuntimeError(f"Failed to load Sundial via AutoModel: {e}") from e

        logger.info("Sundial loaded successfully")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        pred_len = prediction_length or self.config.prediction_length

        import torch

        flat = context.ravel() if context.ndim == 1 else context[:, 0]
        input_token_len = getattr(self._model.config, "input_token_len", 96)
        usable_len = (len(flat) // input_token_len) * input_token_len
        if usable_len < input_token_len:
            raise ValueError(
                f"Sundial requires at least {input_token_len} usable points, got {len(flat)}"
            )
        flat = flat[-usable_len:]

        ctx_tensor = torch.tensor(flat.reshape(1, -1), dtype=torch.float32).to(self._device)

        with torch.no_grad():
            try:
                output = self._model(
                    ctx_tensor,
                    use_cache=False,
                    revin=True,
                    max_output_length=pred_len,
                )
                logits = output.logits
                pred = np.asarray(logits[0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                return SBMOutput(
                    point_forecast=pred[:pred_len],
                    metadata={"model": "thuml/sundial-base-128m", "prediction_length": pred_len},
                )
            except Exception as e:
                raise RuntimeError(f"Sundial forward failed: {e}") from e

    def impute(self, data: np.ndarray, mask: np.ndarray, **kwargs: Any) -> SBMOutput:
        if not self.is_loaded:
            self.load()
        raise NotImplementedError("Sundial imputation is not implemented")
