"""Moirai adapter: Salesforce's universal time-series forecasting model."""

from __future__ import annotations

import importlib
import os
import threading
import ctypes
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig, SBMOutput, SBMTask

MOIRAI_DEFAULT_CONFIG = SBMConfig(
    model_name="moirai1_1_r_base",
    model_path="Salesforce/moirai-1.1-R-base",
    context_length=512,
    prediction_length=96,
    num_samples=20,
)

_IMPORT_LOCK = threading.Lock()
_IMPORT_READY = False


def _prefer_conda_libstdcpp() -> None:
    """Preload Conda's libstdc++ for packages that need newer GLIBCXX symbols."""
    conda_prefix = os.environ.get("CONDA_PREFIX") or ""
    if not conda_prefix:
        return
    libstdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if not os.path.exists(libstdcpp):
        return
    try:
        ctypes.CDLL(libstdcpp, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        logger.debug(f"Failed to preload Conda libstdc++: {e}")


def _ensure_moirai_imports() -> tuple[type[Any], type[Any]]:
    """Import Moirai classes once and preserve the real import failure reason."""
    global _IMPORT_READY
    with _IMPORT_LOCK:
        _prefer_conda_libstdcpp()
        try:
            forecast_mod = importlib.import_module("uni2ts.model.moirai.forecast")
            module_mod = importlib.import_module("uni2ts.model.moirai.module")
        except Exception as e:
            raise ImportError(
                f"Failed to import uni2ts Moirai modules: {type(e).__name__}: {e}"
            ) from e
        _IMPORT_READY = True
        return forecast_mod.MoiraiForecast, module_mod.MoiraiModule


class MoiraiAdapter(SBMBase):
    """Adapter for Salesforce Moirai universal forecasting model.

    Moirai uses a masked-encoder architecture with any-variate attention,
    supporting variable context/prediction lengths and multivariate input.
    """

    name: str = "moirai1_1_r_base"
    supported_tasks: list[SBMTask] = [SBMTask.FORECAST]
    requires_gpu: bool = True

    def load(self) -> None:
        _, MoiraiModule = _ensure_moirai_imports()

        import torch

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self.config.model_path or "Salesforce/moirai-1.1-R-base"
        logger.info(f"Loading Moirai from {model_path} on {device}")

        module = MoiraiModule.from_pretrained(model_path)
        if device != "cpu":
            module = module.to(device)
        self._module = module
        self._device = device
        self._model = True  # flag that model is loaded
        logger.info(f"Moirai loaded successfully on {device}")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        if not self.is_loaded:
            self.load()

        MoiraiForecast, _ = _ensure_moirai_imports()
        import torch

        pred_len = prediction_length or self.config.prediction_length
        num_samples = kwargs.get("num_samples", self.config.num_samples)

        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset

        if context.ndim == 2:
            univariate = context[:, 0]
        else:
            univariate = context

        ctx_len = min(len(univariate), self.config.context_length)
        univariate = univariate[-ctx_len:]

        model = MoiraiForecast(
            module=self._module,
            prediction_length=pred_len,
            context_length=ctx_len,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        s = pd.Series(
            univariate,
            index=pd.period_range(start="2000-01-01", periods=len(univariate), freq="h"),
            name="target",
        )
        ds = PandasDataset(s)

        predictor = model.create_predictor(batch_size=1)
        forecasts = list(predictor.predict(ds))

        if forecasts:
            fc = forecasts[0]
            samples_np = fc.samples  # (num_samples, pred_len, n_vars)
            if samples_np.ndim == 3:
                point_forecast = np.median(samples_np, axis=0)[:, 0]
                samples_2d = samples_np[:, :, 0]
            else:
                point_forecast = np.median(samples_np, axis=0)
                samples_2d = samples_np

            quantiles = {}
            for q in self.config.quantile_levels:
                quantiles[q] = np.quantile(samples_2d, q, axis=0)

            return SBMOutput(
                point_forecast=point_forecast,
                quantile_forecast=quantiles,
                samples=samples_2d,
                metadata={"model": "Salesforce/moirai-1.1-R-base", "prediction_length": pred_len},
            )

        return SBMOutput(
            point_forecast=np.zeros(pred_len),
            metadata={"model": "Salesforce/moirai-1.1-R-base", "error": "no forecasts generated"},
        )
