"""Domain-specific checkpoint forecast -- loads and runs saved model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class CheckpointForecastTool(ClawTool):
    """Load a domain-specific model checkpoint and produce forecasts.

    Supports PyTorch (.pt/.pth), ONNX (.onnx), and pickle (.pkl) checkpoints.
    """

    name: str = "checkpoint_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a time series by loading and running a previously trained model checkpoint from disk. "
        "Supports PyTorch (.pt/.pth), ONNX (.onnx), and pickle (.pkl) checkpoint formats. "
        "Input: 1-D or 2-D numeric array as context, plus a valid checkpoint file path. "
        "Output: dict with 'point_forecast' (array of predicted values), 'horizon', 'model', and 'checkpoint'. "
        "Use when a custom-trained model has been saved and you want to run inference with it "
        "rather than using a general-purpose forecasting method. "
        "Key params: 'checkpoint_path' (required, path to saved model); 'horizon' — steps ahead (default 96); "
        "'model_type' — format hint ('pytorch', 'onnx', 'pickle', or 'auto' to detect from extension)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "checkpoint_path": {
                "type": "string",
                "description": "Path to the model checkpoint file.",
            },
            "horizon": {
                "type": "integer",
                "default": 96,
                "description": "Number of steps to forecast.",
            },
            "model_type": {
                "type": "string",
                "enum": ["pytorch", "onnx", "pickle", "auto"],
                "description": "Checkpoint format.",
            },
        },
        "required": ["checkpoint_path"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        checkpoint_path = params.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        horizon = params.get("horizon", 96)
        model_type = params.get("model_type", "auto")

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if model_type == "auto":
            model_type = self._detect_type(path)

        context = np.asarray(ts_data, dtype=np.float32)
        if context.ndim == 0:
            raise ValueError("ts_data must be at least 1D")

        if model_type == "pytorch":
            return await self._pytorch_forecast(path, context, horizon)
        elif model_type == "onnx":
            return await self._onnx_forecast(path, context, horizon)
        elif model_type == "pickle":
            return await self._pickle_forecast(path, context, horizon)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _detect_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in (".pt", ".pth"):
            return "pytorch"
        elif suffix == ".onnx":
            return "onnx"
        elif suffix in (".pkl", ".pickle"):
            return "pickle"
        raise ValueError(f"Cannot detect model type from extension: {suffix}")

    async def _pytorch_forecast(
        self, path: Path, context: np.ndarray, horizon: int
    ) -> dict[str, Any]:
        import torch

        model = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(model, "eval"):
            model.eval()

        if context.ndim == 1:
            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        else:
            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if hasattr(model, "forecast"):
                pred = model.forecast(x, horizon=horizon)
            elif hasattr(model, "forward"):
                pred = model(x)
            else:
                pred = model(x)

        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy().ravel()[:horizon]
        else:
            pred_np = np.asarray(pred).ravel()[:horizon]

        return {
            "model": "checkpoint_pytorch",
            "checkpoint": str(path),
            "horizon": horizon,
            "point_forecast": pred_np.tolist(),
        }

    async def _onnx_forecast(
        self, path: Path, context: np.ndarray, horizon: int
    ) -> dict[str, Any]:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX checkpoints")

        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name

        if context.ndim == 1:
            x = context.reshape(1, -1, 1).astype(np.float32)
        else:
            x = context.reshape(1, *context.shape).astype(np.float32)

        outputs = session.run(None, {input_name: x})
        pred_np = np.asarray(outputs[0]).ravel()[:horizon]

        return {
            "model": "checkpoint_onnx",
            "checkpoint": str(path),
            "horizon": horizon,
            "point_forecast": pred_np.tolist(),
        }

    async def _pickle_forecast(
        self, path: Path, context: np.ndarray, horizon: int
    ) -> dict[str, Any]:
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        if hasattr(model, "predict"):
            pred = model.predict(context.reshape(1, -1))
            pred_np = np.asarray(pred).ravel()[:horizon]
        elif hasattr(model, "forecast"):
            pred = model.forecast(steps=horizon)
            pred_np = np.asarray(pred).ravel()[:horizon]
        else:
            raise ValueError("Pickle model has no predict() or forecast() method")

        return {
            "model": "checkpoint_pickle",
            "checkpoint": str(path),
            "horizon": horizon,
            "point_forecast": pred_np.tolist(),
        }


tool = CheckpointForecastTool()
