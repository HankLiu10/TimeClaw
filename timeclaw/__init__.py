"""TimeClaw: Native Multimodal Time-Series Autonomous Agent Platform."""

from __future__ import annotations

import ctypes
import os


def _preload_conda_libstdcpp() -> None:
    """Prefer Conda's libstdc++ early for C++ extensions needing newer GLIBCXX."""
    conda_prefix = os.environ.get("CONDA_PREFIX") or ""
    if not conda_prefix:
        return
    libstdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if not os.path.exists(libstdcpp):
        return
    try:
        ctypes.CDLL(libstdcpp, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_preload_conda_libstdcpp()

__version__ = "0.1.0"
