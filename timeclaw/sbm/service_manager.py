"""Managed lifecycle for per-model SBM HTTP services used by benchmark flows."""

from __future__ import annotations

import atexit
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

from loguru import logger

from timeclaw.core.config import TimeClawConfig
from timeclaw.sbm.registry import register_remote_server_client_map

_lock = Lock()
_server_processes: dict[str, subprocess.Popen[bytes]] = {}
_server_urls: dict[str, list[str]] = {}
_server_log_paths: dict[str, Path] = {}
_atexit_registered = False


def _managed_server_log_path(
    cfg: TimeClawConfig,
    model_name: str,
    replica_idx: int = 0,
) -> Path:
    log_dir = Path(cfg.workspace_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(model_name).replace("/", "_")
    return log_dir / f"sbm_managed_server_{safe_name}_r{replica_idx}.log"


def _tail_text(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _server_is_healthy(url: str) -> bool:
    from timeclaw.sbm.http_client import get_server_models, is_server_available

    try:
        if not is_server_available(url):
            return False
        models = [str(x).strip() for x in get_server_models(url) if str(x).strip()]
        return bool(models)
    except Exception:
        return False


def _listener_pids_for_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"( sport = :{int(port)} )"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids: list[int] = []
    for pid_text in re.findall(r"pid=(\d+)", out):
        pid = int(pid_text)
        if pid not in pids:
            pids.append(pid)
    return pids


def _terminate_existing_listener(
    port: int,
    *,
    expected_model: str,
    reason: str,
) -> None:
    pids = _listener_pids_for_port(port)
    if not pids:
        return
    logger.warning(
        f"[SBM-Manager] Port {port} is occupied while preparing {expected_model}; "
        f"{reason}. Terminating listener pid(s) {pids}"
    )
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _listener_pids_for_port(port):
            return
        time.sleep(0.2)
    for pid in _listener_pids_for_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _ensure_target_port_available(port: int, *, expected_model: str) -> None:
    if not _listener_pids_for_port(port):
        return
    _terminate_existing_listener(
        port,
        expected_model=expected_model,
        reason="managed SBM lifecycle requires a fresh per-run service on this port",
    )


def _reuse_existing_healthy_listener(
    *,
    cfg: TimeClawConfig,
    model_name: str,
    ordinal: int,
    replica_idx: int = 0,
) -> str | None:
    """Return an already-running compatible SBM service if one is healthy.

    New definition:
        Managed SBM startup is now "reuse-first" across benchmark processes.
        If the target port already serves the expected model and passes health
        checks, benchmark flows bind to that existing service instead of
        force-killing the listener and recreating it.
    """
    url = _model_url(cfg, model_name, ordinal, replica_idx=replica_idx)
    port = _model_port(cfg, model_name, ordinal, replica_idx=replica_idx)
    if not _listener_pids_for_port(port):
        return None
    from timeclaw.sbm.http_client import get_server_models

    if not _server_is_healthy(url):
        return None
    models = [str(x).strip() for x in get_server_models(url) if str(x).strip()]
    if model_name not in models:
        return None
    logger.info(
        f"[SBM-Manager] Reusing healthy existing listener for {model_name} "
        f"at {url} (replica r{replica_idx})"
    )
    _server_urls.setdefault(model_name, [])
    if url not in _server_urls[model_name]:
        _server_urls[model_name].append(url)
    return url


def _model_device(cfg: TimeClawConfig, model_name: str) -> str:
    return str(
        cfg.sbm.devices.get(model_name)
        or cfg.sbm.devices.get(str(model_name))
        or cfg.sbm.device
    )


def _model_port(
    cfg: TimeClawConfig,
    model_name: str,
    ordinal: int,
    *,
    replica_idx: int = 0,
) -> int:
    raw = cfg.sbm.managed_server_ports.get(model_name)
    if raw is None:
        raw = cfg.sbm.managed_server_ports.get(str(model_name))
    if raw is not None:
        return int(raw) + int(replica_idx)
    replicas = max(1, int(cfg.sbm.managed_server_replicas))
    return int(cfg.sbm.managed_server_base_port) + int(ordinal) * replicas + int(replica_idx)


def _model_url(
    cfg: TimeClawConfig,
    model_name: str,
    ordinal: int,
    *,
    replica_idx: int = 0,
) -> str:
    host = str(cfg.sbm.managed_server_host).strip() or "127.0.0.1"
    port = _model_port(cfg, model_name, ordinal, replica_idx=replica_idx)
    return f"http://{host}:{port}"


def _start_one_model_server(
    cfg: TimeClawConfig,
    *,
    model_name: str,
    ordinal: int,
    replica_idx: int = 0,
) -> str:
    global _server_processes, _server_urls, _server_log_paths

    replica_key = f"{model_name}#r{replica_idx}"
    url = _model_url(cfg, model_name, ordinal, replica_idx=replica_idx)
    host = str(cfg.sbm.managed_server_host).strip() or "127.0.0.1"
    port = _model_port(cfg, model_name, ordinal, replica_idx=replica_idx)
    device = _model_device(cfg, model_name)

    existing = _server_processes.get(replica_key)
    if existing is not None:
        if existing.poll() is not None:
            raise RuntimeError(
                f"Managed SBM server for {replica_key} exited early with code {existing.returncode}. "
                f"See {_server_log_paths.get(replica_key)}."
            )
        return url

    reused_url = _reuse_existing_healthy_listener(
        cfg=cfg,
        model_name=model_name,
        ordinal=ordinal,
        replica_idx=replica_idx,
    )
    if reused_url is not None:
        return reused_url

    _ensure_target_port_available(port, expected_model=model_name)

    log_path = _managed_server_log_path(cfg, model_name, replica_idx)
    _server_log_paths[replica_key] = log_path
    cmd = [
        cfg.sbm.managed_server_python.get(model_name, sys.executable),
        "-m",
        "timeclaw.sbm.server",
        "--host",
        host,
        "--port",
        str(port),
        "--models",
        str(model_name),
        "--device",
        str(device),
    ]

    log_handle = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise

    _server_processes[replica_key] = proc
    timeout_s = max(5.0, float(cfg.sbm.managed_server_health_timeout_s))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail_text(log_path)
            raise RuntimeError(
                f"Managed SBM server for {model_name} exited during startup "
                f"(code={proc.returncode}). Log tail:\n{tail}"
            )
        if _server_is_healthy(url):
            _server_urls.setdefault(model_name, [])
            if url not in _server_urls[model_name]:
                _server_urls[model_name].append(url)
            logger.info(
                f"[SBM-Manager] {replica_key} server ready at {url}; "
                f"log={log_path}; cmd={' '.join(shlex.quote(x) for x in cmd)}"
            )
            return url
        time.sleep(1.0)

    tail = _tail_text(log_path)
    raise RuntimeError(
        f"Managed SBM server for {model_name} did not become healthy within {timeout_s:.1f}s. "
        f"Log tail:\n{tail}"
    )


def ensure_managed_sbm_server(cfg: TimeClawConfig) -> dict[str, list[str]]:
    """Start one isolated SBM service per model and register matching clients.

    New definition:
    - Benchmark train/test flows now require strict per-model isolation.
    - Each configured model gets one or more isolated HTTP server replicas,
      each with its own process, port, and execution queue.
    - Registry registration is logical-model based: one model name can map to
      many replica URLs and is then exposed through round-robin routing.
    """
    global _atexit_registered

    if not cfg.sbm.managed_server_enabled:
        return {}

    model_names = [
        str(name).strip()
        for name in cfg.sbm.managed_server_models
        if str(name).strip()
    ]
    if not model_names:
        raise RuntimeError("SBM managed services have no configured models to load.")

    replicas = max(1, int(cfg.sbm.managed_server_replicas))

    with _lock:
        model_to_url: dict[str, list[str]] = {}
        for ordinal, model_name in enumerate(model_names):
            urls: list[str] = []
            for replica_idx in range(replicas):
                urls.append(_start_one_model_server(
                    cfg,
                    model_name=model_name,
                    ordinal=ordinal,
                    replica_idx=replica_idx,
                ))
            model_to_url[model_name] = urls
        if not _atexit_registered:
            atexit.register(shutdown_managed_sbm_server)
            _atexit_registered = True
        register_remote_server_client_map(model_to_url, replace=True)
        return {name: list(urls) for name, urls in model_to_url.items()}


def shutdown_managed_sbm_server() -> None:
    """Stop all managed per-model SBM service processes started by this script."""
    global _server_processes, _server_urls, _server_log_paths

    with _lock:
        for model_name, proc in list(_server_processes.items()):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5)
            logger.info(f"[SBM-Manager] Managed SBM server stopped for {model_name}")
        _server_processes = {}
        _server_urls = {}
        _server_log_paths = {}
