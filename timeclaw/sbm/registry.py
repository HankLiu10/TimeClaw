"""SBMRegistry: register, discover, and retrieve SBM model adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loguru import logger

from timeclaw.sbm.base import SBMBase, SBMConfig


class SBMRegistry:
    """Central registry for SBM model adapters.

    Supports lazy loading: adapters are registered by factory function
    and instantiated only when first requested.
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[SBMBase]] = {}
        self._instances: dict[str, SBMBase] = {}
        self._default_configs: dict[str, SBMConfig] = {}

    def register(
        self,
        name: str,
        adapter_cls: type[SBMBase],
        default_config: SBMConfig | None = None,
    ) -> None:
        """Register an SBM adapter class."""
        self._factories[name] = adapter_cls
        if default_config:
            self._default_configs[name] = default_config
        logger.debug(f"SBM registered: {name}")

    def get(self, name: str, config: SBMConfig | None = None) -> SBMBase:
        """Get or create an SBM adapter instance."""
        if name in self._instances:
            return self._instances[name]

        if name not in self._factories:
            raise KeyError(
                f"SBM '{name}' not registered. Available: {list(self._factories.keys())}"
            )

        cfg = config or self._default_configs.get(name) or SBMConfig(model_name=name)
        instance = self._factories[name](cfg)
        self._instances[name] = instance
        return instance

    def list_available(self) -> list[str]:
        """List all registered SBM names."""
        return list(self._factories.keys())

    def list_loaded(self) -> list[str]:
        """List currently loaded (in-memory) SBM instances."""
        return [n for n, inst in self._instances.items() if inst.is_loaded]

    def get_profiles(self) -> list[dict[str, Any]]:
        """Get profiles for all registered models."""
        profiles = []
        for name in self._factories:
            if name in self._instances:
                profiles.append(self._instances[name].get_profile())
            else:
                profiles.append({"name": name, "is_loaded": False})
        return profiles

    def unload(self, name: str) -> None:
        """Unload a specific model from memory."""
        if name in self._instances:
            self._instances[name].unload()
            del self._instances[name]

    def unload_all(self) -> None:
        """Unload all models."""
        for inst in self._instances.values():
            inst.unload()
        self._instances.clear()


_global_registry = SBMRegistry()


def _register_local_adapter(
    registry: SBMRegistry,
    *,
    name: str,
    adapter_cls: type[SBMBase],
    default_config: SBMConfig,
    force: bool = False,
) -> None:
    """Register one local adapter on the global registry.

    New definition:
    - ``force=True`` means "replace any existing instance for this model name"
      before re-registering. This matters when a previous call inserted an HTTP
      client placeholder with the same model name, but the current workflow now
      requires the real local GPU-backed adapter.
    """
    if force and name in registry._instances:
        registry.unload(name)
    registry.register(name, adapter_cls, configured_default_config(name, default_config))


def register_local_adapters(*, force: bool = False) -> list[str]:
    """Register all built-in local SBM adapters.

    New definition:
    - This function always targets local adapters only.
    - It does not probe or prefer the optional SBM HTTP server.
    - Returned names are the successfully registered local model names.
    """
    registry = get_global_registry()
    registered: list[str] = []

    try:
        from timeclaw.sbm.chronos_adapter import ChronosAdapter, CHRONOS_DEFAULT_CONFIG
        _register_local_adapter(
            registry,
            name="chronos2",
            adapter_cls=ChronosAdapter,
            default_config=CHRONOS_DEFAULT_CONFIG,
            force=force,
        )
        registered.append("chronos2")
    except ImportError:
        logger.debug("Chronos adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.moirai_adapter import MoiraiAdapter, MOIRAI_DEFAULT_CONFIG
        _register_local_adapter(
            registry,
            name="moirai1_1_r_base",
            adapter_cls=MoiraiAdapter,
            default_config=MOIRAI_DEFAULT_CONFIG,
            force=force,
        )
        registered.append("moirai1_1_r_base")
    except ImportError:
        logger.debug("Moirai adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.timesfm_adapter import TimesFMAdapter, TIMESFM_DEFAULT_CONFIG
        _register_local_adapter(
            registry,
            name="timesfm2",
            adapter_cls=TimesFMAdapter,
            default_config=TIMESFM_DEFAULT_CONFIG,
            force=force,
        )
        registered.append("timesfm2")
    except ImportError:
        logger.debug("TimesFM adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.timer_adapter import TimerAdapter, TIMER_DEFAULT_CONFIG
        _register_local_adapter(
            registry,
            name="sundial_base_128m",
            adapter_cls=TimerAdapter,
            default_config=TIMER_DEFAULT_CONFIG,
            force=force,
        )
        registered.append("sundial_base_128m")
    except ImportError:
        logger.debug("Timer adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.aurora_adapter import AuroraAdapter, AURORA_DEFAULT_CONFIG
        _register_local_adapter(
            registry,
            name="aurora",
            adapter_cls=AuroraAdapter,
            default_config=AURORA_DEFAULT_CONFIG,
            force=force,
        )
        registered.append("aurora")
    except ImportError:
        logger.debug("Aurora adapter not available (missing dependencies)")

    return registered


def preload_local_gpu_models(
    *,
    model_names: Iterable[str] | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Register and eagerly load all requested local GPU SBM adapters.

    New definition:
    - This is the benchmark-side "explicit deployment" entry point.
    - It bypasses the optional HTTP-server auto-discovery path on purpose,
      because training/evaluation here require the real local models to be
      placed onto the configured GPUs rather than remote client stubs.
    - When ``strict=True``, any registered GPU model that fails to load aborts
      the workflow immediately instead of silently falling back to lazy loading.
    """
    registry = get_global_registry()
    local_names = register_local_adapters(force=True)
    requested = list(model_names) if model_names is not None else list(local_names)
    ready: list[dict[str, Any]] = []
    failures: list[str] = []

    for name in requested:
        if name not in local_names:
            continue
        adapter = registry.get(name)
        if not getattr(adapter, "requires_gpu", False):
            continue
        try:
            if not adapter.is_loaded:
                adapter.load()
            ready.append(
                {
                    "name": name,
                    "device": adapter.config.device,
                    "adapter": adapter.__class__.__name__,
                }
            )
        except Exception as e:
            failures.append(f"{name} ({adapter.config.device}): {e}")

    if failures:
        joined = "; ".join(failures)
        if strict:
            raise RuntimeError(f"Local GPU SBM preload failed: {joined}")
        logger.warning(f"Local GPU SBM preload failed: {joined}")

    if ready:
        logger.info(
            "Local GPU SBM preload ready: "
            + ", ".join(f"{item['name']}@{item['device']}" for item in ready)
        )
    else:
        logger.warning("Local GPU SBM preload found no loadable local GPU adapters")
    return ready


def configured_default_config(name: str, default_config: SBMConfig) -> SBMConfig:
    """Apply TimeClaw's SBM runtime device mapping to one adapter config."""
    try:
        from timeclaw.core.config import TimeClawConfig

        runtime_cfg = TimeClawConfig.load().sbm
        device = (
            runtime_cfg.devices.get(name)
            or runtime_cfg.devices.get(default_config.model_name)
            or runtime_cfg.device
        )
        if not device:
            return default_config
        return default_config.model_copy(update={"device": device})
    except Exception as e:
        logger.debug(f"SBM config load failed for {name}: {e}")
        return default_config


def get_global_registry() -> SBMRegistry:
    """Get the global SBM registry singleton."""
    return _global_registry


def auto_register(server_url: str = "http://localhost:8100") -> int:
    """Auto-discover and register available SBM adapters.

    If the SBM HTTP server is running, registers HTTP client adapters
    for all models served there (zero local GPU needed).  Otherwise
    falls back to local adapters.
    """
    count = 0
    registry = get_global_registry()

    try:
        from timeclaw.sbm.http_client import is_server_available, get_server_models
        if is_server_available(server_url):
            from timeclaw.sbm.http_client import SBMHttpClient
            remote_models = get_server_models(server_url)
            for model_name in remote_models:
                client = SBMHttpClient(server_url=server_url, target_model=model_name)
                registry._instances[model_name] = client
                registry._factories[model_name] = SBMHttpClient
                count += 1
            logger.info(f"SBM auto-register: {count} models via HTTP server ({server_url})")
            return count
    except Exception as e:
        logger.debug(f"SBM HTTP server not available: {e}")

    count = len(register_local_adapters())

    logger.info(f"SBM auto-register: {count} local adapters available")
    return count
