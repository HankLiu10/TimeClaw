"""SBMRegistry: register, discover, and retrieve SBM model adapters."""

from __future__ import annotations

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


def get_global_registry() -> SBMRegistry:
    """Get the global SBM registry singleton."""
    return _global_registry


def auto_register() -> int:
    """Auto-discover and register available SBM adapters."""
    count = 0
    registry = get_global_registry()

    try:
        from timeclaw.sbm.chronos_adapter import ChronosAdapter, CHRONOS_DEFAULT_CONFIG
        registry.register("chronos", ChronosAdapter, CHRONOS_DEFAULT_CONFIG)
        count += 1
    except ImportError:
        logger.debug("Chronos adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.moirai_adapter import MoiraiAdapter, MOIRAI_DEFAULT_CONFIG
        registry.register("moirai", MoiraiAdapter, MOIRAI_DEFAULT_CONFIG)
        count += 1
    except ImportError:
        logger.debug("Moirai adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.timesfm_adapter import TimesFMAdapter, TIMESFM_DEFAULT_CONFIG
        registry.register("timesfm", TimesFMAdapter, TIMESFM_DEFAULT_CONFIG)
        count += 1
    except ImportError:
        logger.debug("TimesFM adapter not available (missing dependencies)")

    try:
        from timeclaw.sbm.timer_adapter import TimerAdapter, TIMER_DEFAULT_CONFIG
        registry.register("timer", TimerAdapter, TIMER_DEFAULT_CONFIG)
        count += 1
    except ImportError:
        logger.debug("Timer adapter not available (missing dependencies)")

    logger.info(f"SBM auto-register: {count} adapters available")
    return count
