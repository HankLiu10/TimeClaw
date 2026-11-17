"""Domain knowledge manager: loads and serves domain-specific knowledge files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class DomainKnowledge:
    """Manages domain knowledge stored as Markdown files (similar to OpenClaw's SOUL.md)."""

    def __init__(self, knowledge_dir: str | Path = ".timeclaw/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}

    def load_all(self) -> dict[str, str]:
        """Load all .md files from the knowledge directory."""
        self._cache.clear()
        for f in sorted(self.knowledge_dir.glob("*.md")):
            try:
                self._cache[f.stem] = f.read_text()
            except Exception as e:
                logger.warning(f"Failed to load knowledge file {f}: {e}")
        logger.info(f"Loaded {len(self._cache)} domain knowledge files")
        return dict(self._cache)

    def get(self, name: str) -> str | None:
        """Get a specific knowledge document by name (without .md extension)."""
        if name not in self._cache:
            path = self.knowledge_dir / f"{name}.md"
            if path.exists():
                self._cache[name] = path.read_text()
        return self._cache.get(name)

    def add(self, name: str, content: str) -> None:
        """Add or update a knowledge document."""
        path = self.knowledge_dir / f"{name}.md"
        path.write_text(content)
        self._cache[name] = content

    def list_documents(self) -> list[str]:
        return [f.stem for f in sorted(self.knowledge_dir.glob("*.md"))]

    def get_context_for_task(self, task_type: str) -> str:
        """Build a context string from relevant knowledge for a given task type."""
        relevant = []
        for name, content in self._cache.items():
            if task_type in name.lower() or "general" in name.lower():
                relevant.append(f"## {name}\n{content}")
        return "\n\n".join(relevant) if relevant else ""
