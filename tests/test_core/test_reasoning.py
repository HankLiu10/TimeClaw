from __future__ import annotations

import pytest

from timeclaw.core.evidence import EvidenceLog
from timeclaw.core.reasoning import EvidenceIterator


@pytest.mark.asyncio
async def test_evidence_iterator_raises_on_invalid_json():
    class _Resp:
        parsed = None

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    iterator = EvidenceIterator(llm=_LLM(), tools={})

    with pytest.raises(ValueError, match="invalid JSON"):
        await iterator.iterate(
            goal="forecast",
            evidence_log=EvidenceLog(),
            context={"modalities": ["timeseries"]},
        )


@pytest.mark.asyncio
async def test_evidence_iterator_raises_on_missing_tool():
    class _Resp:
        parsed = {
            "reasoning": "use a missing tool",
            "tool": "missing_tool",
            "args": {},
            "covers_predicates": [],
            "done": False,
        }

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    iterator = EvidenceIterator(llm=_LLM(), tools={})

    with pytest.raises(KeyError, match="missing_tool"):
        await iterator.iterate(
            goal="forecast",
            evidence_log=EvidenceLog(),
            context={"modalities": ["timeseries"]},
        )
