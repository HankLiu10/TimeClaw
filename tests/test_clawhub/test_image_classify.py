import pytest

from timeclaw.clawhub.image.image_classify import ImageClassifyTool


@pytest.mark.asyncio
async def test_image_classify_requires_llm_backend():
    tool = ImageClassifyTool()

    result = await tool.execute(images=[{"path": "/tmp/example.png"}])

    assert result.success is False
    assert "LLM backend is required" in str(result.error)


@pytest.mark.asyncio
async def test_image_classify_rejects_non_json_response():
    class _Resp:
        parsed = None
        text = "probably anomaly"

    class _LLM:
        async def complete_multimodal(self, **_kwargs):
            return _Resp()

    tool = ImageClassifyTool()
    result = await tool.execute(
        images=[{"path": "/tmp/example.png"}],
        llm=_LLM(),
    )

    assert result.success is False
    assert "non-JSON" in str(result.error)


@pytest.mark.asyncio
async def test_image_classify_rejects_missing_category_field():
    class _Resp:
        parsed = {"confidence": 0.9}
        text = None

    class _LLM:
        async def complete_multimodal(self, **_kwargs):
            return _Resp()

    tool = ImageClassifyTool()
    result = await tool.execute(
        images=[{"path": "/tmp/example.png"}],
        llm=_LLM(),
    )

    assert result.success is False
    assert "missing non-empty 'category'" in str(result.error)
