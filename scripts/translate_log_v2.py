"""Translate the entire timeclaw_full_dump.log to Chinese, chunk by chunk.

Splits the file into chunks of ~150 lines, sends each chunk to LLM for translation,
then reassembles. This ensures NOTHING is missed.
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

INPUT = Path("timeclaw_full_dump.log")
OUTPUT = Path("timeclaw_full_dump_cn.log")

AZURE_KEY = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
AZURE_BASE = "https://asure-test-openai-004.openai.azure.com/"
AZURE_VER = "2024-12-01-preview"

CHUNK_LINES = 150
MAX_CONCURRENT = 10

SYSTEM_PROMPT = """\
你是一个AI系统日志的英译中翻译器。你的任务是将下面这段日志中的所有英文内容翻译成简体中文。

## 严格规则

1. **翻译范围**：将所有英文句子、段落翻译成中文。包括但不限于：
   - LLM的system prompt内容
   - LLM的user prompt内容  
   - LLM的返回内容（JSON中的英文字符串值也要翻译）
   - 普通日志行中的英文描述
   - 工具调用参数中的英文描述

2. **不翻译的内容**（原样保留）：
   - 时间戳（如 11:59:06.012）
   - 日志级别标记（如 INFO、ERROR）
   - 方括号中的模块名（如 [Pipeline]、[manager]、[analyzer]）
   - 代码变量名、工具名（如 basic_stats、arima_forecast、stl_decompose）
   - JSON的键名（如 "task_type"、"fingerprint"）
   - 数字和数值
   - 模型名（如 azure/gpt-5.1）
   - 分隔线（如 ════、────）
   - Python对象表示（如 <timeclaw.llm.base.LLMBackend object at 0x...>）
   - 文件路径和URL
   - 数据集名称（如 weather_aligned_short）

3. **格式保留**：
   - 保持每一行的缩进和空格不变
   - 保持所有换行符不变
   - 保持Markdown格式标记（##、-、*、```等）
   - 保持JSON结构完整

4. **翻译风格**：
   - 简洁准确，不要添加额外解释
   - 专业术语首次出现时可以保留英文并括号注释，如 forecast(预测)
   - 只返回翻译后的日志内容，不要加任何前缀、后缀或说明文字

5. **特别注意**：
   - 输入数据中的原始文本数据（如新闻文章、天气报告等长段英文）也需要翻译
   - 如果某行已经是中文，保持不变
"""


async def translate_chunk(chunk: str, chunk_idx: int, total: int, sem: asyncio.Semaphore) -> str:
    import litellm

    if not chunk.strip():
        return chunk

    async with sem:
        for attempt in range(3):
            try:
                print(f"  翻译块 {chunk_idx+1}/{total} ({len(chunk)} chars)...", flush=True)
                resp = await litellm.acompletion(
                    model="azure/gpt-5.1",
                    api_base=AZURE_BASE,
                    api_key=AZURE_KEY,
                    api_version=AZURE_VER,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0.1,
                    max_tokens=16384,
                    timeout=300.0,
                )
                result = resp.choices[0].message.content
                if result:
                    print(f"  ✓ 块 {chunk_idx+1}/{total} 完成", flush=True)
                    return result
                else:
                    print(f"  ⚠ 块 {chunk_idx+1} 返回空，重试...", file=sys.stderr)
            except Exception as e:
                print(f"  ✗ 块 {chunk_idx+1} 失败 (尝试 {attempt+1}/3): {e}", file=sys.stderr)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))

        print(f"  ✗✗ 块 {chunk_idx+1} 翻译失败，保留原文", file=sys.stderr)
        return chunk


def split_into_chunks(lines: list[str], chunk_size: int) -> list[str]:
    """Split lines into chunks, trying to break at natural boundaries."""
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))

        # Try to find a natural break point (task separator, LLM block boundary)
        if end < len(lines):
            best_break = end
            for j in range(end, max(i + chunk_size // 2, i + 1), -1):
                line = lines[j]
                if ("════" in line or "────" in line or
                    "=================" in line or
                    line.strip() == "" or
                    "LLM 调用详情" in line):
                    best_break = j
                    break
            end = best_break

        chunk = "\n".join(lines[i:end])
        chunks.append(chunk)
        i = end

    return chunks


async def main():
    print(f"📖 读取 {INPUT}...")
    raw = INPUT.read_text(encoding="utf-8")
    lines = raw.split("\n")
    print(f"   共 {len(lines)} 行, {len(raw)} 字符")

    print(f"✂️  分块 (每块约 {CHUNK_LINES} 行)...")
    chunks = split_into_chunks(lines, CHUNK_LINES)
    print(f"   共 {len(chunks)} 个块")

    print(f"🌐 开始翻译 (并发数: {MAX_CONCURRENT})...")
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        translate_chunk(chunk, i, len(chunks), sem)
        for i, chunk in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks)

    print("📝 合并结果...")
    output = "\n".join(results)
    OUTPUT.write_text(output, encoding="utf-8")
    output_lines = output.count("\n") + 1
    print(f"✅ 翻译完成！保存到 {OUTPUT} ({output_lines} 行)")


if __name__ == "__main__":
    os.environ.setdefault("AZURE_API_KEY", AZURE_KEY)
    os.environ.setdefault("AZURE_API_BASE", AZURE_BASE)
    os.environ.setdefault("AZURE_API_VERSION", AZURE_VER)
    asyncio.run(main())
