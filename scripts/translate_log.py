"""Translate timeclaw_full_dump.log English content to Chinese.

Strategy: Parse the log into segments, deduplicate identical English text blocks,
translate unique blocks via Azure GPT-5.1, then reassemble.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import os
from pathlib import Path

INPUT = Path("timeclaw_full_dump.log")
OUTPUT = Path("timeclaw_full_dump_cn.log")

AZURE_KEY = "5jDLlQ5ccmal2OAOchrqcatcKMr4FfqQbOD1n0YUwkIFCwdxcI5aJQQJ99ALACYeBjFXJ3w3AAABACOGZLN4"
AZURE_BASE = "https://asure-test-openai-004.openai.azure.com/"
AZURE_VER = "2024-12-01-preview"


async def translate_text(text: str, sem: asyncio.Semaphore) -> str:
    import litellm

    if len(text.strip()) < 5:
        return text

    async with sem:
        try:
            resp = await litellm.acompletion(
                model="azure/gpt-5.1",
                api_base=AZURE_BASE,
                api_key=AZURE_KEY,
                api_version=AZURE_VER,
                messages=[
                    {"role": "system", "content": (
                        "你是一个AI系统日志翻译器。将以下英文内容翻译成简体中文。\n"
                        "严格规则：\n"
                        "1. 保留所有代码、JSON结构、数字、变量名、工具名(如arima_forecast, stl_decompose等)、模型名不翻译\n"
                        "2. 保留所有Markdown格式标记（##、-、*、```等）\n"
                        "3. 保留所有换行符和缩进，不要改变格式\n"
                        "4. 关键术语保留英文并在首次出现时括号注释中文，如 forecast(预测)、anomaly(异常)\n"
                        "5. 如果内容已经是中文或主要是JSON/数据，原样返回\n"
                        "6. 只返回翻译结果，不要加任何额外说明或前缀"
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=8192,
                timeout=180.0,
            )
            return resp.choices[0].message.content or text
        except Exception as e:
            print(f"  [翻译失败] {e}", file=sys.stderr)
            return text


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _is_mostly_english(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total = max(len(text.strip()), 1)
    return (ascii_alpha / total) > 0.35


def parse_llm_blocks(raw: str):
    """Parse the log into a list of (type, content) segments.
    
    Types: 'plain' for non-LLM lines, 'llm_block' for LLM call detail blocks.
    """
    lines = raw.split("\n")
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if "LLM 调用详情" in line:
            block_start = i
            if segments and segments[-1][0] == "plain":
                plain_lines = segments[-1][1].split("\n")
                if plain_lines and "────" in plain_lines[-1]:
                    sep = plain_lines.pop()
                    segments[-1] = ("plain", "\n".join(plain_lines))
                    block_start_lines = [sep]
                else:
                    block_start_lines = []
            else:
                block_start_lines = []

            block_start_lines.append(line)
            i += 1

            end_seps = 0
            while i < len(lines):
                block_start_lines.append(lines[i])
                if "────" in lines[i] and i > block_start:
                    end_seps += 1
                    if end_seps >= 2:
                        i += 1
                        break
                i += 1

            segments.append(("llm_block", "\n".join(block_start_lines)))
        else:
            if segments and segments[-1][0] == "plain":
                segments[-1] = ("plain", segments[-1][1] + "\n" + line)
            else:
                segments.append(("plain", line))
            i += 1

    return segments


def extract_translatable_parts(block_text: str):
    """Extract prompt messages and response from an LLM block.
    
    Returns list of (start_idx, end_idx, text, part_type) for translatable sections.
    """
    parts = []

    # Extract individual message contents
    msg_re = re.compile(
        r"(  --- \[(?:system|user)\] \(msg \d+/\d+\) ---\n)"
        r"(.*?)"
        r"(?=  --- \[(?:system|user)\]|  --- end of prompt ---|【LLM 原始返回】)",
        re.DOTALL,
    )
    for m in msg_re.finditer(block_text):
        header = m.group(1)
        content = m.group(2).rstrip("\n")
        if content and _is_mostly_english(content[:300]):
            parts.append({
                "header": header,
                "content": content,
                "start": m.start(2),
                "end": m.end(2),
                "type": "prompt_msg",
            })

    # Extract response
    resp_re = re.compile(r"(【LLM 原始返回】:\n)(.*?)(?=\n────)", re.DOTALL)
    rm = resp_re.search(block_text)
    if rm:
        resp_content = rm.group(2).rstrip("\n")
        if resp_content and _is_mostly_english(resp_content[:300]):
            parts.append({
                "header": rm.group(1),
                "content": resp_content,
                "start": rm.start(2),
                "end": rm.end(2),
                "type": "response",
            })

    return parts


async def main():
    print(f"📖 读取 {INPUT}...")
    raw = INPUT.read_text(encoding="utf-8")
    total_lines = raw.count("\n") + 1
    print(f"   共 {total_lines} 行")

    print("🔍 解析日志结构...")
    segments = parse_llm_blocks(raw)
    llm_count = sum(1 for t, _ in segments if t == "llm_block")
    print(f"   {len(segments)} 个段落, 其中 {llm_count} 个 LLM 调用块")

    # Collect all translatable parts and deduplicate
    all_parts = []
    unique_texts = {}  # hash -> text

    for seg_idx, (seg_type, seg_content) in enumerate(segments):
        if seg_type != "llm_block":
            continue
        parts = extract_translatable_parts(seg_content)
        for p in parts:
            h = _hash(p["content"])
            p["hash"] = h
            p["seg_idx"] = seg_idx
            all_parts.append(p)
            if h not in unique_texts:
                unique_texts[h] = p["content"]

    print(f"   找到 {len(all_parts)} 段可翻译文本, 去重后 {len(unique_texts)} 段唯一文本")

    # Translate unique texts
    print(f"🌐 开始翻译 {len(unique_texts)} 段唯一文本...")
    sem = asyncio.Semaphore(15)
    translations = {}

    items = list(unique_texts.items())

    async def do_one(h, text):
        t = text
        if len(t) > 8000:
            t = t[:8000] + "\n... [内容过长已截断]"
        result = await translate_text(t, sem)
        translations[h] = result

    batch_size = 15
    done = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await asyncio.gather(*[do_one(h, t) for h, t in batch])
        done += len(batch)
        print(f"   进度: {done}/{len(items)}")

    # Reassemble: replace original text with translations in each LLM block
    print("📝 重建翻译后的日志...")
    new_segments = []

    for seg_idx, (seg_type, seg_content) in enumerate(segments):
        if seg_type != "llm_block":
            new_segments.append(seg_content)
            continue

        translated_block = seg_content
        seg_parts = [p for p in all_parts if p["seg_idx"] == seg_idx]

        # Replace in reverse order to preserve positions
        for p in sorted(seg_parts, key=lambda x: x["start"], reverse=True):
            h = p["hash"]
            if h in translations and translations[h] != unique_texts[h]:
                old = p["content"]
                new = translations[h]
                translated_block = translated_block.replace(old, new, 1)

        new_segments.append(translated_block)

    result = "\n".join(new_segments)
    OUTPUT.write_text(result, encoding="utf-8")
    result_lines = result.count("\n") + 1
    print(f"✅ 翻译完成！保存到 {OUTPUT} ({result_lines} 行)")


if __name__ == "__main__":
    os.environ.setdefault("AZURE_API_KEY", AZURE_KEY)
    os.environ.setdefault("AZURE_API_BASE", AZURE_BASE)
    os.environ.setdefault("AZURE_API_VERSION", AZURE_VER)
    asyncio.run(main())
