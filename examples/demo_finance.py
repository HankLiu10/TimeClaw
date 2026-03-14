"""FinMultiTime Demo: Financial multimodal scenario.

Price time-series + news (text) + financial statements -> trend analysis.
"""

from __future__ import annotations

import asyncio

import numpy as np

from timeclaw.core.tea_protocol import TaskSpec, EnvironmentSnapshot, TaskEnvelope
from timeclaw.multimodal.detector import ModalityDetector
from timeclaw.multimodal.aligner import ModalityAligner
from timeclaw.multimodal.fusion import ModalityFusion
from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance


async def main() -> None:
    print("=" * 60)
    print("TimeClaw Demo: FinMultiTime (Financial Multimodal)")
    print("=" * 60)

    discover_and_register()

    np.random.seed(7)
    n_days = 252
    returns = np.random.randn(n_days) * 0.02
    returns[100:110] -= 0.05
    returns[200:210] += 0.04
    price = 100 * np.exp(np.cumsum(returns))
    volume = np.abs(np.random.randn(n_days) * 1e6 + 5e6)
    volume[100:110] *= 3
    ts_data = np.column_stack([price, volume])

    text_data = [
        {"timestamp": 50, "text": "Company reports strong Q1 earnings, beating expectations by 15%."},
        {"timestamp": 100, "text": "BREAKING: Company misses Q2 earnings. Revenue down 8% YoY."},
        {"timestamp": 105, "text": "Analysts downgrade stock to sell following disappointing earnings."},
        {"timestamp": 150, "text": "New product launch receives positive reviews."},
        {"timestamp": 200, "text": "FDA approves new drug. Stock surges in after-hours."},
        {"timestamp": 230, "text": "Annual report shows improved margins and reduced debt."},
    ]

    images = [
        {"timestamp": 50, "path": "data/finance/q1_statement.png"},
        {"timestamp": 100, "path": "data/finance/q2_statement.png"},
    ]

    detector = ModalityDetector()
    report = detector.detect(ts_data=ts_data, text_data=text_data, images=images)
    print(f"\n[Detection] {report['modalities']}")

    aligner = ModalityAligner(tolerance=10)
    alignment = aligner.align(np.arange(n_days), text_data, images, window_size=5)
    print(f"[Alignment] {alignment['n_text_matched']} texts, {alignment['n_image_matched']} images")

    try:
        sentiment_tool = get_tool_instance("sentiment")
        print("\n[News Sentiment]")
        for doc in text_data:
            r = await sentiment_tool.execute(text=doc["text"])
            if r.success:
                s = r.data["sentiments"][0]
                print(f"  Day {doc['timestamp']:3d}: {s['polarity']:8s} (score={s['score']:+.3f})")
    except KeyError:
        pass

    try:
        kw_tool = get_tool_instance("keyword_extract")
        all_news = " ".join(d["text"] for d in text_data)
        r = await kw_tool.execute(text=all_news, top_k=8)
        if r.success:
            kws = [k["keyword"] for k in r.data["keywords"]]
            print(f"\n[Keywords] {', '.join(kws)}")
    except KeyError:
        pass

    env = EnvironmentSnapshot()
    env.set_data(ts_data)
    env.set_text_data(text_data)
    env.set_images(images)

    task = TaskSpec(
        task_id="finance_demo",
        task_type="forecast",
        description="Predict stock trend using price, news, and financials.",
        horizon=20,
        constraints=[
            {"type": "non_negative"},
            {"type": "temporal", "history_last": float(price[-1]), "max_jump": float(price[-1] * 0.1)},
        ],
    )
    envelope = TaskEnvelope(task=task, environment=env)
    print(f"\n[TEA] Modalities: {env.modalities}, Last price: ${price[-1]:.2f}")

    fusion = ModalityFusion(strategy="late")
    fused = fusion.fuse({
        "timeseries": {"prediction": price[-1] * 1.02, "confidence": 0.6},
        "text": {"prediction": "bullish", "confidence": 0.75},
        "image": {"prediction": "improving", "confidence": 0.55},
    })
    print(f"[Fusion] Confidence: {fused['confidence']}")
    print("\nDemo complete.")


if __name__ == "__main__":
    asyncio.run(main())
