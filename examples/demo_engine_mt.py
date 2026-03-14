"""EngineMT-QA Demo: Industrial multimodal scenario.

Sensor time-series + maintenance logs (text) + fault images -> fault prediction.
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
    print("TimeClaw Demo: EngineMT-QA (Industrial Multimodal)")
    print("=" * 60)

    discover_and_register()

    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    vibration = (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.01 * t ** 1.5
        + np.random.randn(1000) * 0.3
    )
    vibration[800:850] += np.random.randn(50) * 5

    text_data = [
        {"timestamp": 200, "text": "Routine maintenance completed. All systems normal."},
        {"timestamp": 500, "text": "Vibration levels slightly elevated. Monitoring closely."},
        {"timestamp": 780, "text": "Warning: abnormal vibration detected in bearing unit B3."},
        {"timestamp": 850, "text": "Emergency maintenance: bearing replacement performed."},
    ]

    images = [
        {"timestamp": 780, "path": "data/engine/bearing_wear_780.jpg"},
        {"timestamp": 850, "path": "data/engine/bearing_fault_850.jpg"},
    ]

    detector = ModalityDetector()
    report = detector.detect(ts_data=vibration, text_data=text_data, images=images)
    print(f"\n[Modality Detection] {report['modalities']} (multimodal={report['is_multimodal']})")

    aligner = ModalityAligner(tolerance=50)
    alignment = aligner.align(
        ts_timestamps=np.arange(len(vibration)),
        text_data=text_data,
        images=images,
    )
    print(f"[Alignment] {alignment['n_text_matched']} texts, {alignment['n_image_matched']} images")

    try:
        sentiment_tool = get_tool_instance("sentiment")
        print("\n[Sentiment Analysis]")
        for doc in text_data:
            result = await sentiment_tool.execute(text=doc["text"])
            if result.success:
                s = result.data["sentiments"][0]
                print(f"  t={doc['timestamp']}: {s['polarity']} (score={s['score']})")
    except KeyError:
        pass

    try:
        event_tool = get_tool_instance("text_to_event")
        all_text = "\n".join(d["text"] for d in text_data)
        result = await event_tool.execute(
            text=all_text, event_types=["fault", "maintenance", "alert"]
        )
        if result.success:
            print(f"\n[Events] {result.data['n_events']} events extracted")
    except KeyError:
        pass

    env = EnvironmentSnapshot()
    env.set_data(vibration)
    env.set_text_data(text_data)
    env.set_images(images)

    task = TaskSpec(
        task_id="engine_mt_demo",
        task_type="anomaly",
        description="Detect bearing faults using sensor data, logs, and images.",
    )
    envelope = TaskEnvelope(task=task, environment=env)
    print(f"\n[TEA] Modalities: {env.modalities}, Shape: {env.data_shape}")

    fusion = ModalityFusion(strategy="late")
    fused = fusion.fuse({
        "timeseries": {"prediction": "anomaly", "confidence": 0.85},
        "text": {"prediction": "fault", "confidence": 0.9},
        "image": {"prediction": "degraded", "confidence": 0.75},
    })
    print(f"[Fusion] Confidence: {fused['confidence']}")
    print("\nDemo complete.")


if __name__ == "__main__":
    asyncio.run(main())
