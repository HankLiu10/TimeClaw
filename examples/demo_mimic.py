"""MIMIC-IV Demo: Medical multimodal scenario.

Vital signs + clinical notes + lab reports -> patient outcome prediction.
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
    print("TimeClaw Demo: MIMIC-IV (Medical Multimodal)")
    print("=" * 60)

    discover_and_register()

    np.random.seed(123)
    n = 2880  # 48 hours at minute-level
    t = np.arange(n)
    hr = 75 + 10 * np.sin(2 * np.pi * t / 1440) + np.random.randn(n) * 3
    hr[2000:2200] += 30
    bp = 120 + 5 * np.sin(2 * np.pi * t / 720) + np.random.randn(n) * 5
    spo2 = 98 - 0.001 * t + np.random.randn(n) * 0.5
    spo2[2000:2200] -= 5
    vitals = np.column_stack([hr, bp, spo2])

    text_data = [
        {"timestamp": 0, "text": "Patient admitted with shortness of breath. History of COPD."},
        {"timestamp": 720, "text": "Stable overnight. O2 saturation above 95%."},
        {"timestamp": 1440, "text": "Increased dyspnea reported. Chest X-ray ordered."},
        {"timestamp": 2000, "text": "ALERT: Tachycardia and desaturation detected."},
        {"timestamp": 2200, "text": "BiPAP initiated. Patient stabilizing."},
    ]

    images = [
        {"timestamp": 1440, "path": "data/mimic/chest_xray.jpg"},
        {"timestamp": 2100, "path": "data/mimic/abg_report.jpg"},
    ]

    detector = ModalityDetector()
    report = detector.detect(ts_data=vitals, text_data=text_data, images=images)
    print(f"\n[Detection] {report['modalities']}")

    aligner = ModalityAligner(tolerance=120)
    alignment = aligner.align(np.arange(n), text_data, images, window_size=60)
    print(f"[Alignment] {alignment['n_text_matched']} texts, {alignment['n_image_matched']} images")

    try:
        ner_tool = get_tool_instance("ner")
        for doc in text_data[:2]:
            r = await ner_tool.execute(text=doc["text"])
            if r.success and r.data["entities"][0]:
                print(f"[NER t={doc['timestamp']}] {len(r.data['entities'][0])} entities")
    except KeyError:
        pass

    env = EnvironmentSnapshot()
    env.set_data(vitals)
    env.set_text_data(text_data)
    env.set_images(images)

    task = TaskSpec(
        task_id="mimic_demo",
        task_type="forecast",
        description="Predict patient deterioration from vitals and clinical notes.",
        horizon=60,
        constraints=[
            {"type": "physical", "domain": "heart_rate", "min": 30, "max": 250},
            {"type": "physical", "domain": "spo2", "min": 0, "max": 100},
        ],
    )
    envelope = TaskEnvelope(task=task, environment=env)
    print(f"\n[TEA] Modalities: {env.modalities}, Shape: {env.data_shape}")

    fusion = ModalityFusion(strategy="late")
    fused = fusion.fuse({
        "timeseries": {"prediction": "deterioration", "confidence": 0.82},
        "text": {"prediction": "deterioration", "confidence": 0.88},
        "image": {"prediction": "mild_abnormality", "confidence": 0.65},
    })
    print(f"[Fusion] Confidence: {fused['confidence']}")
    print("\nDemo complete.")


if __name__ == "__main__":
    asyncio.run(main())
