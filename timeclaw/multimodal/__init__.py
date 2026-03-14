"""Multimodal alignment, detection, and fusion for TimeClaw."""

from timeclaw.multimodal.detector import ModalityDetector
from timeclaw.multimodal.aligner import ModalityAligner
from timeclaw.multimodal.confidence import ModalityConfidenceEstimator, estimate_all_confidences
from timeclaw.multimodal.fusion import ModalityFusion

__all__ = [
    "ModalityDetector",
    "ModalityAligner",
    "ModalityConfidenceEstimator",
    "ModalityFusion",
    "estimate_all_confidences",
]
