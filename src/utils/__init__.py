"""Shared utilities.

Responsibilities:

- Reproducibility: one seed argument fanned out to numpy, scikit-learn, and
  PennyLane (including shot-based device seeds).
- Metrics: average precision (PR-AUC) as the primary metric, plus ROC-AUC,
  F1, Brier score, and confusion-matrix reporting.
- Configuration: frozen dataclass describing the training run.
- Logging helpers for experiment tracking.

This package must stay free of dependencies on ``src.data`` and
``src.models``.
"""
