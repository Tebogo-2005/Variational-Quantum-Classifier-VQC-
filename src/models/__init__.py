"""Model zoo: quantum circuits and classical baselines.

Responsibilities:

- Data re-uploading feature maps (Perez-Salinas et al., 2021): input features
  are re-encoded as rotation angles at every circuit layer.
- Hardware-efficient variational ansaetze: trainable Euler rotations plus a
  CNOT ring entangler between re-uploading layers.
- Hybrid classifier: a PennyLane QNode wrapped in an sklearn-style
  ``fit`` / ``predict`` / ``predict_proba`` interface.
- Classical baselines (logistic regression, random forest) for benchmarking.

See ``docs/ARCHITECTURE.md`` (Section 4) for the circuit-level specification.
"""
