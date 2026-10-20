"""Hybrid Quantum-Classical Loan Default Predictor.

Root package of the VQC monorepo. Subpackages:

- ``data``:   dataset loading, leakage-safe preprocessing, and quantum-ready
              feature encoding (PCA to the qubit input width).
- ``models``: data re-uploading feature maps, variational ansaetze, the
              hybrid classifier wrapper, and classical baselines.
- ``utils``:  reproducibility (seeding), metrics, configuration, logging.

See ``docs/ARCHITECTURE.md`` for the full engineering specification.
"""

__version__ = "0.1.0"
