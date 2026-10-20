"""Data pipeline for the loan-default dataset.

Responsibilities:

- Raw data ingestion: synthetic loan generator (default, reproducible) and
  loaders for file / OpenML sources.
- Leakage-safe classical preprocessing: median imputation, one-hot encoding,
  standardization, PCA down to the quantum input width ``k = 2 * n_qubits``.
- Stratified train / validation / test splitting.

Design rule: every statistic (imputer, scaler, PCA basis) is fit on the
training split only and applied to the other splits afterwards.

See ``docs/ARCHITECTURE.md`` (Section 3) for the pipeline specification.
"""
