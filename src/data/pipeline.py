"""Classical data preprocessing pipeline for the hybrid loan-default VQC.

This module turns raw tabular loan data into quantum-ready rotation angles.

Processing stages (all but the final split statistic are learned on the
training split only, to prevent information leakage):

1. **Ingestion** -- simulate a reproducible loan-risk dataset (income,
   credit score, debt ratio, employment history) or load one from CSV.
2. **Cleaning** -- median imputation of missing values.
3. **Normalization** -- ``StandardScaler`` (zero mean, unit variance).
4. **Dimensionality reduction** -- PCA compressed to exactly
   ``n_components`` (default 4) dimensions, matching the qubit budget.
5. **Angle scaling** -- min-max scaling of the PCA components to
   ``[0, pi]`` so each component is a valid Bloch-sphere rotation angle
   for the data re-uploading feature map.

Label convention: the raw binary target ``default`` in {0, 1} is mapped to
{-1, +1}, where **+1 encodes default** (the positive / risk class) and
-1 encodes repaid. This pairs naturally with a Pauli-Z readout whose sign
is the predicted class.

See ``docs/ARCHITECTURE.md`` (Sections 3 and 4) for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__all__ = [
    "ANGLE_HIGH",
    "ANGLE_LOW",
    "DEFAULT_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "LoanDataset",
    "LoanPreprocessor",
    "build_angle_pipeline",
    "get_loan_data",
    "load_loan_data",
    "simulate_loan_data",
]

#: Feature columns consumed by the pipeline (exactly the user-facing schema).
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "income",
    "credit_score",
    "debt_ratio",
    "employment_years",
)

#: Binary target column; 1 = default, 0 = repaid.
TARGET_COLUMN: str = "default"

#: Inclusive bounds of the rotation-angle range after final scaling.
ANGLE_LOW: float = 0.0
ANGLE_HIGH: float = float(np.pi)

#: Type aliases for array-typed public interfaces.
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_feature_matrix(
    features: pd.DataFrame | FloatArray,
) -> pd.DataFrame | FloatArray:
    """Normalize user input into a matrix the sklearn pipeline accepts.

    Parameters
    ----------
    features:
        Either a dataframe (from which exactly ``DEFAULT_FEATURE_COLUMNS``
        is selected; extra columns are ignored) or a 2-D float array whose
        column order matches ``DEFAULT_FEATURE_COLUMNS``.

    Returns
    -------
    pd.DataFrame | FloatArray
        A feature matrix ready for imputation / scaling.

    Raises
    ------
    ValueError
        If a dataframe is passed without all required feature columns, or
        if an array is passed that is not two-dimensional.
    """
    if isinstance(features, pd.DataFrame):
        missing = [c for c in DEFAULT_FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(
                f"Input dataframe is missing required feature columns: {missing}. "
                f"Expected at least {list(DEFAULT_FEATURE_COLUMNS)}."
            )
        return features.loc[:, list(DEFAULT_FEATURE_COLUMNS)]
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Feature array must be 2-D (n_samples, n_features); got shape "
            f"{matrix.shape}."
        )
    return matrix


def simulate_loan_data(
    n_samples: int = 1_000,
    default_rate: float = 0.2,
    missing_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate a tabular loan-risk dataset with realistic correlations.

    A latent creditworthiness factor ``r ~ N(0, 1)`` drives every feature
    and the default decision, so the data contains genuine (nonlinear-ish)
    learnable signal rather than noise. A configurable fraction of feature
    cells is set to NaN to exercise the imputation stage.

    Parameters
    ----------
    n_samples:
        Number of loan applications to generate. Must be at least 10.
    default_rate:
        Approximate fraction of loans that default, in (0, 1). Achieved by
        thresholding the latent risk score at its ``1 - default_rate``
        quantile.
    missing_fraction:
        Probability that any given feature cell is missing (NaN), in
        [0, 0.5). The target column is never missing.
    random_state:
        Seed for the ``numpy.random.default_rng`` generator; identical
        seeds produce identical datasets.

    Returns
    -------
    pd.DataFrame
        Frame with columns ``income`` (annual, dollars, log-normal),
        ``credit_score`` (clipped to [300, 850]), ``debt_ratio`` (clipped
        to [0, 1]), ``employment_years`` (clipped to [0, 40]), and
        ``default`` in {0, 1}. Feature columns may contain NaN.

    Raises
    ------
    ValueError
        If any argument is outside its documented range.
    """
    if n_samples < 10:
        raise ValueError(f"n_samples must be >= 10; got {n_samples}.")
    if not 0.0 < default_rate < 1.0:
        raise ValueError(f"default_rate must lie in (0, 1); got {default_rate}.")
    if not 0.0 <= missing_fraction < 0.5:
        raise ValueError(
            f"missing_fraction must lie in [0, 0.5); got {missing_fraction}."
        )

    rng = np.random.default_rng(random_state)
    creditworthiness = rng.standard_normal(n_samples)

    income = np.exp(
        11.0 + 0.5 * creditworthiness + 0.3 * rng.standard_normal(n_samples)
    )
    credit_score = np.clip(
        650.0 + 80.0 * creditworthiness + 25.0 * rng.standard_normal(n_samples),
        300.0,
        850.0,
    )
    debt_ratio = np.clip(
        0.35 - 0.10 * creditworthiness + 0.08 * rng.standard_normal(n_samples),
        0.0,
        1.0,
    )
    employment_years = np.clip(
        8.0 + 3.0 * creditworthiness + 4.0 * rng.standard_normal(n_samples),
        0.0,
        40.0,
    )

    # Default risk rises as creditworthiness falls; threshold at the
    # (1 - default_rate) quantile to hit the requested base rate.
    risk = -creditworthiness + 0.25 * rng.standard_normal(n_samples)
    threshold = np.quantile(risk, 1.0 - default_rate)
    default = (risk > threshold).astype(np.int64)

    frame = pd.DataFrame(
        {
            "income": income,
            "credit_score": credit_score,
            "debt_ratio": debt_ratio,
            "employment_years": employment_years,
        }
    )

    # Inject missing values into feature columns only (never the target).
    if missing_fraction > 0.0:
        mask = rng.random((n_samples, len(DEFAULT_FEATURE_COLUMNS))) < missing_fraction
        for column_index, column in enumerate(DEFAULT_FEATURE_COLUMNS):
            frame.loc[mask[:, column_index], column] = np.nan

    frame[TARGET_COLUMN] = default
    return frame


def load_loan_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a loan-risk dataset from a CSV file.

    The file must contain the columns listed in ``DEFAULT_FEATURE_COLUMNS``
    plus the binary target ``default``. Extra columns are ignored. Feature
    cells that cannot be parsed as numbers are coerced to NaN and handled
    by the imputation stage.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with a header row.

    Returns
    -------
    pd.DataFrame
        Frame with exactly the schema produced by
        :func:`simulate_loan_data` (feature columns numeric, possibly with
        NaN; ``default`` in {0, 1} with no missing values).

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not point to an existing file.
    ValueError
        If required columns are missing, or the target column contains
        values other than {0, 1} (or has missing entries).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Loan data CSV not found: {path}")

    frame = pd.read_csv(path)

    required = (*DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {missing}. "
            f"Expected columns: {list(required)}."
        )

    for column in DEFAULT_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain only the values "
            f"0 (repaid) and 1 (default), with no missing entries."
        )
    frame[TARGET_COLUMN] = target.astype(np.int64)

    return frame.loc[:, list(required)]


def build_angle_pipeline(
    n_components: int = 4,
    angle_high: float = ANGLE_HIGH,
) -> Pipeline:
    """Build the leakage-safe classical preprocessing pipeline.

    The returned (unfitted) pipeline executes, in order:

    1. ``SimpleImputer(strategy="median")`` -- fill missing cells.
    2. ``StandardScaler`` -- zero mean / unit variance per feature.
    3. ``PCA(n_components)`` -- compress to the qubit input width.
    4. ``MinMaxScaler(feature_range=(0, angle_high))`` -- map every
       component into a valid rotation-angle interval.

    Parameters
    ----------
    n_components:
        Exact number of PCA components to keep. Must be >= 1.
    angle_high:
        Upper bound of the final angle range; the lower bound is always 0.
        Defaults to pi for Bloch-sphere rotations.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The unfitted preprocessing pipeline.

    Raises
    ------
    ValueError
        If ``n_components`` < 1 or ``angle_high`` <= 0.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}.")
    if angle_high <= 0.0:
        raise ValueError(f"angle_high must be > 0; got {angle_high}.")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("angle_scaler", MinMaxScaler(feature_range=(ANGLE_LOW, angle_high))),
        ]
    )


class LoanPreprocessor:
    """Fit-once, transform-many wrapper around the angle pipeline.

    All statistics (imputation medians, feature means/variances, PCA basis,
    min-max bounds) are learned in :meth:`fit` -- call it on the *training*
    split only, then reuse the fitted instance for every other split. This
    makes the preprocessing leakage-safe by construction.

    Parameters
    ----------
    n_components:
        Exact number of PCA output components (default 4, matching the
        4-qubit re-uploading feature map that consumes two angles per
        qubit).
    angle_high:
        Upper bound of the output angle range (default pi).

    Examples
    --------
    >>> pre = LoanPreprocessor(n_components=4)
    >>> X_train_angles = pre.fit_transform(train_frame)
    >>> X_test_angles = pre.transform(test_frame)
    """

    def __init__(
        self,
        n_components: int = 4,
        angle_high: float = ANGLE_HIGH,
    ) -> None:
        """Store configuration; nothing is fitted yet."""
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1; got {n_components}.")
        if angle_high <= 0.0:
            raise ValueError(f"angle_high must be > 0; got {angle_high}.")
        self.n_components = n_components
        self.angle_high = float(angle_high)
        self._pipeline: Pipeline | None = None

    @property
    def pipeline(self) -> Pipeline | None:
        """The fitted sklearn pipeline, or ``None`` before :meth:`fit`."""
        return self._pipeline

    @property
    def explained_variance_ratio(self) -> FloatArray | None:
        """Per-component explained variance ratio, or ``None`` before fit."""
        if self._pipeline is None:
            return None
        pca: PCA = self._pipeline.named_steps["pca"]
        return pca.explained_variance_ratio_.copy()

    def fit(self, features: pd.DataFrame | FloatArray) -> LoanPreprocessor:
        """Learn all preprocessing statistics from training features.

        Parameters
        ----------
        features:
            Training features as a dataframe (required columns selected
            automatically) or a 2-D array in ``DEFAULT_FEATURE_COLUMNS``
            order.

        Returns
        -------
        LoanPreprocessor
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the number of features or samples is smaller than
            ``n_components``.
        """
        matrix = _as_feature_matrix(features)
        n_features = np.asarray(matrix).shape[1]
        if n_features < self.n_components:
            raise ValueError(
                f"Cannot keep {self.n_components} PCA components from only "
                f"{n_features} input features."
            )
        if np.asarray(matrix).shape[0] < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} samples to fit PCA; got "
                f"{np.asarray(matrix).shape[0]}."
            )

        self._pipeline = build_angle_pipeline(self.n_components, self.angle_high)
        self._pipeline.fit(matrix)
        return self

    def transform(self, features: pd.DataFrame | FloatArray) -> FloatArray:
        """Apply the fitted pipeline and clip to valid rotation angles.

        Out-of-distribution samples can fall outside the min-max bounds
        learned on the training split; those values are clipped to
        ``[0, angle_high]`` so every output is guaranteed to be a valid
        rotation angle (rotations are periodic, but the bounded invariant
        keeps downstream feature maps well-conditioned and testable).

        Parameters
        ----------
        features:
            Features in the same schema as passed to :meth:`fit`.

        Returns
        -------
        FloatArray
            Array of shape ``(n_samples, n_components)`` with all values in
            ``[0, angle_high]``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "LoanPreprocessor.transform() called before fit(); call "
                "fit(features) on the training split first."
            )
        matrix = _as_feature_matrix(features)
        angles = self._pipeline.transform(matrix)
        return np.clip(angles, ANGLE_LOW, self.angle_high)

    def fit_transform(self, features: pd.DataFrame | FloatArray) -> FloatArray:
        """Fit on ``features`` and transform it in one call.

        Parameters
        ----------
        features:
            Training features (dataframe or 2-D array).

        Returns
        -------
        FloatArray
            Array of shape ``(n_samples, n_components)`` with all values in
            ``[0, angle_high]``.
        """
        self.fit(features)
        return self.transform(features)


@dataclass(frozen=True)
class LoanDataset:
    """Quantum-ready train/test splits for the loan-default classifier.

    Attributes
    ----------
    X_train:
        Training rotation angles, shape ``(n_train, n_components)``, all
        values in ``[0, pi]``.
    y_train:
        Training labels in {-1, +1}; +1 = default, -1 = repaid.
    X_test:
        Held-out rotation angles, same shape contract as ``X_train``.
    y_test:
        Held-out labels in {-1, +1}.
    train_batches:
        Mini-batches of the (shuffled) training set: a tuple of
        ``(X_batch, y_batch)`` pairs, each with at most ``batch_size``
        samples. The final batch may be smaller.
    feature_names:
        Names of the original feature columns behind the pipeline.
    explained_variance_ratio:
        Per-PCA-component explained variance ratio of the fitted pipeline.
    preprocessor:
        The fitted :class:`LoanPreprocessor` (fit on the training split
        only). Exposed so inference-time callers -- e.g. the Streamlit app
        -- can transform raw borrower records into rotation angles with
        exactly the statistics the model was trained against.
    """

    X_train: FloatArray
    y_train: IntArray
    X_test: FloatArray
    y_test: IntArray
    train_batches: tuple[tuple[FloatArray, IntArray], ...]
    feature_names: tuple[str, ...]
    explained_variance_ratio: FloatArray
    preprocessor: LoanPreprocessor | None = None

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        """Number of held-out test samples."""
        return int(self.X_test.shape[0])

    def __post_init__(self) -> None:
        """Validate shape consistency of the stored splits."""
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError("X_train and X_test must be 2-D arrays.")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(
                "X_train and X_test must share the same number of "
                f"components; got {self.X_train.shape[1]} and "
                f"{self.X_test.shape[1]}."
            )
        if len(self.y_train) != self.n_train or len(self.y_test) != self.n_test:
            raise ValueError(
                "Feature and label counts must match per split; got "
                f"({self.n_train}, {len(self.y_train)}) train and "
                f"({self.n_test}, {len(self.y_test)}) test."
            )


def get_loan_data(
    batch_size: int = 32,
    csv_path: str | Path | None = None,
    n_samples: int = 1_000,
    default_rate: float = 0.2,
    missing_fraction: float = 0.05,
    test_size: float = 0.2,
    n_components: int = 4,
    random_state: int = 42,
) -> LoanDataset:
    """Build quantum-ready loan-default train/test splits.

    High-level entry point: loads (or simulates) the loan dataset, splits
    stratified by class, fits the preprocessing pipeline on the training
    split only, converts labels to {-1, +1}, and pre-slices the shuffled
    training set into mini-batches ready for a variational training loop.

    Parameters
    ----------
    batch_size:
        Maximum number of samples per training mini-batch. Must be >= 1.
    csv_path:
        Optional path to a loan CSV (see :func:`load_loan_data`). When
        ``None`` (default), :func:`simulate_loan_data` generates the data.
    n_samples:
        Number of synthetic samples (ignored when ``csv_path`` is given).
    default_rate:
        Approximate default base rate of the synthetic data, in (0, 1).
    missing_fraction:
        Fraction of missing feature cells in the synthetic data.
    test_size:
        Fraction of samples held out for testing, in (0, 1).
    n_components:
        Exact number of PCA components / rotation angles per sample.
    random_state:
        Single seed controlling data generation, splitting, and batch
        shuffling; identical arguments yield identical outputs.

    Returns
    -------
    LoanDataset
        Splits with ``X`` in ``[0, pi]`` of shape ``(n, n_components)``,
        labels in {-1, +1} (+1 = default), and shuffled train batches.

    Raises
    ------
    ValueError
        If arguments are out of range, the dataset has fewer than two
        classes, or features are insufficient for ``n_components``.

    Examples
    --------
    >>> data = get_loan_data(batch_size=32, n_samples=500, random_state=0)
    >>> data.X_train.shape
    (400, 4)
    >>> sorted(set(data.y_train.tolist()))
    [-1, 1]
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must lie in (0, 1); got {test_size}.")

    if csv_path is not None:
        frame = load_loan_data(csv_path)
    else:
        frame = simulate_loan_data(
            n_samples=n_samples,
            default_rate=default_rate,
            missing_fraction=missing_fraction,
            random_state=random_state,
        )

    features = frame.loc[:, list(DEFAULT_FEATURE_COLUMNS)]
    labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int64)

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(
            "Dataset must contain both classes (0 = repaid, 1 = default) "
            f"to build a classifier; found only {classes.tolist()}."
        )

    # Split BEFORE preprocessing so every statistic is learned on train only.
    features_train, features_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    preprocessor = LoanPreprocessor(n_components=n_components)
    X_train = preprocessor.fit_transform(features_train)
    X_test = preprocessor.transform(features_test)

    # Map {0, 1} -> {-1, +1}: default (1) becomes +1, repaid (0) becomes -1.
    y_train_pm = np.where(y_train == 1, 1, -1).astype(np.int64)
    y_test_pm = np.where(y_test == 1, 1, -1).astype(np.int64)

    # Shuffle the training split with a seeded generator, then slice into
    # mini-batches (final batch may be smaller than batch_size).
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(y_train_pm))
    X_shuffled = X_train[order]
    y_shuffled = y_train_pm[order]
    train_batches = tuple(
        (X_shuffled[start : start + batch_size], y_shuffled[start : start + batch_size])
        for start in range(0, len(y_shuffled), batch_size)
    )

    return LoanDataset(
        X_train=X_train,
        y_train=y_train_pm,
        X_test=X_test,
        y_test=y_test_pm,
        train_batches=train_batches,
        feature_names=DEFAULT_FEATURE_COLUMNS,
        explained_variance_ratio=preprocessor.explained_variance_ratio,  # type: ignore[arg-type]
        preprocessor=preprocessor,
    )
