"""Unit tests for the classical preprocessing pipeline (src.data.pipeline).

Contract under test:
- synthetic simulation reproduces schema, base rate, missingness, seed;
- CSV loading validates schema and target;
- the preprocessor outputs exactly ``n_components`` angles in [0, pi];
- ``get_loan_data`` returns {-1, +1} labels, stratified splits, and
  correctly sized train batches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    LoanPreprocessor,
    get_loan_data,
    load_loan_data,
    simulate_loan_data,
)

SEED = 7


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestSimulateLoanData:
    def test_schema_and_shape(self) -> None:
        frame = simulate_loan_data(n_samples=200, random_state=SEED)
        assert list(frame.columns) == [*DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN]
        assert len(frame) == 200

    def test_default_rate_is_approximately_requested(self) -> None:
        rate = 0.2
        frame = simulate_loan_data(n_samples=4_000, default_rate=rate, random_state=SEED)
        actual = frame[TARGET_COLUMN].mean()
        assert rate - 0.02 <= actual <= rate + 0.02

    def test_missing_values_injected_in_features_only(self) -> None:
        frame = simulate_loan_data(
            n_samples=2_000, missing_fraction=0.05, random_state=SEED
        )
        total_missing = int(frame[list(DEFAULT_FEATURE_COLUMNS)].isna().sum().sum())
        assert 100 < total_missing < 600  # ~5% of 2000 x 4 cells
        assert not frame[TARGET_COLUMN].isna().any()

    def test_seed_reproducibility(self) -> None:
        a = simulate_loan_data(n_samples=100, random_state=SEED)
        b = simulate_loan_data(n_samples=100, random_state=SEED)
        pd.testing.assert_frame_equal(a, b)

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError):
            simulate_loan_data(n_samples=0)
        with pytest.raises(ValueError):
            simulate_loan_data(default_rate=1.5)
        with pytest.raises(ValueError):
            simulate_loan_data(missing_fraction=0.9)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _write_csv(frame: pd.DataFrame, path) -> str:
    csv_path = path / "loans.csv"
    frame.to_csv(csv_path, index=False)
    return str(csv_path)


class TestLoadLoanData:
    def test_loads_valid_csv(self, tmp_path) -> None:
        frame = simulate_loan_data(n_samples=150, missing_fraction=0.0, random_state=SEED)
        loaded = load_loan_data(_write_csv(frame, tmp_path))
        assert list(loaded.columns) == [*DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN]
        assert len(loaded) == 150

    def test_missing_required_column_raises(self, tmp_path) -> None:
        frame = simulate_loan_data(n_samples=50, random_state=SEED).drop(
            columns=["debt_ratio"]
        )
        with pytest.raises(ValueError, match="debt_ratio"):
            load_loan_data(_write_csv(frame, tmp_path))

    def test_non_binary_target_raises(self, tmp_path) -> None:
        frame = simulate_loan_data(n_samples=50, random_state=SEED)
        frame[TARGET_COLUMN] = 2
        with pytest.raises(ValueError, match="default"):
            load_loan_data(_write_csv(frame, tmp_path))

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_loan_data(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestLoanPreprocessor:
    def test_output_shape_and_angle_range(self) -> None:
        frame = simulate_loan_data(n_samples=300, random_state=SEED)
        pre = LoanPreprocessor(n_components=4)
        angles = pre.fit_transform(frame)
        assert angles.shape == (300, 4)
        assert angles.min() >= 0.0
        assert angles.max() <= np.pi

    def test_explained_variance_available_after_fit(self) -> None:
        frame = simulate_loan_data(n_samples=200, random_state=SEED)
        pre = LoanPreprocessor(n_components=4)
        assert pre.explained_variance_ratio is None
        pre.fit(frame)
        ratio = pre.explained_variance_ratio
        assert ratio is not None
        assert ratio.shape == (4,)
        assert ratio.sum() <= 1.0 + 1e-9

    def test_transform_requires_fit_first(self) -> None:
        pre = LoanPreprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            pre.transform(np.zeros((3, 4)))

    def test_out_of_distribution_samples_are_clipped(self) -> None:
        frame = simulate_loan_data(n_samples=300, random_state=SEED)
        pre = LoanPreprocessor(n_components=4).fit(frame)
        extreme = pd.DataFrame(
            np.full((1, 4), 1e6), columns=list(DEFAULT_FEATURE_COLUMNS)
        )
        angles = pre.transform(extreme)
        assert angles.shape == (1, 4)
        assert angles.min() >= 0.0
        assert angles.max() <= np.pi

    def test_more_components_than_features_raises(self) -> None:
        frame = simulate_loan_data(n_samples=100, random_state=SEED)
        with pytest.raises(ValueError, match="PCA components"):
            LoanPreprocessor(n_components=5).fit(frame)

    def test_fitted_on_train_applied_to_test_is_deterministic(self) -> None:
        frame = simulate_loan_data(n_samples=200, random_state=SEED)
        train, test = frame.iloc[:160], frame.iloc[160:]
        pre_a = LoanPreprocessor().fit(train)
        pre_b = LoanPreprocessor().fit(train)
        np.testing.assert_array_equal(pre_a.transform(test), pre_b.transform(test))


# ---------------------------------------------------------------------------
# get_loan_data: the public entry point
# ---------------------------------------------------------------------------


class TestGetLoanData:
    def test_split_shapes_and_component_count(self) -> None:
        data = get_loan_data(batch_size=32, n_samples=500, random_state=SEED)
        assert data.X_train.shape == (400, 4)
        assert data.X_test.shape == (100, 4)
        assert data.n_train == 400
        assert data.n_test == 100

    def test_all_angles_within_rotation_range(self) -> None:
        data = get_loan_data(batch_size=32, n_samples=500, random_state=SEED)
        for matrix in (data.X_train, data.X_test):
            assert matrix.min() >= 0.0
            assert matrix.max() <= np.pi

    def test_labels_are_pm1_with_both_classes_present(self) -> None:
        data = get_loan_data(batch_size=32, n_samples=500, random_state=SEED)
        for labels in (data.y_train, data.y_test):
            assert set(np.unique(labels).tolist()) == {-1, 1}

    def test_stratification_preserves_class_balance(self) -> None:
        rate = 0.2
        data = get_loan_data(
            batch_size=32, n_samples=2_000, default_rate=rate, random_state=SEED
        )
        for labels in (data.y_train, data.y_test):
            positive = float((labels == 1).mean())
            assert rate - 0.05 <= positive <= rate + 0.05

    def test_train_batches_partition_the_training_set(self) -> None:
        batch_size = 32
        data = get_loan_data(
            batch_size=batch_size, n_samples=210, random_state=SEED
        )
        assert data.n_train == 168
        assert len(data.train_batches) == 6  # ceil(168 / 32)
        for X_batch, y_batch in data.train_batches:
            assert 1 <= len(X_batch) <= batch_size
            assert len(X_batch) == len(y_batch)
            assert X_batch.shape[1] == 4
        total = sum(len(y) for _, y in data.train_batches)
        assert total == data.n_train
        # Every training sample appears in exactly one batch.
        stacked_y = np.concatenate([y for _, y in data.train_batches])
        assert sorted(stacked_y.tolist()) == sorted(data.y_train.tolist())

    def test_batch_size_validation(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            get_loan_data(batch_size=0, n_samples=100)

    def test_seed_reproducibility(self) -> None:
        a = get_loan_data(batch_size=16, n_samples=300, random_state=SEED)
        b = get_loan_data(batch_size=16, n_samples=300, random_state=SEED)
        np.testing.assert_array_equal(a.X_train, b.X_train)
        np.testing.assert_array_equal(a.y_train, b.y_train)
        np.testing.assert_array_equal(a.X_test, b.X_test)
        np.testing.assert_array_equal(a.y_test, b.y_test)
        for (xa, ya), (xb, yb) in zip(a.train_batches, b.train_batches):
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)

    def test_csv_source_roundtrip(self, tmp_path) -> None:
        frame = simulate_loan_data(
            n_samples=300, missing_fraction=0.0, random_state=SEED
        )
        csv_path = _write_csv(frame, tmp_path)
        data = get_loan_data(batch_size=16, csv_path=csv_path, random_state=SEED)
        assert data.n_train + data.n_test == 300
        assert set(np.unique(data.y_train).tolist()) == {-1, 1}
        assert data.X_train.min() >= 0.0
        assert data.X_train.max() <= np.pi

    def test_single_class_dataset_raises(self, tmp_path) -> None:
        frame = simulate_loan_data(n_samples=100, random_state=SEED)
        frame[TARGET_COLUMN] = 0
        csv_path = _write_csv(frame, tmp_path)
        with pytest.raises(ValueError, match="both classes"):
            get_loan_data(csv_path=csv_path)


