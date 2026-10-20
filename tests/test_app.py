"""Unit tests for the Streamlit application (src.app).

Contracts under test (pure helpers only -- no Streamlit runtime needed):

- ``risk_percent`` linearly maps the biased expectation ``[-1, 1]`` onto
  ``[0, 100]`` and clips biased scores that leave the interval;
- ``risk_band`` buckets percentages into Low / Moderate / Elevated / High;
- ``build_borrower_frame`` emits the exact pipeline schema;
- ``load_model_resources`` reconstructs the training-time preprocessor and
  loads the checkpoint, and a single forward pass is deterministic and
  bounded (skipped when no checkpoint has been trained yet);
- ``gauge_svg`` produces self-contained SVG markup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app import (
    DEFAULT_BORROWER,
    build_borrower_frame,
    gauge_svg,
    load_model_resources,
    risk_band,
    risk_percent,
)
from src.data.pipeline import DEFAULT_FEATURE_COLUMNS, get_loan_data
from src.models.quantum_vqc import variational_classifier
from src.utils.checkpoint import BIAS_FILENAME, WEIGHTS_FILENAME

SEED = 7

#: Repository checkpoint directory used by the default training run.
REPO_CHECKPOINT_DIR = "src/utils"


# ---------------------------------------------------------------------------
# risk_percent
# ---------------------------------------------------------------------------


class TestRiskPercent:
    def test_endpoints_map_to_0_and_100(self) -> None:
        assert risk_percent(-1.0) == pytest.approx(0.0)
        assert risk_percent(1.0) == pytest.approx(100.0)

    def test_midpoint_maps_to_50(self) -> None:
        assert risk_percent(0.0) == pytest.approx(50.0)

    def test_linear_interpolation(self) -> None:
        assert risk_percent(0.5) == pytest.approx(75.0)
        assert risk_percent(-0.6) == pytest.approx(20.0)

    def test_biased_scores_are_clipped(self) -> None:
        assert risk_percent(1.7) == pytest.approx(100.0)
        assert risk_percent(-2.3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# risk_band
# ---------------------------------------------------------------------------


class TestRiskBand:
    def test_band_boundaries(self) -> None:
        assert risk_band(0.0)[0] == "Low"
        assert risk_band(34.9)[0] == "Low"
        assert risk_band(35.0)[0] == "Moderate"
        assert risk_band(54.9)[0] == "Moderate"
        assert risk_band(55.0)[0] == "Elevated"
        assert risk_band(74.9)[0] == "Elevated"
        assert risk_band(75.0)[0] == "High"
        assert risk_band(100.0)[0] == "High"

    def test_bands_carry_colors(self) -> None:
        for percent in (10.0, 40.0, 60.0, 90.0):
            label, color = risk_band(percent)
            assert label
            assert color.startswith("#") and len(color) == 7


# ---------------------------------------------------------------------------
# build_borrower_frame
# ---------------------------------------------------------------------------


class TestBuildBorrowerFrame:
    def test_emits_pipeline_schema_with_one_row(self) -> None:
        frame = build_borrower_frame(
            credit_score=720, income=120_000, debt_ratio=0.22,
            employment_years=11,
        )
        assert list(frame.columns) == list(DEFAULT_FEATURE_COLUMNS)
        assert len(frame) == 1

    def test_values_are_numeric(self) -> None:
        frame = build_borrower_frame(600, 50_000, 0.4, 5)
        assert frame.dtypes.map(pd.api.types.is_numeric_dtype).all()
        assert np.isfinite(frame.to_numpy(dtype=float)).all()

    def test_defaults_are_within_documented_ranges(self) -> None:
        frame = build_borrower_frame(
            DEFAULT_BORROWER["credit_score"],
            DEFAULT_BORROWER["income"],
            DEFAULT_BORROWER["debt_ratio"],
            DEFAULT_BORROWER["employment_years"],
        )
        row = frame.iloc[0]
        assert 300 <= row["credit_score"] <= 850
        assert 0.0 <= row["debt_ratio"] <= 1.0
        assert 0.0 <= row["employment_years"] <= 40


# ---------------------------------------------------------------------------
# gauge_svg
# ---------------------------------------------------------------------------


class TestGaugeSvg:
    def test_produces_self_contained_svg(self) -> None:
        markup = gauge_svg(62.5, "Elevated", "#f97316")
        assert markup.lstrip().startswith("<svg")
        assert markup.rstrip().endswith("</svg>")
        assert "62.5%" in markup
        assert "Elevated" in markup
        # Self-contained: no external asset references (the xmlns
        # namespace declaration is required and harmless).
        assert "src=" not in markup
        assert "href=" not in markup

    def test_extremes_render(self) -> None:
        for percent in (0.0, 100.0):
            markup = gauge_svg(percent, "band", "#000000")
            assert f"{percent:.1f}%" in markup


# ---------------------------------------------------------------------------
# load_model_resources + single forward pass (requires trained checkpoint)
# ---------------------------------------------------------------------------


def _checkpoint_available() -> bool:
    from pathlib import Path

    directory = Path(REPO_CHECKPOINT_DIR)
    return (directory / WEIGHTS_FILENAME).is_file() and (
        directory / BIAS_FILENAME
    ).is_file()


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason="no trained checkpoint in src/utils/ -- run python -m src.train",
)
class TestLoadModelResources:
    def test_returns_fitted_artifacts(self) -> None:
        resources = load_model_resources()
        preprocessor = resources["preprocessor"]
        assert preprocessor is not None
        assert preprocessor.pipeline is not None  # fitted
        assert resources["weights"].shape[1:] == (4, 3)
        assert resources["n_layers"] == resources["weights"].shape[0]
        assert resources["explained_variance_ratio"].shape == (4,)

    def test_preprocessor_matches_training_time_statistics(self) -> None:
        # Same seed -> the reconstructed preprocessor must be bit-identical
        # to the one inside a fresh get_loan_data call with training params.
        resources = load_model_resources()
        data = get_loan_data(
            batch_size=32,
            n_samples=800,
            default_rate=0.2,
            missing_fraction=0.05,
            test_size=0.2,
            random_state=42,
        )
        probe = build_borrower_frame(700, 90_000, 0.3, 10)
        np.testing.assert_array_equal(
            resources["preprocessor"].transform(probe),
            data.preprocessor.transform(probe),  # type: ignore[union-attr]
        )

    def test_single_forward_pass_is_deterministic_and_bounded(self) -> None:
        resources = load_model_resources()
        borrower = build_borrower_frame(580, 30_000, 0.55, 2)
        angles = resources["preprocessor"].transform(borrower)[0]
        assert angles.shape == (4,)
        first = float(
            variational_classifier(
                resources["weights"], resources["bias"], angles
            )
        )
        second = float(
            variational_classifier(
                resources["weights"], resources["bias"], angles
            )
        )
        assert first == second
        # Biased score leaves [-1, 1] by at most |bias|.
        bias = float(resources["bias"])
        assert -1.0 - abs(bias) <= first <= 1.0 + abs(bias)
        # ... and the risk mapping stays in [0, 100].
        assert 0.0 <= risk_percent(first) <= 100.0
