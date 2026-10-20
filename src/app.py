"""Streamlit web UI: quantum loan default risk explorer.

An interactive front end for the hybrid quantum-classical loan default
predictor:

1. **Sidebar** -- sliders/inputs for a (dummy) borrower's metrics: credit
   score, annual income, debt-to-income ratio, employment history.
2. **Classical preprocessing** -- the raw borrower record is pushed through
   the exact ``LoanPreprocessor`` used at training time (median imputer,
   ``StandardScaler``, PCA to 4 components, min-max scaling to ``[0, pi]``),
   reconstructed deterministically from the training seed.
3. **Quantum forward pass** -- the optimized circuit parameters are loaded
   from the ``.npy`` checkpoint (``src/utils/weights.npy`` /
   ``bias.npy``) and a single ``variational_classifier`` evaluation is run
   on ``default.qubit``.
4. **Gauge** -- the biased expectation value ``<Z_0> + b`` in ``[-1, 1]``
   is mapped linearly to a 0-100% "Quantum-Calculated Default Risk"
   rating and rendered as an SVG gauge.
5. **Circuit view** -- a live ``qml.draw`` schematic of the 4-qubit circuit
   with the borrower's actual rotation angles, plus a stage-by-stage
   breakdown of the layers.

Run from the repository root::

    .venv/bin/streamlit run src/app.py

The module keeps every ``streamlit`` call inside :func:`main` (and the
cached loader), so the pure helpers (:func:`risk_percent`,
:func:`risk_band`, :func:`build_borrower_frame`,
:func:`load_model_resources`) remain importable and unit-testable without
a Streamlit runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import src.*` work no matter where streamlit launches the script
# from (streamlit puts the script's own directory on sys.path, not the
# repository root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pennylane as qml
import streamlit as st

from src.data.pipeline import DEFAULT_FEATURE_COLUMNS, get_loan_data
from src.models.quantum_vqc import (
    N_QUBITS,
    quantum_classifier,
    variational_classifier,
)
from src.utils.checkpoint import DEFAULT_CHECKPOINT_DIR, load_checkpoint

__all__ = [
    "DEFAULT_BORROWER",
    "TRAINING_SEED",
    "build_borrower_frame",
    "gauge_svg",
    "load_model_resources",
    "risk_band",
    "risk_percent",
]

# ---------------------------------------------------------------------------
# Training-run constants: the app reconstructs the *exact* preprocessing
# statistics the checkpoint was trained against, so these must mirror the
# `python -m src.train` defaults.
# ---------------------------------------------------------------------------

#: Seed of the default training run (data generation, split, init).
TRAINING_SEED: int = 42

#: Number of synthetic samples of the default training run.
TRAINING_N_SAMPLES: int = 800

#: Default base rate of the training data.
TRAINING_DEFAULT_RATE: float = 0.2

#: Missing-cell fraction of the training data.
TRAINING_MISSING_FRACTION: float = 0.05

#: Held-out test fraction of the training run.
TRAINING_TEST_SIZE: float = 0.2

#: Sidebar defaults for the dummy borrower.
DEFAULT_BORROWER: dict[str, float] = {
    "credit_score": 650.0,
    "income": 65_000.0,
    "debt_ratio": 0.35,
    "employment_years": 8.0,
}


# ---------------------------------------------------------------------------
# Pure helpers (no streamlit calls -- unit-testable)
# ---------------------------------------------------------------------------


def risk_percent(score: float) -> float:
    """Map a biased expectation value to a 0-100% default risk rating.

    The hybrid model outputs ``f = <Z_0> + b`` where ``<Z_0>`` is bounded
    to ``[-1, 1]`` and ``b`` is the trained bias. The linear map

    ``risk = 100 * clip((f + 1) / 2, 0, 1)``

    sends ``-1 -> 0%`` (strongly "repay") and ``+1 -> 100%`` (strongly
    "default"), matching the label convention (+1 = default).

    Parameters
    ----------
    score:
        The biased quantum classifier output (may lie slightly outside
        ``[-1, 1]`` when the bias is large).

    Returns
    -------
    float
        Default risk percentage in ``[0, 100]``.
    """
    clipped = min(1.0, max(-1.0, float(score)))
    return 100.0 * (clipped + 1.0) / 2.0


def risk_band(percent: float) -> tuple[str, str]:
    """Bucket a risk percentage into a human-readable band.

    Parameters
    ----------
    percent:
        Risk percentage in ``[0, 100]``.

    Returns
    -------
    tuple[str, str]
        ``(label, hex_color)`` with one of the bands Low / Moderate /
        Elevated / High.
    """
    if percent < 35.0:
        return "Low", "#16a34a"
    if percent < 55.0:
        return "Moderate", "#eab308"
    if percent < 75.0:
        return "Elevated", "#f97316"
    return "High", "#dc2626"


def build_borrower_frame(
    credit_score: float,
    income: float,
    debt_ratio: float,
    employment_years: float,
) -> pd.DataFrame:
    """Assemble a single-row borrower frame in the pipeline schema.

    Parameters
    ----------
    credit_score:
        Borrower credit score, typically 300-850.
    income:
        Annual income in dollars.
    debt_ratio:
        Debt-to-income ratio in ``[0, 1]``.
    employment_years:
        Years of employment history.

    Returns
    -------
    pd.DataFrame
        One row with exactly the ``DEFAULT_FEATURE_COLUMNS`` schema, ready
        for :meth:`LoanPreprocessor.transform`.
    """
    return pd.DataFrame(
        {
            "income": [float(income)],
            "credit_score": [float(credit_score)],
            "debt_ratio": [float(debt_ratio)],
            "employment_years": [float(employment_years)],
        }
    )


def load_model_resources(checkpoint_dir: str | Path | None = None) -> dict:
    """Load every artifact the inference UI needs.

    Two deterministic steps:

    1. Rebuild the training dataset via :func:`get_loan_data` with the
       training constants (same seed -> identical imputer medians, scaler
       statistics, PCA basis, and min-max bounds) and keep its fitted
       :class:`~src.data.pipeline.LoanPreprocessor`.
    2. Load the optimized circuit parameters from the checkpoint directory
       (``weights.npy`` / ``bias.npy``; defaults to ``src/utils/``).

    Parameters
    ----------
    checkpoint_dir:
        Optional checkpoint directory override for tests.

    Returns
    -------
    dict
        Keys: ``preprocessor`` (fitted LoanPreprocessor), ``weights``,
        ``bias``, ``n_layers``, ``explained_variance_ratio``,
        ``train_accuracy`` reference values from the reconstructed split.

    Raises
    ------
    FileNotFoundError
        If the checkpoint files do not exist (train first with
        ``python -m src.train``).
    """
    data = get_loan_data(
        batch_size=32,
        n_samples=TRAINING_N_SAMPLES,
        default_rate=TRAINING_DEFAULT_RATE,
        missing_fraction=TRAINING_MISSING_FRACTION,
        test_size=TRAINING_TEST_SIZE,
        random_state=TRAINING_SEED,
    )
    weights, bias = load_checkpoint(
        checkpoint_dir if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    )
    return {
        "preprocessor": data.preprocessor,
        "weights": weights,
        "bias": bias,
        "n_layers": int(np.asarray(weights).shape[0]),
        "explained_variance_ratio": data.explained_variance_ratio,
    }


def gauge_svg(percent: float, label: str, color: str) -> str:
    """Render a semicircular SVG gauge for a risk percentage.

    Parameters
    ----------
    percent:
        Risk percentage in ``[0, 100]``; drives the needle and arc.
    label:
        Human-readable band label shown under the value.
    color:
        Hex color for the value arc, needle, and label.

    Returns
    -------
    str
        Self-contained SVG markup (no external assets).
    """
    percent = min(100.0, max(0.0, float(percent)))
    cx, cy, radius = 170.0, 170.0, 120.0
    track_length = np.pi * radius
    value_length = track_length * percent / 100.0
    # Needle: 0% points left (180 deg), 100% points right (0 deg).
    needle_angle = np.deg2rad(180.0 - 1.8 * percent)
    needle_len = radius - 18.0
    nx = cx + needle_len * np.cos(needle_angle)
    ny = cy - needle_len * np.sin(needle_angle)

    def tick(angle_percent: float, caption: str) -> str:
        angle = np.deg2rad(180.0 - 1.8 * angle_percent)
        tx = cx + (radius + 14) * np.cos(angle)
        ty = cy - (radius + 14) * np.sin(angle) + 4.0
        return (
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="11" fill="#64748b" '
            f'text-anchor="middle" font-family="sans-serif">{caption}</text>'
        )

    ticks = "".join(
        tick(p, c) for p, c in ((0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%"))
    )
    return f"""
<svg viewBox="0 0 340 205" width="100%" xmlns="http://www.w3.org/2000/svg">
  <path d="M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}"
        fill="none" stroke="#e2e8f0" stroke-width="22" stroke-linecap="round"/>
  <path d="M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}"
        fill="none" stroke="{color}" stroke-width="22" stroke-linecap="round"
        stroke-dasharray="{value_length:.2f} {track_length:.2f}"/>
  <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
        stroke="#334155" stroke-width="4" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="7" fill="#334155"/>
  <text x="{cx}" y="{cy - 46}" font-size="34" font-weight="700" fill="#0f172a"
        text-anchor="middle" font-family="sans-serif">{percent:.1f}%</text>
  <text x="{cx}" y="{cy - 22}" font-size="14" font-weight="600" fill="{color}"
        text-anchor="middle" font-family="sans-serif">{label} risk</text>
  {ticks}
</svg>
"""


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading quantum model...")
def _cached_load_model_resources() -> dict:
    """Cache the (deterministic) model artifacts across reruns."""
    return load_model_resources()


def main() -> None:
    """Render the quantum loan default risk explorer."""
    st.set_page_config(
        page_title="Quantum Loan Default Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Hybrid Quantum-Classical Loan Default Predictor")
    st.caption(
        "A 4-qubit variational quantum classifier (PennyLane) inside a "
        "scikit-learn preprocessing pipeline, trained on synthetic loan "
        "data. Educational demo -- not financial advice."
    )

    try:
        model = _cached_load_model_resources()
    except FileNotFoundError as error:
        st.error(
            "**Checkpoint not found.** Train the model first so the app "
            "can load the optimized circuit parameters:\n\n"
            "```\n.venv/bin/python -m src.train\n```\n\n"
            f"Details: {error}"
        )
        st.stop()

    preprocessor = model["preprocessor"]
    weights, bias = model["weights"], model["bias"]
    n_layers = model["n_layers"]

    # --- Sidebar: borrower metrics -----------------------------------
    with st.sidebar:
        st.header("Borrower metrics")
        credit_score = st.slider(
            "Credit score", min_value=300, max_value=850,
            value=int(DEFAULT_BORROWER["credit_score"]), step=1,
            help="Standard FICO-style range 300-850.",
        )
        income = st.slider(
            "Annual income ($)", min_value=10_000, max_value=500_000,
            value=int(DEFAULT_BORROWER["income"]), step=5_000,
            help="Gross annual income in dollars.",
        )
        debt_ratio = st.slider(
            "Debt-to-income ratio", min_value=0.0, max_value=1.0,
            value=float(DEFAULT_BORROWER["debt_ratio"]), step=0.01,
            help="Monthly debt payments divided by gross monthly income.",
        )
        employment_years = st.slider(
            "Employment history (years)", min_value=0, max_value=40,
            value=int(DEFAULT_BORROWER["employment_years"]), step=1,
            help="Years in current employment.",
        )
        st.divider()
        st.markdown(
            "**Model**\n\n"
            f"- {N_QUBITS} qubits, {n_layers} variational layers\n"
            f"- {n_layers * N_QUBITS * 3} circuit parameters + 1 bias\n"
            f"- trained for 20 epochs (Nesterov momentum)"
        )
        with st.expander("How preprocessing is reproduced"):
            st.markdown(
                "The imputer, scaler, PCA basis, and min-max bounds are "
                f"re-fitted deterministically from training seed "
                f"`{TRAINING_SEED}` -- identical statistics to the training "
                "run, so the gauge uses the exact PCA matrix the weights "
                "were optimized against."
            )

    # --- Classical preprocessing + quantum forward pass ---------------
    borrower = build_borrower_frame(
        credit_score, income, debt_ratio, employment_years
    )
    angles = preprocessor.transform(borrower)[0]

    z0 = float(quantum_classifier(weights, angles))
    score = float(variational_classifier(weights, bias, angles))
    percent = risk_percent(score)
    band, color = risk_band(percent)
    prediction = "DEFAULT" if score >= 0.0 else "REPAY"

    # --- Gauge + metrics ----------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Quantum-calculated default risk")
        st.markdown(gauge_svg(percent, band, color), unsafe_allow_html=True)
        st.markdown(
            f"Decision rule: the biased expectation "
            f"`<Z0> + b = {score:+.4f}` is "
            f"{'**non-negative** -> predicted default' if score >= 0 else '**negative** -> predicted repay'} "
            f"(label convention: +1 = default, -1 = repay)."
        )
    with right:
        st.subheader("Forward pass readout")
        col1, col2 = st.columns(2)
        col1.metric("Readout ⟨Z₀⟩", f"{z0:+.4f}", help="Pauli-Z expectation on qubit 0, bounded to [-1, 1]")
        col2.metric("Trained bias b", f"{float(bias):+.4f}")
        col1.metric("Biased score ⟨Z₀⟩+b", f"{score:+.4f}")
        col2.metric("Risk rating", f"{percent:.1f}% ({band})")
        st.markdown(
            "The risk rating maps the biased expectation linearly onto "
            "`[0%, 100%]` via `(score + 1) / 2`."
        )
        st.bar_chart(
            pd.DataFrame(
                {"Rotation angle (rad)": angles},
                index=[f"PC{i + 1}" for i in range(len(angles))],
            )
        )
        st.caption(
            "PCA components fed to the circuit as RY rotation angles "
            f"(explained variance: "
            f"{np.sum(model['explained_variance_ratio']):.0%} across "
            f"{len(angles)} components)."
        )

    # --- Circuit view ---------------------------------------------------
    st.divider()
    st.subheader("Under the hood: the 4-qubit circuit")
    st.markdown(
        "Every prediction runs this exact circuit on the "
        "`default.qubit` simulator -- drawn live with **this borrower's** "
        "rotation angles."
    )
    st.code(
        qml.draw(quantum_classifier)(weights, angles), language=None
    )
    stage_table = pd.DataFrame(
        {
            "Stage": [
                "1. State preparation",
                f"2. Variational layer (x{n_layers})",
                "3. Entanglement",
                "4. Readout",
            ],
            "Operations": [
                "RY(x_i) on qubit i",
                "RX, RY, RZ per qubit",
                "CNOT ring 0-1-2-3-0",
                "PauliZ expectation on qubit 0",
            ],
            "Parameters": [
                "none (data)",
                f"{n_layers * N_QUBITS * 3} trainable",
                "none (fixed)",
                "none (fixed)",
            ],
            "Purpose": [
                "Angle-embed the 4 PCA features on the Bloch sphere",
                "Trainable Euler rotations mix each qubit's state",
                "Spread feature interactions across all qubits",
                "Bounded score in [-1, 1]; sign decides default vs repay",
            ],
        }
    )
    st.table(stage_table)
    with st.expander("Classical preprocessing chain (before the circuit)"):
        st.markdown(
            "1. **Impute** -- median fill for missing values (this record "
            "is complete)\n"
            "2. **Standardize** -- zero mean / unit variance per feature\n"
            "3. **PCA** -- compress the 4 standardized features to 4 "
            "components (the qubit input width)\n"
            "4. **Scale to [0, pi]** -- min-max mapping so each component "
            "is a valid Bloch-sphere rotation angle"
        )

    st.divider()
    st.caption(
        "Synthetic training data (seed 42, ~20% default rate) -- "
        "educational quantum ML demonstration, not a credit decisioning "
        "tool."
    )


if __name__ == "__main__":
    main()
