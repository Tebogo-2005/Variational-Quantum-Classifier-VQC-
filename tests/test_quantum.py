"""Unit tests for the quantum execution layer (src.models.quantum_vqc).

Contracts under test:
- ``state_preparation`` embeds exactly one RY rotation per qubit;
- ``vqc_layer`` applies RX/RY/RZ per qubit plus a 4-gate CNOT ring with
  wrap-around;
- ``quantum_classifier`` returns <Z_0> in [-1, 1], is deterministic, and
  its gate counts scale with the number of layers;
- ``variational_classifier`` adds the bias exactly and is differentiable
  (gradients match finite differences);
- ``initialize_weights`` is seeded, bounded, and trainable;
- end-to-end: angles from the data pipeline feed the circuit directly;
- output validation: the QNode returns a floating-point value strictly
  bounded in (-1, 1) for dummy data (see ``TestQNodeOutputValidation``).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from src.data.pipeline import get_loan_data
from src.models.quantum_vqc import (
    N_QUBITS,
    initialize_weights,
    quantum_classifier,
    state_preparation,
    variational_classifier,
    vqc_layer,
)

SEED = 7

#: Probe device mirroring the module device, used to draw sub-circuits.
_probe_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_probe_dev)
def _probe_prep(x):
    """Isolate state_preparation for structural inspection."""
    state_preparation(x)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(_probe_dev)
def _probe_layer(weights):
    """Isolate vqc_layer for structural inspection."""
    vqc_layer(weights)
    return qml.expval(qml.PauliZ(0))


def _wire_line(drawn: str, wire: int) -> str:
    """Extract the drawing line of a single wire from a circuit drawing."""
    for line in drawn.split("\n"):
        if line.startswith(f"{wire}:"):
            return line
    raise AssertionError(f"wire {wire} not found in drawing:\n{drawn}")


# ---------------------------------------------------------------------------
# state_preparation
# ---------------------------------------------------------------------------


class TestStatePreparation:
    def test_applies_exactly_one_ry_per_qubit(self) -> None:
        drawn = qml.draw(_probe_prep)(np.array([0.1, 0.4, 0.9, 2.2]))
        assert drawn.count("RY(") == N_QUBITS
        assert "RX(" not in drawn and "RZ(" not in drawn

    def test_zero_features_leave_qubits_in_ground_state(self) -> None:
        # RY(0) = identity, so <Z_0> must be exactly +1.
        assert _probe_prep(np.zeros(N_QUBITS)) == pytest.approx(1.0)

    def test_pi_feature_flips_readout_qubit(self) -> None:
        # RY(pi) on qubit 0 maps |0> -> |1>, so <Z_0> = -1.
        angles = np.zeros(N_QUBITS)
        angles[0] = np.pi
        assert _probe_prep(angles) == pytest.approx(-1.0)

    def test_invalid_feature_count_raises(self) -> None:
        with pytest.raises(ValueError, match="features"):
            state_preparation(np.zeros(3))


# ---------------------------------------------------------------------------
# vqc_layer
# ---------------------------------------------------------------------------


class TestVqcLayer:
    def test_applies_full_rotation_triple_per_qubit(self) -> None:
        weights = np.arange(12).reshape(N_QUBITS, 3) * 0.1
        drawn = qml.draw(_probe_layer)(weights)
        assert drawn.count("RX(") == N_QUBITS
        assert drawn.count("RY(") == N_QUBITS
        assert drawn.count("RZ(") == N_QUBITS

    def test_cnot_ring_entangles_all_qubits_with_wraparound(self) -> None:
        weights = np.zeros((N_QUBITS, 3))
        drawn = qml.draw(_probe_layer)(weights)
        assert drawn.count("●") == N_QUBITS  # one CNOT per qubit as control
        # Ring check. Wire 0 controls CNOT(0,1) ("\u256d\u25cf") and is the
        # target of the wrap-around CNOT(3,0) ("\u256dX", box opens downward
        # because the control wire 3 lies below). Wire 3 controls CNOT(3,0)
        # ("\u2570\u25cf", control below target) and is the target of
        # CNOT(2,3) ("\u2570X").
        line0 = _wire_line(drawn, 0)
        line3 = _wire_line(drawn, 3)
        assert "╭●" in line0 and "╭X" in line0
        assert "╰●" in line3 and "╰X" in line3

    def test_invalid_weights_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            vqc_layer(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# quantum_classifier
# ---------------------------------------------------------------------------


class TestQuantumClassifier:
    def test_output_lies_in_pm_one(self) -> None:
        rng = np.random.default_rng(SEED)
        for _ in range(10):
            weights = rng.uniform(-np.pi, np.pi, size=(3, N_QUBITS, 3))
            x = rng.uniform(0.0, np.pi, size=N_QUBITS)
            output = quantum_classifier(weights, x)
            assert -1.0 <= output <= 1.0

    def test_identity_configuration_returns_plus_one(self) -> None:
        # Zero features and zero weights: no rotations act, CNOTs leave
        # |0>^4 untouched, hence <Z_0> = +1 exactly.
        output = quantum_classifier(np.zeros((2, N_QUBITS, 3)), np.zeros(N_QUBITS))
        assert output == pytest.approx(1.0)

    def test_deterministic_for_fixed_inputs(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        x = np.array([0.3, 1.1, 2.0, 2.8])
        first = quantum_classifier(weights, x)
        second = quantum_classifier(weights, x)
        assert first == second

    def test_gate_counts_scale_with_layer_count(self) -> None:
        n_layers = 3
        weights = initialize_weights(n_layers, seed=SEED)
        x = np.full(N_QUBITS, 0.5)
        drawn = qml.draw(quantum_classifier)(weights, x)
        assert drawn.count("RX(") == n_layers * N_QUBITS
        assert drawn.count("RY(") == (n_layers + 1) * N_QUBITS  # + embedding
        assert drawn.count("RZ(") == n_layers * N_QUBITS
        assert drawn.count("●") == n_layers * N_QUBITS

    def test_additional_layers_change_the_output(self) -> None:
        x = np.array([0.4, 0.8, 1.5, 2.6])
        one_layer = initialize_weights(1, seed=SEED)
        two_layers = initialize_weights(2, seed=SEED + 1)
        two_layers[0] = one_layer[0]  # same first layer, extra second layer
        out1 = quantum_classifier(one_layer, x)
        out2 = quantum_classifier(two_layers, x)
        assert out1 != pytest.approx(out2)

    def test_invalid_weights_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            quantum_classifier(np.zeros((2, 3, 3)), np.zeros(N_QUBITS))

    def test_invalid_x_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="features"):
            quantum_classifier(initialize_weights(1, SEED), np.zeros(5))


# ---------------------------------------------------------------------------
# variational_classifier
# ---------------------------------------------------------------------------


class TestVariationalClassifier:
    def test_bias_is_added_exactly(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        x = np.array([0.2, 0.7, 1.3, 2.9])
        quantum_out = quantum_classifier(weights, x)
        assert variational_classifier(weights, 0.5, x) == pytest.approx(
            quantum_out + 0.5
        )

    def test_zero_bias_equals_quantum_output(self) -> None:
        weights = initialize_weights(1, seed=SEED)
        x = np.array([0.9, 1.4, 2.1, 3.0])
        assert variational_classifier(weights, 0.0, x) == pytest.approx(
            quantum_classifier(weights, x)
        )

    def test_gradients_are_finite_and_match_finite_differences(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        bias = pnp.array(0.3, requires_grad=True)
        x = np.array([0.1, 0.5, 1.2, 2.7])

        grad_weights, grad_bias = qml.grad(
            variational_classifier, argnums=(0, 1)
        )(weights, bias, x)

        assert grad_weights.shape == weights.shape
        assert np.isfinite(grad_weights).all()
        # d/db (quantum_out + b) = 1 exactly.
        assert grad_bias == pytest.approx(1.0)

        # Finite-difference spot check on weights[0, 0, 0].
        eps = 1e-6
        plus = np.array(weights, dtype=float)
        minus = np.array(weights, dtype=float)
        plus[0, 0, 0] += eps
        minus[0, 0, 0] -= eps
        fd = (
            variational_classifier(plus, float(bias), x)
            - variational_classifier(minus, float(bias), x)
        ) / (2 * eps)
        assert fd == pytest.approx(float(grad_weights[0, 0, 0]), abs=1e-4)

    def test_gradients_are_not_all_zero(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        bias = pnp.array(0.0, requires_grad=True)
        x = np.array([0.3, 0.9, 1.7, 2.4])
        grad_weights, _ = qml.grad(variational_classifier, argnums=(0, 1))(
            weights, bias, x
        )
        assert np.abs(np.asarray(grad_weights)).sum() > 0.0


# ---------------------------------------------------------------------------
# initialize_weights
# ---------------------------------------------------------------------------


class TestInitializeWeights:
    def test_shape_and_bounds(self) -> None:
        weights = initialize_weights(n_layers=3, seed=SEED)
        assert weights.shape == (3, N_QUBITS, 3)
        assert np.asarray(weights).min() >= -np.pi
        assert np.asarray(weights).max() <= np.pi

    def test_parameters_are_trainable(self) -> None:
        weights = initialize_weights(n_layers=1, seed=SEED)
        assert bool(weights.requires_grad) is True

    def test_seed_reproducibility(self) -> None:
        a = np.asarray(initialize_weights(2, seed=SEED))
        b = np.asarray(initialize_weights(2, seed=SEED))
        np.testing.assert_array_equal(a, b)
        c = np.asarray(initialize_weights(2, seed=SEED + 1))
        assert not np.array_equal(a, c)

    def test_invalid_layer_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            initialize_weights(n_layers=0)


# ---------------------------------------------------------------------------
# End-to-end: data pipeline -> quantum circuit
# ---------------------------------------------------------------------------


class TestDataPipelineIntegration:
    def test_pipeline_angles_feed_the_circuit(self) -> None:
        data = get_loan_data(batch_size=8, n_samples=100, random_state=SEED)
        weights = initialize_weights(2, seed=SEED)
        outputs = [
            float(variational_classifier(weights, 0.0, sample))
            for sample in data.X_train[:5]
        ]
        assert len(outputs) == 5
        assert all(-1.0 <= value <= 1.0 for value in outputs)

    def test_sign_matches_pm1_label_convention(self) -> None:
        # The wrapper output is sign-compatible with {-1, +1} labels: the
        # biased expectation can be thresholded at zero.
        data = get_loan_data(batch_size=8, n_samples=100, random_state=SEED)
        weights = initialize_weights(1, seed=SEED)
        sample = data.X_train[0]
        value = variational_classifier(weights, 0.1, sample)
        predicted = 1 if value >= 0 else -1
        assert predicted in (-1, 1)


# ---------------------------------------------------------------------------
# Output validation: the quantum state must survive training intact
# ---------------------------------------------------------------------------


class TestQNodeOutputValidation:
    """Guard the QNode's output contract with dummy data.

    ``quantum_classifier`` is a Pauli-Z expectation value, so physics
    guarantees it lies in [-1, 1]; these tests turn that guarantee into a
    CI-enforced invariant (float type, strict interior bounds) so a broken
    circuit, malformed readout, or bad input handling is caught before it
    poisons a training run.
    """

    def test_output_is_floating_point(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        x = np.array([0.3, 1.1, 2.0, 2.8])
        output = quantum_classifier(weights, x)
        # PennyLane returns a 0-D autograd tensor; its dtype must be a
        # floating type and its value a finite real number.
        assert np.issubdtype(np.asarray(output).dtype, np.floating)
        assert np.isfinite(float(output))

    def test_output_strictly_bounded_for_dummy_data(self) -> None:
        # Random dummy weights and dummy features: the readout must land
        # strictly inside (-1, 1) -- hitting a bound would indicate a
        # degenerate state (pure |0> or |1>) or a broken observable.
        rng = np.random.default_rng(SEED)
        for n_layers in (1, 2, 3):
            for _ in range(5):
                weights = rng.uniform(-np.pi, np.pi, size=(n_layers, N_QUBITS, 3))
                x = rng.uniform(0.0, np.pi, size=N_QUBITS)
                output = float(quantum_classifier(weights, x))
                assert -1.0 < output < 1.0, (
                    f"<Z_0> escaped (-1, 1) with {n_layers} layers: {output}"
                )

    def test_output_bounded_at_input_extremes(self) -> None:
        # Features pinned to the Bloch-sphere arc endpoints {0, pi} across
        # many random circuits: the expectation may reach the boundary at
        # exact eigenstates but must never leave [-1, 1] or go non-finite.
        rng = np.random.default_rng(SEED)
        for _ in range(10):
            weights = rng.uniform(-np.pi, np.pi, size=(2, N_QUBITS, 3))
            for extreme in (
                np.zeros(N_QUBITS),
                np.full(N_QUBITS, np.pi),
                np.array([0.0, np.pi, 0.0, np.pi]),
            ):
                output = float(quantum_classifier(weights, extreme))
                assert -1.0 <= output <= 1.0
                assert np.isfinite(output)

    def test_biased_wrapper_output_is_finite_float(self) -> None:
        # The hybrid wrapper shifts the expectation by a classical bias;
        # it must still be a finite floating-point scalar.
        weights = initialize_weights(2, seed=SEED)
        x = np.array([0.5, 1.0, 1.5, 2.5])
        output = variational_classifier(weights, 0.25, x)
        assert np.issubdtype(np.asarray(output).dtype, np.floating)
        assert np.isfinite(float(output))
