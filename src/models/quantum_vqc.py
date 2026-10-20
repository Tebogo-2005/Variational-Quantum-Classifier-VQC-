"""Quantum execution layer: the variational quantum classifier circuit.

This module owns every quantum operation in the project. It implements the
loan-default classifier circuit on the ``default.qubit`` simulator:

1. **State preparation** -- :func:`state_preparation` angle-embeds the 4
   compressed PCA features (already scaled to ``[0, pi]`` by
   ``src.data.pipeline.LoanPreprocessor``) as ``RY`` rotations, one per
   qubit.
2. **Variational layer** -- :func:`vqc_layer` applies parameterized
   ``RX``/``RY``/``RZ`` rotations on every qubit, followed by a linear ring
   of CNOT gates that entangles neighbouring qubits (with wrap-around from
   the last qubit back to the first).
3. **Classifier circuit** -- :func:`quantum_classifier` (a PennyLane QNode)
   chains the state preparation with ``L`` variational layers and returns
   the expectation value of Pauli-Z on the first qubit (readout qubit 0).
4. **Hybrid wrapper** -- :func:`variational_classifier` adds a classical
   bias to the quantum expectation value, giving a trainable affine output
   whose sign is the predicted class. This pairs with the label convention
   of ``src.data.pipeline`` (labels in {-1, +1}, +1 = default): predict
   ``+1`` (default) when the biased expectation is positive.

Every trainable circuit parameter enters the circuit exactly once, as the
angle of a single Pauli rotation. This keeps gradients exact under the
parameter-shift rule, which PennyLane dispatches automatically through the
autograd interface.

Circuit shape summary (``n_qubits = 4``, ``L`` layers):

- ``weights``: array of shape ``(L, 4, 3)`` -- per layer, one
  ``(RX, RY, RZ)`` angle triple per qubit (12 trainable angles per layer);
- ``x``: array of shape ``(4,)`` -- one rotation angle per qubit;
- output of :func:`quantum_classifier`: ``float`` in ``[-1, 1]``;
- output of :func:`variational_classifier`: ``float`` (biased).

See ``docs/ARCHITECTURE.md`` (Section 4) for the circuit-family
specification; this module is its demo-grade realization (single encoding
instead of per-layer re-uploading, which can be enabled layer-wise later).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

__all__ = [
    "N_QUBITS",
    "N_PARAMS_PER_QUBIT",
    "dev",
    "initialize_weights",
    "quantum_classifier",
    "state_preparation",
    "variational_classifier",
    "vqc_layer",
]

#: Number of qubits (wires) in the classifier circuit.
N_QUBITS: int = 4

#: Number of rotation angles per qubit in one variational layer (RX, RY, RZ).
N_PARAMS_PER_QUBIT: int = 3

#: The project simulator: exact statevector backend on 4 wires.
dev = qml.device("default.qubit", wires=N_QUBITS)


def state_preparation(x) -> None:
    """Angle-embed the input features as single-qubit rotations.

    Feature ``x[i]`` is applied as an ``RY(x[i])`` rotation on qubit ``i``
    (a "Y-axis" angle embedding). Because :class:`src.data.pipeline.LoanPreprocessor`
    scales every feature into ``[0, pi]``, each angle moves its qubit from
    the north pole (``|0>``) toward the south pole (``|1>``) of the Bloch
    sphere, monotonically in the feature value.

    Parameters
    ----------
    x:
        Input features, any 1-D array-like of length ``N_QUBITS`` (4).
        Plain NumPy arrays (non-trainable inputs) are expected.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D with exactly 4 entries.
    """
    # ``qml.math.shape`` is interface-safe: unlike ``np.asarray`` it also
    # works on autograd ``ArrayBox`` arguments during gradient tracing.
    shape = qml.math.shape(x)
    if len(shape) != 1 or shape[0] != N_QUBITS:
        raise ValueError(
            f"state_preparation expects {N_QUBITS} features (one rotation "
            f"angle per qubit); got array of shape {shape}."
        )
    for wire in range(N_QUBITS):
        qml.RY(x[wire], wires=wire)


def vqc_layer(weights) -> None:
    """Apply one variational layer: rotations, then a CNOT ring.

    Per qubit ``q`` the layer applies the Euler rotation triple

    ``RX(weights[q, 0]) -> RY(weights[q, 1]) -> RZ(weights[q, 2])``

    and then entangles the qubits with the linear CNOT ring

    ``CNOT(0,1), CNOT(1,2), CNOT(2,3), CNOT(3,0)``

    where the last gate wraps around so qubit 0 both sends and receives
    entanglement. ``RX`` and ``RY`` and ``RZ`` do not mutually commute, so
    the triple spans arbitrary single-qubit rotations; the ring spreads
    feature information from every qubit toward the readout qubit within a
    single layer.

    Parameters
    ----------
    weights:
        Trainable angles for this layer, any array-like of shape
        ``(N_QUBITS, 3)`` -- row ``q`` holds the ``(RX, RY, RZ)`` angles of
        qubit ``q``. PennyLane-wrapped arrays (``pennylane.numpy``) are
        expected during training so gradients flow.

    Raises
    ------
    ValueError
        If ``weights`` is not 2-D with shape ``(4, 3)``.
    """
    # ``qml.math.shape`` instead of ``np.asarray``: safe for autograd
    # ArrayBox arguments during gradient tracing.
    shape = qml.math.shape(weights)
    if shape != (N_QUBITS, N_PARAMS_PER_QUBIT):
        raise ValueError(
            f"vqc_layer expects weights of shape "
            f"({N_QUBITS}, {N_PARAMS_PER_QUBIT}); got {shape}."
        )
    for wire in range(N_QUBITS):
        qml.RX(weights[wire, 0], wires=wire)
        qml.RY(weights[wire, 1], wires=wire)
        qml.RZ(weights[wire, 2], wires=wire)
    for wire in range(N_QUBITS):
        qml.CNOT(wires=[wire, (wire + 1) % N_QUBITS])


@qml.qnode(dev, interface="autograd")
def quantum_classifier(weights, x) -> float:
    """Full VQC circuit: embed, apply L variational layers, read out Z_0.

    The circuit applies :func:`state_preparation` once, then loops through
    ``L = weights.shape[0]`` calls of :func:`vqc_layer`, and finally
    measures the Pauli-Z expectation value on qubit 0:

    ``<Z_0> = <psi(x, weights) | Z x I x I x I | psi(x, weights)>``

    The expectation lies in ``[-1, 1]``: +1 means the readout qubit is
    ``|0>``-aligned, -1 means ``|1>``-aligned.

    Parameters
    ----------
    weights:
        Trainable circuit parameters of shape ``(L, N_QUBITS, 3)`` with
        ``L >= 1``; slice ``weights[l]`` is passed to layer ``l``.
    x:
        Input features, 1-D array-like of length ``N_QUBITS`` (non-trainable).

    Returns
    -------
    float
        The expectation value ``<Z_0>`` in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If ``weights`` does not have shape ``(L, 4, 3)`` with ``L >= 1`` or
        ``x`` is not 1-D of length 4.
    """
    # ``qml.math.shape`` instead of ``np.asarray``: safe for autograd
    # ArrayBox arguments during gradient tracing.
    weights_shape = qml.math.shape(weights)
    if len(weights_shape) != 3 or weights_shape[1:] != (
        N_QUBITS,
        N_PARAMS_PER_QUBIT,
    ):
        raise ValueError(
            f"quantum_classifier expects weights of shape (n_layers, "
            f"{N_QUBITS}, {N_PARAMS_PER_QUBIT}); got {weights_shape}."
        )
    if weights_shape[0] < 1:
        raise ValueError("quantum_classifier requires at least one layer.")
    x_shape = qml.math.shape(x)
    if len(x_shape) != 1 or x_shape[0] != N_QUBITS:
        raise ValueError(
            f"quantum_classifier expects {N_QUBITS} features; got array of "
            f"shape {x_shape}."
        )

    state_preparation(x)
    for layer in range(weights_shape[0]):
        vqc_layer(weights[layer])
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x) -> float:
    """Add a classical bias to the quantum classifier output.

    This is the hybrid model's scalar output for one sample:

    ``f(weights, bias, x) = quantum_classifier(weights, x) + bias``

    The bias is a purely classical trainable parameter; it lets the model
    shift the decision boundary away from ``<Z_0> = 0``, which matters when
    the two classes are imbalanced (defaults are the minority class).
    Prediction rule with the {-1, +1} label convention of
    ``src.data.pipeline``: predict +1 (default) when
    ``variational_classifier(...) >= 0``.

    Parameters
    ----------
    weights:
        Trainable circuit parameters of shape ``(L, N_QUBITS, 3)``, passed
        through to :func:`quantum_classifier`.
    bias:
        Classical bias term; a scalar (use ``pennylane.numpy`` scalars with
        ``requires_grad=True`` during training so gradients flow).
    x:
        Input features, 1-D array-like of length ``N_QUBITS``.

    Returns
    -------
    float
        ``quantum_classifier(weights, x) + bias``; lies in
        ``[bias - 1, bias + 1]``.

    Raises
    ------
    ValueError
        Propagated from :func:`quantum_classifier` for invalid shapes.
    """
    return quantum_classifier(weights, x) + bias


def initialize_weights(n_layers: int = 3, seed: int = 42) -> pnp.ndarray:
    """Draw trainable circuit weights from a uniform distribution.

    Parameters
    ----------
    n_layers:
        Number of variational layers; must be >= 1.
    seed:
        Seed for the ``numpy.random.default_rng`` generator. Identical
        seeds produce identical initializations (reproducibility contract).

    Returns
    -------
    pennylane.numpy.ndarray
        Array of shape ``(n_layers, N_QUBITS, N_PARAMS_PER_QUBIT)`` with
        angles drawn uniformly from ``[-pi, pi]`` and
        ``requires_grad=True`` so autograd tracks every parameter.

    Raises
    ------
    ValueError
        If ``n_layers`` < 1.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1; got {n_layers}.")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-np.pi, np.pi, size=(n_layers, N_QUBITS, N_PARAMS_PER_QUBIT))
    return pnp.array(weights, requires_grad=True)
