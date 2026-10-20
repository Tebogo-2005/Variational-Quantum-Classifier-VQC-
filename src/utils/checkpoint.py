"""Checkpoint persistence for trained VQC parameters.

Saves and loads the optimized circuit weights and classical bias produced
by :mod:`src.train` as plain ``.npy`` files (requirement: artifacts live in
``src/utils/``).

Layout inside the checkpoint directory:

- ``weights.npy`` -- circuit weights, shape ``(n_layers, n_qubits, 3)``;
- ``bias.npy``    -- scalar classical bias term.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pennylane import numpy as pnp

__all__ = [
    "BIAS_FILENAME",
    "WEIGHTS_FILENAME",
    "DEFAULT_CHECKPOINT_DIR",
    "load_checkpoint",
    "save_checkpoint",
]

#: Default artifact directory: the ``src/utils`` package folder.
DEFAULT_CHECKPOINT_DIR: Path = Path(__file__).resolve().parent

#: File name of the persisted circuit weights.
WEIGHTS_FILENAME: str = "weights.npy"

#: File name of the persisted classical bias.
BIAS_FILENAME: str = "bias.npy"


def save_checkpoint(
    weights,
    bias,
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """Persist trained weights and bias as ``.npy`` files.

    Parameters
    ----------
    weights:
        Circuit weights of shape ``(n_layers, n_qubits, 3)``; any
        array-like (PennyLane autograd tensors are accepted, the trainable
        flag is not serialized).
    bias:
        Scalar classical bias term.
    directory:
        Target directory; created if it does not exist. Defaults to
        ``src/utils/``.

    Returns
    -------
    Path
        The directory the two ``.npy`` files were written to.

    Raises
    ------
    ValueError
        If ``bias`` is not scalar or ``weights`` is not a 3-D array.
    """
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 3:
        raise ValueError(
            f"weights must be a 3-D array (n_layers, n_qubits, 3); got "
            f"shape {weights_array.shape}."
        )
    bias_array = np.asarray(bias, dtype=float)
    if bias_array.ndim != 0:
        raise ValueError(
            f"bias must be scalar; got array of shape {bias_array.shape}."
        )

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / WEIGHTS_FILENAME, weights_array)
    np.save(target / BIAS_FILENAME, bias_array)
    return target


def load_checkpoint(
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> tuple[pnp.ndarray, pnp.ndarray]:
    """Load weights and bias written by :func:`save_checkpoint`.

    Parameters
    ----------
    directory:
        Directory containing ``weights.npy`` and ``bias.npy``. Defaults to
        ``src/utils/``.

    Returns
    -------
    tuple[pennylane.numpy.ndarray, pennylane.numpy.ndarray]
        ``(weights, bias)`` with ``requires_grad=True`` so training can be
        resumed or the model evaluated immediately.

    Raises
    ------
    FileNotFoundError
        If either checkpoint file is missing in ``directory``.
    """
    weights_path = Path(directory) / WEIGHTS_FILENAME
    bias_path = Path(directory) / BIAS_FILENAME
    for path in (weights_path, bias_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}. Train first "
                f"(python -m src.train) or pass the correct directory."
            )

    weights = pnp.array(np.load(weights_path), requires_grad=True)
    bias = pnp.array(np.load(bias_path), requires_grad=True)
    return weights, bias
