"""Training framework for the hybrid quantum-classical loan default VQC.

Binds the classical data pipeline (``src.data.pipeline``) to the quantum
execution layer (``src.models.quantum_vqc``):

1. builds quantum-ready train/test splits via :func:`get_loan_data`;
2. initializes trainable circuit weights -- shape ``(n_layers, 4, 3)``,
   ``requires_grad=True`` -- and a trainable classical bias;
3. minimizes the **square loss** between ``variational_classifier``
   outputs and the {-1, +1} default labels with PennyLane's
   ``NesterovMomentumOptimizer`` (default) or ``AdamOptimizer``;
4. prints loss and training accuracy at the end of every epoch;
5. persists the optimized weights and bias as ``.npy`` files in
   ``src/utils/`` via :mod:`src.utils.checkpoint`.

Run the default 20-epoch training as::

    python -m src.train            # defaults: 3 layers, 20 epochs, Nesterov
    python -m src.train --optimizer adam --stepsize 0.01

Label convention (inherited from the pipeline): +1 = default, -1 = repaid.
Prediction rule: ``+1`` when ``variational_classifier(weights, bias, x) >= 0``.

The square loss

.. math::

    \\mathcal{L}(\\Theta) = \\frac{1}{B}\\sum_{i=1}^{B}
    \\big(f(\\Theta, x_i) - y_i\\big)^2

is smooth, differentiable through the QNode, and well matched to the
{+1, -1} regression-to-classification setup: zero loss forces the biased
expectation value onto the correct sign with unit magnitude.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.data.pipeline import get_loan_data
from src.models.quantum_vqc import initialize_weights, variational_classifier
from src.utils.checkpoint import DEFAULT_CHECKPOINT_DIR, save_checkpoint

__all__ = [
    "TrainingResult",
    "accuracy",
    "cost",
    "main",
    "make_optimizer",
    "predict_labels",
    "square_loss",
    "train",
]


def square_loss(targets, predictions) -> float:
    """Mean squared error between targets and predictions.

    .. math::

        \\text{loss} = \\frac{1}{N}\\sum_i (t_i - p_i)^2

    Written with plain Python arithmetic so autograd can trace it when the
    predictions are PennyLane tensors (no ``numpy`` dispatch on ArrayBox
    values).

    Parameters
    ----------
    targets:
        Ground-truth labels, 1-D array-like in {-1, +1}.
    predictions:
        Model outputs, 1-D array-like of the same length (may be autograd
        tensors during gradient tracing).

    Returns
    -------
    float
        The mean squared error; ``0.0`` for a perfect fit.

    Raises
    ------
    ValueError
        If ``targets`` and ``predictions`` have different lengths or the
        batch is empty.
    """
    if len(targets) != len(predictions):
        raise ValueError(
            f"targets and predictions must have the same length; got "
            f"{len(targets)} and {len(predictions)}."
        )
    if len(targets) == 0:
        raise ValueError("square_loss requires a non-empty batch.")
    loss = 0.0
    for target, prediction in zip(targets, predictions):
        loss = loss + (target - prediction) ** 2
    return loss / len(targets)


def cost(weights, bias, X, Y) -> float:
    """Square-loss cost of the hybrid classifier on a batch.

    ``f(x) = variational_classifier(weights, bias, x)`` is evaluated for
    every sample in ``X`` and compared against ``Y`` with
    :func:`square_loss`.

    Parameters
    ----------
    weights:
        Trainable circuit weights of shape ``(n_layers, 4, 3)``.
    bias:
        Trainable scalar classical bias.
    X:
        Batch of feature vectors, shape ``(batch, 4)``; angles in
        ``[0, pi]`` from the data pipeline.
    Y:
        Batch of labels in {-1, +1}, shape ``(batch,)``.

    Returns
    -------
    float
        Mean square loss over the batch (autograd-traceable).

    Raises
    ------
    ValueError
        Propagated from :func:`variational_classifier` / :func:`square_loss`
        on shape mismatches.
    """
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


def predict_labels(weights, bias, X) -> np.ndarray:
    """Class predictions in {-1, +1} for a batch of samples.

    Applies the decision rule of the hybrid model: predict +1 (default)
    when ``variational_classifier(weights, bias, x) >= 0``, else -1
    (repaid).

    Parameters
    ----------
    weights:
        Circuit weights of shape ``(n_layers, 4, 3)``.
    bias:
        Scalar classical bias.
    X:
        Feature vectors, shape ``(n_samples, 4)``.

    Returns
    -------
    numpy.ndarray
        Integer labels of shape ``(n_samples,)`` with values in {-1, +1}.
    """
    return np.array(
        [1 if variational_classifier(weights, bias, x) >= 0 else -1 for x in X],
        dtype=np.int64,
    )


def accuracy(weights, bias, X, Y) -> float:
    """Classification accuracy of the hybrid model on a dataset.

    Parameters
    ----------
    weights:
        Circuit weights of shape ``(n_layers, 4, 3)``.
    bias:
        Scalar classical bias.
    X:
        Feature vectors, shape ``(n_samples, 4)``.
    Y:
        Ground-truth labels in {-1, +1}, shape ``(n_samples,)``.

    Returns
    -------
    float
        Fraction of correctly classified samples in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``X`` is empty.
    """
    if len(X) == 0:
        raise ValueError("accuracy requires a non-empty feature set.")
    predicted = predict_labels(weights, bias, X)
    return float(np.mean(predicted == np.asarray(Y)))


def make_optimizer(name: str = "nesterov", stepsize: float = 0.05):
    """Instantiate a PennyLane gradient-descent optimizer by name.

    Parameters
    ----------
    name:
        ``"nesterov"`` for ``NesterovMomentumOptimizer`` (momentum 0.9) or
        ``"adam"`` for ``AdamOptimizer``. Case-insensitive.
    stepsize:
        Optimizer learning rate.

    Returns
    -------
        A PennyLane optimizer instance with a ``step`` method.

    Raises
    ------
    ValueError
        If ``name`` is not one of ``{"nesterov", "adam"}`` or ``stepsize``
        is not positive.
    """
    if stepsize <= 0.0:
        raise ValueError(f"stepsize must be > 0; got {stepsize}.")
    normalized = name.lower()
    if normalized == "nesterov":
        return qml.NesterovMomentumOptimizer(stepsize=stepsize, momentum=0.9)
    if normalized == "adam":
        return qml.AdamOptimizer(stepsize=stepsize)
    raise ValueError(
        f"Unknown optimizer '{name}'; expected 'nesterov' or 'adam'."
    )


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a full training run.

    Attributes
    ----------
    weights:
        Optimized circuit weights, shape ``(n_layers, 4, 3)``,
        ``requires_grad=True``.
    bias:
        Optimized scalar classical bias, ``requires_grad=True``.
    loss_history:
        Per-epoch mean square loss on the training set (pre-update batch
        losses, weighted by batch size), length ``epochs``.
    accuracy_history:
        Per-epoch training-set accuracy, length ``epochs``.
    test_accuracy:
        Accuracy of the final model on the held-out test split.
    elapsed_seconds:
        Wall-clock training time in seconds.
    checkpoint_dir:
        Directory the checkpoint was written to, or ``None`` when saving
        was disabled.
    """

    weights: pnp.ndarray
    bias: pnp.ndarray
    loss_history: tuple[float, ...]
    accuracy_history: tuple[float, ...]
    test_accuracy: float
    elapsed_seconds: float
    checkpoint_dir: Path | None


def train(
    n_layers: int = 3,
    epochs: int = 20,
    batch_size: int = 32,
    n_samples: int = 800,
    default_rate: float = 0.2,
    missing_fraction: float = 0.05,
    test_size: float = 0.2,
    optimizer: str = "nesterov",
    stepsize: float = 0.05,
    seed: int = 42,
    csv_path: str | Path | None = None,
    save: bool = True,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    verbose: bool = True,
) -> TrainingResult:
    """Train the hybrid VQC on loan-default data and checkpoint it.

    The loop iterates ``epochs`` times over the mini-batches produced by
    :func:`get_loan_data`. For every batch the cost is evaluated (for
    logging), then the optimizer takes one gradient step on the batch.
    After each epoch the training loss (batch-size-weighted mean of the
    pre-update batch losses) and the full training-set accuracy are
    printed, and the histories are recorded.

    Parameters
    ----------
    n_layers:
        Number of variational circuit layers; weights are initialized
        uniformly in ``[-pi, pi]`` with shape ``(n_layers, 4, 3)`` and
        ``requires_grad=True``. Must be >= 1.
    epochs:
        Number of passes over the training set. Must be >= 1.
    batch_size:
        Maximum samples per mini-batch. Must be >= 1.
    n_samples:
        Number of synthetic loan samples (ignored if ``csv_path`` given).
    default_rate:
        Approximate default base rate of the synthetic data.
    missing_fraction:
        Fraction of missing feature cells in the synthetic data.
    test_size:
        Held-out test fraction of the dataset.
    optimizer:
        ``"nesterov"`` (default) or ``"adam"``; see :func:`make_optimizer`.
    stepsize:
        Optimizer learning rate (default 0.05 suits Nesterov momentum; use
        about 0.01 with Adam).
    seed:
        Single seed for data generation, splitting, batching, and weight
        initialization; identical arguments reproduce identical runs.
    csv_path:
        Optional CSV data source; ``None`` simulates data.
    save:
        When ``True`` (default), persist weights and bias via
        :func:`src.utils.checkpoint.save_checkpoint`.
    checkpoint_dir:
        Directory for the ``.npy`` checkpoint; defaults to ``src/utils/``.
    verbose:
        When ``True`` (default), print loss and accuracy every epoch.

    Returns
    -------
    TrainingResult
        Optimized parameters, loss/accuracy histories, test accuracy,
        timing, and checkpoint location.

    Raises
    ------
    ValueError
        If ``n_layers``, ``epochs``, or ``batch_size`` are below 1.

    Examples
    --------
    >>> result = train(n_layers=2, epochs=5, n_samples=200, verbose=False)
    >>> result.weights.shape
    (2, 4, 3)
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1; got {n_layers}.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")

    data = get_loan_data(
        batch_size=batch_size,
        csv_path=csv_path,
        n_samples=n_samples,
        default_rate=default_rate,
        missing_fraction=missing_fraction,
        test_size=test_size,
        random_state=seed,
    )

    weights = initialize_weights(n_layers, seed=seed)
    bias = pnp.array(0.0, requires_grad=True)
    opt = make_optimizer(optimizer, stepsize)

    loss_history: list[float] = []
    accuracy_history: list[float] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        samples_seen = 0
        for X_batch, y_batch in data.train_batches:
            # Pre-update batch loss for logging (weighted by batch size).
            epoch_loss += len(y_batch) * float(cost(weights, bias, X_batch, y_batch))
            samples_seen += len(y_batch)
            weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=y_batch)
        epoch_loss /= samples_seen

        train_accuracy = accuracy(weights, bias, data.X_train, data.y_train)
        loss_history.append(epoch_loss)
        accuracy_history.append(train_accuracy)
        if verbose:
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"loss: {epoch_loss:.4f} | "
                f"train accuracy: {train_accuracy:.4f}"
            )

    test_accuracy = accuracy(weights, bias, data.X_test, data.y_test)
    elapsed = time.perf_counter() - started

    checkpoint_path: Path | None = None
    if save:
        checkpoint_path = save_checkpoint(weights, bias, checkpoint_dir)

    if verbose:
        print(
            f"Training finished in {elapsed:.1f}s | "
            f"test accuracy: {test_accuracy:.4f}"
        )
        if checkpoint_path is not None:
            print(f"Checkpoint saved to {checkpoint_path}")

    return TrainingResult(
        weights=weights,
        bias=bias,
        loss_history=tuple(loss_history),
        accuracy_history=tuple(accuracy_history),
        test_accuracy=test_accuracy,
        elapsed_seconds=elapsed,
        checkpoint_dir=checkpoint_path,
    )


def main(argv: list[str] | None = None) -> TrainingResult:
    """Command-line entry point: ``python -m src.train``.

    Parameters
    ----------
    argv:
        Argument list to parse; defaults to ``sys.argv[1:]``. Exposed for
        testing.

    Returns
    -------
    TrainingResult
        The result of the invoked training run.
    """
    parser = argparse.ArgumentParser(
        description="Train the hybrid quantum-classical loan default VQC."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="variational layers (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=800,
        help="synthetic dataset size (default: 800)",
    )
    parser.add_argument(
        "--optimizer", choices=["nesterov", "adam"], default="nesterov",
        help="gradient optimizer (default: nesterov)",
    )
    parser.add_argument(
        "--stepsize", type=float, default=0.05,
        help="optimizer learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed (default: 42)"
    )
    parser.add_argument(
        "--csv-path", type=str, default=None, help="optional CSV data source"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="checkpoint directory (default: src/utils/)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="skip writing the checkpoint"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-epoch printing"
    )
    args = parser.parse_args(argv)

    checkpoint_dir = (
        DEFAULT_CHECKPOINT_DIR if args.checkpoint_dir is None else args.checkpoint_dir
    )
    return train(
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        optimizer=args.optimizer,
        stepsize=args.stepsize,
        seed=args.seed,
        csv_path=args.csv_path,
        save=not args.no_save,
        checkpoint_dir=checkpoint_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
