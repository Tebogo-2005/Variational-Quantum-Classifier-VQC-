"""Unit tests for the training framework (src.train, src.utils.checkpoint).

Contracts under test:
- ``square_loss`` / ``cost`` compute exact, positive, finite values;
- ``predict_labels`` / ``accuracy`` implement the sign decision rule;
- ``make_optimizer`` dispatches Nesterov/Adam and rejects bad input;
- ``train`` reduces the loss across epochs, records histories, evaluates
  the test split, and writes a loadable checkpoint;
- checkpoint save/load round-trips exactly;
- the CLI entry point runs a tiny training.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from src.models.quantum_vqc import initialize_weights, variational_classifier
from src.train import (
    TrainingResult,
    accuracy,
    cost,
    main,
    make_optimizer,
    predict_labels,
    square_loss,
    train,
)
from src.utils.checkpoint import (
    BIAS_FILENAME,
    WEIGHTS_FILENAME,
    load_checkpoint,
    save_checkpoint,
)

SEED = 7


# ---------------------------------------------------------------------------
# square_loss and cost
# ---------------------------------------------------------------------------


class TestSquareLoss:
    def test_perfect_predictions_give_zero(self) -> None:
        assert square_loss([1, -1, 1], [1, -1, 1]) == pytest.approx(0.0)

    def test_exact_value(self) -> None:
        # (1-0.5)^2 + (-1-0.5)^2 = 0.25 + 2.25 = 2.5, / 2 = 1.25
        assert square_loss([1, -1], [0.5, 0.5]) == pytest.approx(1.25)

    def test_zero_predictions_against_pm1_targets(self) -> None:
        # Untrained model outputting ~0 loses mean(y^2) = 1.
        assert square_loss([1, -1, 1, -1], [0.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            square_loss([1, -1], [1.0])

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            square_loss([], [])


class TestCost:
    def test_positive_and_finite(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        bias = pnp.array(0.1, requires_grad=True)
        X = np.full((5, 4), 0.7)
        Y = np.array([1, -1, 1, -1, 1])
        value = cost(weights, bias, X, Y)
        assert np.isfinite(value)
        assert value > 0.0

    def test_agrees_with_manual_computation(self) -> None:
        weights = initialize_weights(1, seed=SEED)
        bias = 0.0
        X = np.array([[0.2, 0.9, 1.4, 2.8], [3.0, 0.1, 0.5, 1.9]])
        Y = np.array([1, -1])
        manual = np.mean(
            [
                (float(variational_classifier(weights, bias, x)) - y) ** 2
                for x, y in zip(X, Y)
            ]
        )
        assert cost(weights, bias, X, Y) == pytest.approx(manual)

    def test_differentiable(self) -> None:
        weights = initialize_weights(1, seed=SEED)
        bias = pnp.array(0.0, requires_grad=True)
        X = np.full((4, 4), 0.5)
        Y = np.array([1, -1, 1, -1])
        grads = qml.grad(cost, argnums=(0, 1))(weights, bias, X, Y)
        grad_weights, grad_bias = grads
        assert np.asarray(grad_weights).shape == weights.shape
        assert np.isfinite(np.asarray(grad_weights)).all()
        assert grad_bias == pytest.approx(
            2 * np.mean(
                [
                    float(variational_classifier(weights, 0.0, x)) - y
                    for x, y in zip(X, Y)
                ]
            )
        )


# ---------------------------------------------------------------------------
# predict_labels and accuracy
# ---------------------------------------------------------------------------


class TestPredictions:
    def test_positive_bias_forces_default_predictions(self) -> None:
        # bias = +100 dominates <Z_0> in [-1, 1]: every output is positive.
        weights = initialize_weights(2, seed=SEED)
        X = np.array([[0.1, 0.2, 0.3, 0.4], [2.9, 1.5, 0.7, 3.0]])
        predicted = predict_labels(weights, 100.0, X)
        np.testing.assert_array_equal(predicted, [1, 1])

    def test_negative_bias_forces_repay_predictions(self) -> None:
        weights = initialize_weights(2, seed=SEED)
        X = np.array([[0.1, 0.2, 0.3, 0.4], [2.9, 1.5, 0.7, 3.0]])
        predicted = predict_labels(weights, -100.0, X)
        np.testing.assert_array_equal(predicted, [-1, -1])

    def test_accuracy_with_dominating_bias_matches_class_rate(self) -> None:
        weights = initialize_weights(1, seed=SEED)
        X = np.array([[0.5, 1.0, 1.5, 2.0]] * 4)
        Y = np.array([1, -1, -1, -1])  # 25% defaults
        assert accuracy(weights, 100.0, X, Y) == pytest.approx(0.25)
        assert accuracy(weights, -100.0, X, Y) == pytest.approx(0.75)

    def test_accuracy_empty_features_raises(self) -> None:
        weights = initialize_weights(1, seed=SEED)
        with pytest.raises(ValueError, match="non-empty"):
            accuracy(weights, 0.0, np.empty((0, 4)), np.empty(0, dtype=int))


# ---------------------------------------------------------------------------
# make_optimizer
# ---------------------------------------------------------------------------


class TestMakeOptimizer:
    def test_nesterov_and_adam_dispatch(self) -> None:
        nesterov = make_optimizer("nesterov", stepsize=0.05)
        adam = make_optimizer("Adam", stepsize=0.01)  # case-insensitive
        assert type(nesterov).__name__ == "NesterovMomentumOptimizer"
        assert type(adam).__name__ == "AdamOptimizer"

    def test_unknown_optimizer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimizer"):
            make_optimizer("sgd")

    def test_nonpositive_stepsize_raises(self) -> None:
        with pytest.raises(ValueError, match="stepsize"):
            make_optimizer("adam", stepsize=0.0)


# ---------------------------------------------------------------------------
# train (fast smoke configuration)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_loss_decreases_and_histories_recorded(self) -> None:
        result = train(
            n_layers=1,
            epochs=4,
            batch_size=16,
            n_samples=80,
            stepsize=0.1,
            seed=SEED,
            save=False,
            verbose=False,
        )
        assert isinstance(result, TrainingResult)
        assert result.weights.shape == (1, 4, 3)
        assert bool(result.weights.requires_grad) is True
        assert len(result.loss_history) == 4
        assert len(result.accuracy_history) == 4
        assert result.loss_history[-1] < result.loss_history[0]
        assert all(0.0 <= acc <= 1.0 for acc in result.accuracy_history)
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.checkpoint_dir is None  # save=False

    def test_training_is_seed_reproducible(self) -> None:
        kwargs = dict(
            n_layers=1, epochs=2, batch_size=16, n_samples=60,
            seed=SEED, save=False, verbose=False,
        )
        a = train(**kwargs)
        b = train(**kwargs)
        np.testing.assert_allclose(
            np.asarray(a.weights), np.asarray(b.weights), atol=1e-10
        )
        assert float(a.bias) == pytest.approx(float(b.bias))
        assert a.loss_history == pytest.approx(b.loss_history)

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            train(n_layers=0, epochs=1, n_samples=50, save=False)
        with pytest.raises(ValueError, match="epochs"):
            train(epochs=0, n_samples=50, save=False)
        with pytest.raises(ValueError, match="batch_size"):
            train(batch_size=0, n_samples=50, save=False)

    def test_checkpoint_written_and_loadable(self, tmp_path) -> None:
        result = train(
            n_layers=2,
            epochs=2,
            batch_size=16,
            n_samples=60,
            seed=SEED,
            save=True,
            checkpoint_dir=tmp_path,
            verbose=False,
        )
        assert result.checkpoint_dir == tmp_path
        weights_path = tmp_path / WEIGHTS_FILENAME
        bias_path = tmp_path / BIAS_FILENAME
        assert weights_path.is_file() and bias_path.is_file()

        loaded_weights, loaded_bias = load_checkpoint(tmp_path)
        assert loaded_weights.shape == (2, 4, 3)
        assert bool(loaded_weights.requires_grad) is True
        assert bool(loaded_bias.requires_grad) is True
        np.testing.assert_allclose(
            np.asarray(loaded_weights), np.asarray(result.weights), atol=1e-12
        )
        assert float(loaded_bias) == pytest.approx(float(result.bias))

    def test_verbose_printing_reports_every_epoch(self, capsys) -> None:
        train(
            n_layers=1, epochs=3, batch_size=16, n_samples=60,
            seed=SEED, save=False, verbose=True,
        )
        output = capsys.readouterr().out
        for epoch in (1, 2, 3):
            assert f"Epoch {epoch:2d}/3" in output
        assert "loss:" in output and "accuracy:" in output
        assert "test accuracy:" in output


# ---------------------------------------------------------------------------
# checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_then_load_roundtrips_exactly(self, tmp_path) -> None:
        weights = initialize_weights(3, seed=SEED)
        bias = pnp.array(-0.42, requires_grad=True)
        directory = save_checkpoint(weights, bias, tmp_path)
        assert directory == tmp_path

        loaded_weights, loaded_bias = load_checkpoint(tmp_path)
        np.testing.assert_allclose(
            np.asarray(loaded_weights), np.asarray(weights), atol=1e-15
        )
        assert float(loaded_bias) == pytest.approx(float(bias))

    def test_load_missing_checkpoint_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="weights.npy"):
            load_checkpoint(tmp_path)

    def test_non_3d_weights_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="3-D"):
            save_checkpoint(np.zeros((4, 3)), 0.0, tmp_path)

    def test_non_scalar_bias_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="scalar"):
            save_checkpoint(np.zeros((1, 4, 3)), np.zeros(2), tmp_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_cli_runs_tiny_training(self, tmp_path) -> None:
        result = main(
            [
                "--n-layers", "1",
                "--epochs", "2",
                "--batch-size", "16",
                "--n-samples", "60",
                "--stepsize", "0.1",
                "--checkpoint-dir", str(tmp_path),
                "--quiet",
            ]
        )
        assert isinstance(result, TrainingResult)
        assert result.weights.shape == (1, 4, 3)
        assert (tmp_path / WEIGHTS_FILENAME).is_file()

    def test_cli_no_save_flag(self, tmp_path) -> None:
        result = main(
            [
                "--n-layers", "1",
                "--epochs", "1",
                "--n-samples", "50",
                "--no-save",
                "--quiet",
            ]
        )
        assert result.checkpoint_dir is None
