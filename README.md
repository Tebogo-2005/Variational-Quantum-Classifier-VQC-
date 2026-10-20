# Hybrid Quantum-Classical Loan Default Predictor

A **Variational Quantum Classifier (VQC)** for binary loan-default risk
prediction, built with [PennyLane](https://pennylane.ai/) for the quantum
layer and [scikit-learn](https://scikit-learn.org/) for classical
preprocessing — wrapped in a clean **Streamlit web UI** for interactive
inference.

> **Status:** Phases 0-3 complete and verified — data pipeline, quantum
> circuit, training framework (20-epoch run: train accuracy **0.861**, test
> accuracy **0.850** vs. 0.800 majority baseline), 83+-test validation
> suite, and the interactive risk-explorer UI. Full engineering spec:
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Live Demo: the Risk Explorer UI](#live-demo-the-risk-explorer-ui)
3. [How the Hybrid Model Works](#how-the-hybrid-model-works)
4. [Repository Layout](#repository-layout)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Testing](#testing)
8. [Key Design Decisions](#key-design-decisions)
9. [Roadmap](#roadmap)
10. [References](#references)

---

## What This Project Is

Loan-default prediction is a noisy, imbalanced, tabular classification
problem where classical ML remains the production tool of choice. This
project makes **no claim of quantum advantage**. What it delivers:

- **A reference hybrid architecture** in which a parameterized quantum
  circuit is a drop-in, differentiable layer inside a classical
  scikit-learn-style pipeline.
- **End-to-end engineering rigor**: leakage-safe preprocessing, exact
  analytic gradients via the parameter-shift rule, seeded reproducibility,
  and an 83+-test validation suite guarding every interface contract.
- **A full-stack demo**: a trained VQC checkpoint served through an
  interactive Streamlit UI with a live circuit diagram and risk gauge.

## Live Demo: the Risk Explorer UI

```bash
.venv/bin/streamlit run src/app.py
```

| UI element | What it does |
|---|---|
| **Sidebar sliders** | Tweak a borrower's credit score, annual income, debt-to-income ratio, and employment history |
| **Classical preprocessing** | The record is pushed through the *exact* training-time `LoanPreprocessor` (median impute → standardize → PCA(4) → min-max to \([0,\pi]\)), reconstructed deterministically from the training seed |
| **Quantum forward pass** | One evaluation of the `variational_classifier` QNode on `default.qubit` with the optimized `.npy` weights + bias |
| **Risk gauge** | The biased expectation \(\langle Z_0\rangle + b\) mapped linearly to a 0-100% "Quantum-Calculated Default Risk" rating with Low/Moderate/Elevated/High bands |
| **Circuit view** | A live `qml.draw` schematic of the 4-qubit circuit drawn with **that borrower's** rotation angles, plus a stage-by-stage layer table |

## How the Hybrid Model Works

```text
┌──────────────────────────── CLASSICAL DOMAIN ────────────────────────────┐
│  loan features ─► impute ─► standardize ─► PCA(4) ─► scale to [0, pi]    │
│  (income, credit score, debt ratio, employment years)                    │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │  x ∈ R^4  (rotation angles)
                                  ▼
┌───────────────────────────── QUANTUM DOMAIN ─────────────────────────────┐
│   |0>⁴ ─► RY(x₀)·RY(x₁)·RY(x₂)·RY(x₃) ─► [RX RY RZ × 4 qubits ─► CNOT   │
│          angle embedding                 ring] × L layers ─► ⟨Z₀⟩        │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │  ⟨Z₀⟩ ∈ [-1, 1]
                                  ▼
                   score = ⟨Z₀⟩ + b    →    risk% = (score + 1)/2 · 100
                   predict DEFAULT when score ≥ 0  (+1 = default, -1 = repay)
```

**State preparation** — each of the 4 PCA components is angle-embedded as
`RY(x_i)` on qubit `i`, moving each qubit from \(|0\rangle\) toward
\(|1\rangle\) monotonically in the feature value.

**Variational layer** — per qubit a trainable Euler rotation triple
`RX → RY → RZ` (weights shape `(L, 4, 3)`), followed by the linear CNOT
ring `0→1→2→3→0` that spreads feature interactions across all qubits.

**Readout & training** — the local Pauli-Z expectation on qubit 0 plus a
classical bias forms the score; the sign is the decision. Training
minimizes the square loss \(\frac{1}{B}\sum_i (f(x_i) - y_i)^2\) against
±1 labels with `NesterovMomentumOptimizer` (or `AdamOptimizer`), gradients
computed exactly through the QNode.

## Repository Layout

```text
.
├── README.md                  # This file
├── requirements.txt           # Pinned, resolution-verified dependencies
├── .gitignore
├── docs/
│   └── ARCHITECTURE.md        # Full engineering specification
├── src/
│   ├── __init__.py
│   ├── app.py                 # Streamlit web UI (risk explorer)
│   ├── train.py               # Training loop + CLI (python -m src.train)
│   ├── data/
│   │   └── pipeline.py        # Simulation, CSV loading, leakage-safe preprocessing
│   ├── models/
│   │   └── quantum_vqc.py     # QNode circuit: embedding, layers, readout
│   └── utils/
│       ├── checkpoint.py      # .npy save/load of optimized weights + bias
│       ├── weights.npy        # Trained circuit parameters (gitignored)
│       └── bias.npy           # Trained classical bias (gitignored)
└── tests/
    ├── conftest.py            # Wires `src` into sys.path
    ├── test_pipeline.py       # Data pipeline + qubit-width contracts
    ├── test_quantum.py        # Circuit structure, bounds, gradients
    ├── test_train.py          # Loss, optimizer, training loop, checkpoints
    └── test_app.py            # UI helpers: risk mapping, gauge, loading
```

## Getting Started

**Prerequisites:** Python 3.12+ (pinned NumPy 2.5.4 is the binding
constraint) and `pip`.

```bash
# 1. Isolated environment (never install into a shared conda base --
#    PennyLane requires numpy>=2, which breaks numpy-1.x-ABI packages)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Pinned dependency set
pip install -r requirements.txt

# 3. Train the model (20 epochs, ~7 min on CPU; writes src/utils/*.npy)
python -m src.train

# 4. Launch the web UI
streamlit run src/app.py

# 5. Run the validation suite
pytest
```

## Usage

### Training CLI

```bash
python -m src.train [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--n-layers` | 3 | Variational circuit layers |
| `--epochs` | 20 | Passes over the training set |
| `--batch-size` | 32 | Mini-batch size |
| `--n-samples` | 800 | Synthetic dataset size |
| `--optimizer` | nesterov | `nesterov` or `adam` |
| `--stepsize` | 0.05 | Learning rate (use ~0.01 with Adam) |
| `--seed` | 42 | Global seed (data, split, init, batching) |
| `--csv-path` | — | Optional CSV data source |
| `--checkpoint-dir` | src/utils | Checkpoint directory |
| `--no-save` / `--quiet` | — | Skip checkpointing / suppress printing |

Programmatic access:

```python
from src.data.pipeline import get_loan_data
from src.models.quantum_vqc import initialize_weights, variational_classifier
from src.train import train
from src.utils.checkpoint import load_checkpoint

result = train(n_layers=3, epochs=20)          # -> TrainingResult dataclass
data = get_loan_data(batch_size=32)            # -> LoanDataset (±1 labels)
weights, bias = load_checkpoint("src/utils")   # -> trainable tensors
score = float(variational_classifier(weights, bias, data.X_test[0]))
```

### Python API highlights

- `simulate_loan_data(n_samples, default_rate, ...)` — reproducible synthetic
  loans driven by a latent creditworthiness factor
- `LoanPreprocessor(n_components=4)` — fit-once/transform-many
  impute → scale → PCA → angle pipeline (leakage-safe by construction)
- `get_loan_data(batch_size)` — stratified 80/20 splits, ±1 labels, seeded
  train batches, fitted preprocessor exposed for inference
- `quantum_classifier(weights, x)` — the QNode returning \(\langle Z_0\rangle\)
- `variational_classifier(weights, bias, x)` — biased hybrid score

## Testing

```bash
pytest            # or: .venv/bin/python -m pytest
```

**83+ tests, all passing** (~14 s), organized by module:

- **`test_pipeline.py`** — simulation schema/base-rate/missingness/seed,
  CSV validation, angle ranges and leakage safety, batch partitioning, and
  the *quantum-interface contract*: every pipeline output has exactly
  `N_QUBITS` columns of finite floating-point Bloch-valid angles.
- **`test_quantum.py`** — gate placement via `qml.draw` (including the
  wrap-around CNOT ring), exact physics identities (zero angles →
  \(\langle Z_0\rangle = +1\), \(\pi\) on qubit 0 → \(-1\)), output dtype
  and strict \((-1, 1)\) bounds on dummy data, gradients verified against
  finite differences.
- **`test_train.py`** — exact loss values, analytic gradient checks,
  sign-rule accuracy, optimizer dispatch, seed reproducibility, loss
  decrease across epochs, checkpoint round-trips, CLI.
- **`test_app.py`** — risk mapping (endpoints, clipping), band
  thresholds, borrower schema, SVG gauge output, and the deterministic
  reconstruction of the training-time preprocessor + single forward pass
  (auto-skips if no checkpoint exists).

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Feature encoding | RY angle embedding, 1 angle/qubit | Bounded \([0,\pi]\) angles from the pipeline; NISQ-friendly single-qubit gates |
| Ansatz | RX/RY/RZ triples + CNOT ring | Native gates, linear connectivity, non-commuting Euler rotations span \(SU(2)\) |
| Input width | PCA to 4 components = qubit count | Qubits are the scarce resource; one rotation angle per qubit |
| Readout | Local \(\langle Z_0\rangle\) + classical bias | Bounded score; bias shifts the decision boundary under class imbalance |
| Loss | Square loss on ±1 labels | Smooth, differentiable through the QNode, sign-consistent |
| Gradients | Parameter-shift via autograd QNode | Exact, hardware-compatible; every parameter enters one Pauli rotation once |
| Primary guard | Test-enforced shape/bounds contracts | Regressions caught before they poison a training run |
| Serving | Streamlit + deterministic preprocessor reconstruction | Same-seed refit reproduces training statistics exactly; no pickle needed |

## Roadmap

- [x] **Phase 0 — Scaffold.** Monorepo, pinned deps, README, architecture spec.
- [x] **Phase 1 — Data module.** Synthetic generator, leakage-safe
      preprocessor, stratified splits, unit tests.
- [x] **Phase 2 — Quantum model.** RY embedding, variational layers, QNode,
      bias wrapper, structural tests.
- [x] **Phase 3 — Training & evaluation.** Square-loss loop, Nesterov/Adam,
      checkpointing, CLI, 0.850 test accuracy.
- [x] **Validation suite.** Pipeline↔qubit shape contracts, QNode output
      bounds, gradient checks — 83+ tests.
- [x] **Web UI.** Streamlit risk explorer: borrower sliders, gauge,
      live circuit diagram.
- [ ] **Phase 4 — Scaling & noise.** `lightning.qubit` backend, shot-based
      noise studies, data re-uploading per layer, classical baseline
      benchmarks (PR-AUC harness), fairness audit.

## References

1. A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre,
   *Data re-uploading on a quantum computer*, Phys. Rev. A **104**, 032416
   (2021). arXiv:2007.08306
2. M. Schuld, R. Sweke, J. J. Meyer, *Effect of data encoding on the
   expressive power of variational quantum machine learning models*,
   Phys. Rev. A **103**, 032430 (2021). arXiv:2008.08605
3. K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, *Quantum circuit learning*,
   Phys. Rev. A **98**, 032309 (2018). arXiv:1803.00745
4. J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven,
   *Barren plateaus in quantum neural network training landscapes*,
   Nat. Commun. **9**, 4812 (2018). arXiv:1803.11173
5. PennyLane documentation — <https://pennylane.ai/>
6. scikit-learn documentation — <https://scikit-learn.org/>
7. Streamlit documentation — <https://docs.streamlit.io/>
