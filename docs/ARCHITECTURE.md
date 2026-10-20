# Architecture Specification

**Project:** Hybrid Quantum-Classical Loan Default Predictor (VQC)
**Document:** Engineering architecture and circuit specification
**Status:** v1.0 — scaffold phase (implementation follows this document)

> **Scope.** This document is the single source of truth for how the system is
> built: the classical data pipeline, the quantum circuit family (data
> re-uploading feature map + variational ansatz), the hybrid training loop,
> module boundaries and planned interfaces, the evaluation protocol, and the
> risk register. The [README](../README.md) is the user-facing summary.

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [System Overview and Design Principles](#2-system-overview-and-design-principles)
3. [Classical Data Pipeline (`src/data`)](#3-classical-data-pipeline-srcdata)
4. [Quantum Circuit Specification (`src/models`)](#4-quantum-circuit-specification-srcmodels)
5. [Hybrid Training Loop](#5-hybrid-training-loop)
6. [Software Architecture](#6-software-architecture)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Performance and Scaling](#8-performance-and-scaling)
9. [Risk Register](#9-risk-register)
10. [Extension Points](#10-extension-points)
11. [References](#11-references)
12. [Revision History](#12-revision-history)

---

## 1. Purpose and Scope

The system trains a **variational quantum classifier** to predict binary
loan default from tabular credit-risk features, inside a classical
scikit-learn-style workflow. Two goals, in priority order:

1. **Engineering rigor.** Leakage-safe preprocessing, exact gradients,
   seeded reproducibility, benchmark gates against classical baselines.
   No claim about the quantum model is reportable unless it clears those
   gates.
2. **Architectural clarity.** The quantum circuit is a *differentiable
   function* \(f_\Theta: \mathbb{R}^k \to [0,1]\) consumed by an ordinary
   classical training loop. Every design choice below serves that
   composition.

Out of scope for v1: production serving, streaming data, model Explainability
beyond basic audits, quantum advantage claims.

## 2. System Overview and Design Principles

```text
                    ┌──────────────────────────────────────────┐
                    │                src/data                  │
                    │  raw ─► clean ─► encode ─► scale ─► PCA  │
                    └───────────────────┬──────────────────────┘
                                        │ x ∈ R^k, y ∈ {0,1}
                    ┌───────────────────▼──────────────────────┐
                    │               src/models                 │
                    │  feature map ─► ansatz ─► QNode ─► p(x)  │
                    └───────────────────┬──────────────────────┘
                                        │ probabilities, loss
                    ┌───────────────────▼──────────────────────┐
                    │        hybrid loop (classical)           │
                    │  Adam + parameter-shift gradients         │
                    │  metrics & reports ──► src/utils          │
                    └──────────────────────────────────────────┘
```

**Design principles.**

- **P1 — The circuit is a layer.** All quantum logic lives behind QNode
  boundaries; the classical loop never inspects quantum state.
- **P2 — Leakage safety by construction.** Every learned statistic
  (imputer, encoder, scaler, PCA basis) is fit on the training split only,
  enforced by a single `sklearn.pipeline.Pipeline` object.
- **P3 — Baselines before claims.** Logistic regression and random forest
  are first-class citizens, produced by the same evaluation harness.
- **P4 — Reproducibility by default.** One seed controls every source of
  randomness (data generation, batching, parameter init, device shots).
- **P5 — Hardware plausibility.** Default circuit uses only native gates,
  linear qubit connectivity, and a gradient method valid on real devices.

## 3. Classical Data Pipeline (`src/data`)

### 3.1 Ingestion and Schema

The default source is a **synthetic loan generator** (no network dependency,
fully reproducible). A latent creditworthiness factor \(r \sim
\mathcal{N}(0,1)\) drives both features and the target, producing learnable,
nonlinear signal with a realistic minority-class rate:

| Feature | Type | Notes |
|---|---|---|
| `annual_income` | numeric | log-normal; correlated with \(r\) |
| `loan_amount` | numeric | requested principal |
| `term_months` | categorical | 12 / 24 / 36 / 60 |
| `interest_rate` | numeric | offered APR, anti-correlated with \(r\) |
| `credit_score` | numeric | 300–850 |
| `employment_years` | numeric | 0–40; missing at random (~5%) |
| `debt_to_income` | numeric | 0–1 |
| `num_prior_defaults` | count | 0–3 |
| `home_ownership` | categorical | rent / own / mortgage |
| `default` | binary | target; \(y{=}1\) means default, rate ≈ 12% |

Target rule: \(y = \mathbb{1}[\,r + \varepsilon < \tau\,]\) with \(\tau\)
set to hit the desired base rate. File- and OpenML-based loaders (e.g.
German credit, `credit-g`) are Phase-4 extensions with the same schema
contract.

### 3.2 Preprocessing

A single `ColumnTransformer` inside one `Pipeline`:

- **numeric:** `SimpleImputer(strategy="median")` → `StandardScaler`
- **categorical:** `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore")`

The pipeline object is fit **on the training split only** and applied to
validation/test splits (P2). Tests must assert that fitting on train and
re-fitting on test yields different transforms (the guard against leakage).

### 3.3 Dimensionality Reduction and the Qubit Budget

Qubits are the scarce resource: angle encoding spends one rotation gate per
feature per layer, so the input width is capped at

\[
k \;=\; 2\,n_{\text{qubits}}
\]

PCA (applied after encoding/scaling, still inside the leakage-safe pipeline)
projects the engineered features to \(k\) components. Acceptance criterion:
retained explained variance ≥ 95% at the default \(k = 8\); if not, log the
actual variance and continue (credit-risk one-hots are cheap to compress).
The canonical configuration is therefore **\(k = 8\) features ↔ 4 qubits**.

### 3.4 Angle Scaling and Feature Assignment

PCA scores are standardized (zero mean, unit variance, clipped at ±4σ) and
assigned round-robin: qubit \(q\) receives features \((x_{2q},
x_{2q+1})\). The trainable input-scaling weights \(\omega\) of the encoding
layers (Section 4.2) learn the effective rotation magnitudes, so no further
manual range mapping is imposed.

## 4. Quantum Circuit Specification (`src/models`)

### 4.1 Circuit Family

For \(n\) qubits, \(L\) re-uploading layers, inputs \(\mathbf{x} \in
\mathbb{R}^{2n}\), and trainable parameters \(\Theta = (\theta, \omega)\),
the classifier implements

\[
U(\mathbf{x}, \Theta) \;=\;
\Big( \prod_{\ell=1}^{L} \Theta_\ell\, E_\ell(\mathbf{x}) \Big)
\quad\text{with } U_{\mathrm{ent}} \text{ inserted between layers},
\]

applied to \(|0\rangle^{\otimes n}\). Gates compose right-to-left; qubits are
zero-indexed; the circuit is \(\ell\)-ordered: encode first, rotate, then
entangle.

### 4.2 Data Re-Uploading Encoding Layer \(E_\ell\)

\[
E_\ell(\mathbf{x}) \;=\; \bigotimes_{q=0}^{n-1}
R_z\!\Big(\omega^{(q,\ell)}_{2}\, x_{2q+1}\Big)\,
R_y\!\Big(\omega^{(q,\ell)}_{1}\, x_{2q}\Big)
\]

Each qubit re-encodes **its own feature pair at every layer**, with trainable
per-(qubit, layer, feature) scaling weights \(\omega\), initialized to 1.0.

**Rationale.** A single angle-encoding layer yields decision functions whose
Fourier spectrum in the inputs is low-degree — insufficient for nonlinear
credit-risk boundaries. Re-uploading interleaved with trainable unitaries
accumulates accessible frequencies across layers: with \(L\) layers the
function class contains Fourier terms of degree up to \(\mathcal{O}(L)\)
(Schuld–Sweke–Meyer), and a single qubit with enough layers is a universal
approximator (Pérez-Salinas et al.). The encoding remains cheap
single-qubit rotations: no state-preparation oracles, no ancillas, and the
same structure is directly executable on NISQ hardware (P5). Trainable
\(\omega\) additionally lets the circuit learn each feature's influence
strength per qubit and layer.

### 4.3 Variational Ansatz Layer \(\Theta_\ell\)

\[
\Theta_\ell \;=\; \bigotimes_{q=0}^{n-1}
R_z\!\Big(\theta^{(q,\ell)}_{2}\Big)\,
R_y\!\Big(\theta^{(q,\ell)}_{1}\Big)
\]

- Two trainable Euler angles per qubit per layer; \(R_y\) and \(R_z\)
  do not commute, so layer products span \(SU(2)\) on each qubit.
- Hardware-efficient: only native single-qubit rotations.
- Initialization: \(\theta \sim \mathcal{U}(-\pi, \pi)\); a near-identity
  option (small angles) is reserved for layer-wise training against barren
  plateaus (Risk R1).
- **Design invariant:** each trainable parameter enters the circuit exactly
  once, as the angle of a single Pauli-rotation gate. This keeps the
  parameter-shift rule exact per parameter (Section 5.3).

### 4.4 Entanglement \(U_{\mathrm{ent}}\)

A CNOT ring applied after each ansatz layer except the last:

\[
U_{\mathrm{ent}} \;=\; \prod_{q=0}^{n-1}
\mathrm{CNOT}\big(q,\; (q{+}1) \bmod n\big)
\]

Depth \(\mathcal{O}(n)\) (negligible at \(n{=}4\)), linear-connectivity
friendly, and sufficient to spread per-qubit feature information so that the
readout qubit can see all \(k\) features within two layers.

### 4.5 Readout

Local Pauli-Z expectation on qubit 0:

\[
\langle Z_0 \rangle =
\langle \psi(\mathbf{x},\Theta) |\, Z \otimes I^{\otimes(n-1)}\, |
\psi(\mathbf{x},\Theta) \rangle \in [-1, 1]
\]

\[
\boxed{\; p(\text{default} \mid \mathbf{x}) \;=\;
\frac{1 - \langle Z_0 \rangle}{2} \;}
\]

so \(|0\rangle\)-like states mean "repay" and \(|1\rangle\)-like states mean
"default". A local observable is chosen deliberately: local cost functions
are markedly less prone to barren plateaus than global ones (Risk R1).
Configuration flags: weighted parity readout
\(p = \tfrac{1}{2}\big(1 - \tfrac{1}{n}\sum_q \alpha_q \langle Z_q
\rangle\big)\), and post-hoc affine calibration for probability quality.

### 4.6 Parameter Accounting and Default Configuration

| Quantity | Formula | Default (\(n{=}4, L{=}4\)) |
|---|---|---|
| Ansatz angles \(\theta\) | \(2nL\) | 32 |
| Input scalings \(\omega\) | \(2nL\) | 32 |
| **Total circuit parameters** | \(4nL\) | **64** |

| Parameter | Symbol | Default | Allowed range |
|---|---|---|---|
| Qubits | \(n\) | 4 | 2–10 (sim), ≤ 20 (`lightning.qubit`) |
| Re-uploading layers | \(L\) | 4 | 1–6 (deeper risks R1) |
| Input dimension | \(k\) | \(2n = 8\) | fixed to \(2n\) |
| Gates / qubit / layer | — | 2 encoding + 2 ansatz | fixed |
| Entangler | — | CNOT ring | ring / chain |
| Readout | — | local \(\langle Z_0\rangle\) | + parity flag |
| Simulator | — | `default.qubit` | `lightning.qubit` |
| Shots | — | `None` (exact) | e.g. 1024 for noise studies |
| Trainable \(\omega\) | — | yes | freezable |

## 5. Hybrid Training Loop

### 5.1 Forward Pass

For a mini-batch \(B\), each sample \(\mathbf{x}_i\) is evaluated by the
batched QNode (PennyLane executes all \(B\) circuits in one dispatch),
yielding \(p_i \in [0,1]\) via Section 4.5, clipped to
\([\varepsilon, 1-\varepsilon]\), \(\varepsilon = 10^{-7}\).

### 5.2 Loss

Class-weighted binary cross-entropy, because defaults are the rare class
(≈ 12%):

\[
\mathcal{L}(\Theta) \;=\;
-\frac{1}{B}\sum_{i=1}^{B}
\Big[\, w_{+}\, y_i \log p_i
\;+\; w_{-}\, (1 - y_i)\, \log(1 - p_i) \Big],
\qquad
w_{\pm} = \frac{B}{2\,B_{\pm}}
\]

### 5.3 Gradients: Parameter-Shift Rule

Every parameter \(\phi \in \Theta\) enters as a single Pauli-rotation angle
(design invariant, Section 4.3), so exact analytic gradients are:

\[
\frac{\partial \langle Z_0 \rangle}{\partial \phi}
\;=\;
\frac{\langle Z_0 \rangle_{\phi + \pi/2}
\;-\; \langle Z_0 \rangle_{\phi - \pi/2}}{2}
\]

PennyLane's autograd interface dispatches this automatically; no finite
differences anywhere in the codebase. On the exact simulator the default
`diff_method` is backpropagation (cheaper); `diff_method="parameter-shift"`
is the hardware-faithful mode used in shot-based studies — the two must agree
to numerical tolerance in tests.

**Cost model.** Per Adam step with batch size \(B\) and \(P\) trainable
parameters, exact-mode parameter-shift evaluation costs \(\approx
(2P + 1)\,B\) circuit executions (\(\approx 129\,B\) at the default
configuration); shot mode multiplies by the shot count. See Section 8.

### 5.4 Optimizer, Batching, Early Stopping

- **Optimizer:** Adam, learning rate 0.01 (default), \(\beta = (0.9,
  0.999)\).
- **Batching:** mini-batches of 32, shuffled by a seeded generator (P4).
- **Early stopping:** monitor validation **average precision**; patience 10
  evaluations; restore best checkpoint.
- **Regularization:** the parameter budget itself (\(4nL \le 64\) by
  default) plus early stopping; weight decay is a config flag, off by
  default.
- **\(\omega\) freezing:** optional schedule that freezes the input-scaling
  weights after a warm-up epoch count, halving gradient cost.

## 6. Software Architecture

### 6.1 Module Map and Dependency Rules

| Module | Owns | Must not |
|---|---|---|
| `src/data` | ingestion, schema, leakage-safe pipeline, PCA, splits | import PennyLane |
| `src/models` | feature maps, ansaetze, QNodes, hybrid classifier, baselines | perform file/data IO |
| `src/utils` | seeding, metrics, config dataclass, logging | import `src.data` or `src.models` |
| `tests` | unit + integration tests; `conftest.py` wires `src` into `sys.path` | touch the network |

Allowed dependency direction: `tests → {models, data, utils}`,
`models → utils`, `data → utils`. Nothing imports from `tests`.

### 6.2 Planned Interfaces

Signatures below are the contract for implementation phases; names are
binding, internals are free.

```python
# src/data/loader.py
def generate_synthetic_loans(
    n_samples: int = 5000, default_rate: float = 0.12, seed: int = 0
) -> "pd.DataFrame": ...

def load_loan_data(source: str = "synthetic", seed: int = 0) -> "pd.DataFrame": ...

# src/data/preprocess.py
def build_preprocessor(n_components: int = 8) -> "sklearn.pipeline.Pipeline":
    """impute -> one-hot -> scale -> PCA; fit on TRAIN split only."""

def stratified_split(
    df: "pd.DataFrame", ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, "pd.DataFrame"]: ...

# src/models/feature_map.py
def data_reuploading_encoding(wires, x, layer: int, omega) -> None:
    """Applies E_l(x) of Section 4.2 on all wires."""

# src/models/ansatz.py
def variational_ansatz(wires, theta, layer: int) -> None:
    """Applies Theta_l of Section 4.3 on all wires."""

def ring_entangler(wires) -> None: ...

# src/models/circuit.py
def make_vqc_qnode(
    n_qubits: int = 4, n_layers: int = 4,
    device: str = "default.qubit", shots: int | None = None,
    diff_method: str = "default",
) -> "qml.QNode":
    """Returns the batched QNode x -> <Z_0>; params (theta, omega) bound."""

# src/models/hybrid.py
class HybridLoanClassifier:
    """sklearn-style wrapper: fit / predict / predict_proba / score."""
    def __init__(self, n_qubits=4, n_layers=4, lr=0.01, batch_size=32,
                 max_epochs=100, patience=10, seed=0, shots=None): ...
    def fit(self, X, y) -> "HybridLoanClassifier": ...
    def predict_proba(self, X) -> "np.ndarray": ...
    def predict(self, X, threshold: float = 0.5) -> "np.ndarray": ...
    def score(self, X, y) -> float: ...   # average precision on (X, y)

# src/models/baselines.py
def make_baselines(seed: int = 0) -> dict[str, "sklearn.base.ClassifierMixin"]: ...

# src/utils/metrics.py
def evaluate(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """average_precision (primary), roc_auc, f1, brier, confusion matrix."""

# src/utils/seed.py
def set_global_seed(seed: int) -> "np.random.Generator": ...

# src/utils/config.py
@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    n_qubits: int = 4
    n_layers: int = 4
    lr: float = 0.01
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    seed: int = 0
    shots: int | None = None
```

### 6.3 Devices and Interfaces (PennyLane 0.45)

- `qml.device("default.qubit", wires=n)` — exact statevector, deterministic
  given seed; default for development and CI.
- `qml.device("lightning.qubit", wires=n)` — high-performance simulator for
  \(n \ge 10\); ships with the pinned pennylane-lightning build.
- Shot mode: pass `shots=1024` to the device for sampling-noise studies;
  device RNG seeded from the global generator (P4).
- Trainable parameters must be `qml.numpy` (autograd-wrapped) arrays with
  `requires_grad=True`; input data stays plain NumPy.
- `qml.draw` is the debugging tool of record; unit tests assert gate
  placement against the specification of Sections 4.2–4.4.

### 6.4 Configuration

One frozen `TrainingConfig` dataclass per run (Section 6.2); every reported
result must be reproducible from `(config, seed)` alone. Configs are logged
with results; no global mutable state.

## 7. Evaluation Protocol

### 7.1 Splits

Stratified 70 / 15 / 15 train / validation / test. Headline numbers come
from 5-fold repeated stratified cross-validation; the test split is touched
once, at the end, by exactly one model per experiment.

### 7.2 Metrics

| Metric | Role |
|---|---|
| **Average precision (PR-AUC)** | **primary** — honest under class imbalance |
| ROC-AUC | secondary, reported for literature comparability |
| Recall @ precision ≥ 0.5 | business-relevant operating point |
| F1, confusion matrix | at threshold chosen on validation |
| Brier score | probability calibration quality |

Threshold selection happens on validation only; the fairness audit
(subgroup AP gaps across protected attributes when present) is a Phase-4
gate (Risk R6).

### 7.3 Baselines and Statistical Comparison

Every VQC report includes, run through the identical harness:

1. `DummyClassifier(strategy="prior")` — sanity floor.
2. `LogisticRegression` — linear classical reference.
3. `RandomForestClassifier` — nonlinear classical reference.

A "quantum wins" statement requires the VQC to beat the best baseline on
average precision with a paired-bootstrap significance test (1000 resamples,
\(\alpha = 0.05\)).

### 7.4 Reproducibility Requirements

- Single seed fans out to: data generation, splits, parameter init,
  batching, and device shot RNG.
- Exact simulator results must be bit-stable for fixed `(config, seed)`.
- CI runs the deterministic (shot-free) configuration only.

## 8. Performance and Scaling

- **Gradient cost dominates.** \((2P{+}1)B\) executions per step under
  parameter-shift; mitigations: backprop `diff_method` on simulators,
  freezing \(\omega\) after warm-up (halves \(P\)), shallow \(L\).
- **Statevector memory** grows as \(\mathcal{O}(2^n)\): `default.qubit`
  beyond ~12 qubits, then `lightning.qubit` (CPU) / GPU plugins.
- **Batching** PennyLane QNode calls amortizes Python dispatch overhead;
  always evaluate a mini-batch as one call, never per-sample loops.
- **Shot mode** trades gradient variance for runtime; when enabled, report
  shot budgets with results (Risk R5).

## 9. Risk Register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Barren plateaus / vanishing gradients | training stalls at chance | shallow \(L \le 6\); local readout; layer-wise training option; near-identity init |
| R2 | Class imbalance (~12% defaults) | misleading accuracy | class-weighted loss; AP as primary metric; validation-tuned threshold |
| R3 | Preprocessing leakage | silently inflated metrics | single Pipeline fit on train only (P2); leakage-guard unit test |
| R4 | Overfitting (64 params, small data) | poor generalization | early stopping; parameter budget heuristic \(4nL \le n_{\text{train}}/20\); baseline gates |
| R5 | Simulator–hardware gap | results don't transfer | shot-based noise studies; reported shot budgets; parameter-shift gradients throughout |
| R6 | Unfair / miscalibrated scores | regulatory & ethical exposure | subgroup metric audit; Brier score reporting; fairness-aware thresholding (extension) |
| R7 | Dependency drift | irreproducible environments | pinned `requirements.txt`; Python ≥ 3.12 floor documented |

## 10. Extension Points

- **Kernel route:** QSVM with a re-uploading fidelity kernel
  (`qml.kernels`) as an alternative to variational training.
- **Alternative ansätze:** `StronglyEntanglingLayers`, real-amplitude
  circuits — drop-in behind the same QNode factory contract.
- **Other interfaces:** PyTorch/JAX interfaces for GPU-accelerated classical
  optimization at scale.
- **Real datasets:** OpenML `credit-g` (German credit) behind
  `load_loan_data(source=...)`; LendingClub-style data via the file loader.
- **Hardware:** PennyLane device plugins keep the model code unchanged
  (P5); execution budgets and transpilation are the only new concerns.

## 11. References

1. A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre,
   *Data re-uploading on a quantum computer*, Phys. Rev. A **104**, 032416
   (2021). arXiv:2007.08306
2. M. Schuld, R. Sweke, J. J. Meyer, *Effect of data encoding on the
   expressive power of variational quantum machine learning models*,
   Phys. Rev. A **103**, 032430 (2021). arXiv:2008.08605
3. K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, *Quantum circuit learning*,
   Phys. Rev. A **98**, 032309 (2018). arXiv:1803.00745
4. M. Benedetti, S. Lloyd, S. Sorce, M. Fiorentini, *Parameterized quantum
   circuits as machine learning models*, Quantum Sci. Technol. **4**, 043001
   (2019). arXiv:1906.07682
5. J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven,
   *Barren plateaus in quantum neural network training landscapes*,
   Nat. Commun. **9**, 4812 (2018). arXiv:1803.11173
6. PennyLane documentation — <https://pennylane.ai/>
7. scikit-learn documentation — <https://scikit-learn.org/>

## 12. Revision History

| Version | Change |
|---|---|
| v1.0 | Initial specification: pipeline, circuit family, training loop, module contracts, evaluation protocol, risk register. |
