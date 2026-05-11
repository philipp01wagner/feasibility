# Synthetic Transfer-Learning Constrained-BO Benchmark

A closed-form 2-D family of related minimisation tasks with binary
feasibility labels. Designed for unit-testing and ablation of constrained
Bayesian optimisation methods that exploit source data, including the
RGPE-vote and equal-vote schemes documented in
`constrained_bo_transfer.md`.

Implementation: `synthetic_benchmark.py` (pure NumPy, no fitted models).

---

## 1. Why a synthetic benchmark?

Real-world TL-BO benchmarks (laser cutting, PID tuning) mix the effect
under study (transfer + feasibility) with confounders -- noisy
oracles, unknown ground truth, distributional shift of unknown
magnitude. A purely mathematical benchmark removes these:

- **Closed-form, deterministic oracle.** No surrogate fitting, no
  random measurement noise, no bias from data preprocessing.
- **Known ground truth.** A grid search recovers the constrained
  global minimum for every task, enabling regret evaluation.
- **Tunable task similarity.** Two scalar parameters control how
  related the tasks are (Section 3.2), so studies can sweep from
  "near-identical" to "loosely related" task families.
- **Tunable feasibility difficulty.** Three parameters control the
  size and roughness of the feasible region (Section 3.3).
- **Visualisable.** 2-D inputs make sanity-checking surrogates and
  acquisitions easy.

---

## 2. Domain and notation

We work in two dimensions on the symmetric Ackley domain

$$
\mathcal{X} \;=\; [-5, 5] \times [-5, 5] \;\subset\; \mathbb{R}^2.
$$

A point is $\mathbf{x} = (x_1, x_2)\in\mathcal{X}$. We define $T \ge 2$
**tasks**, each indexed by $t \in \{0, \ldots, T-1\}$, generated from a
shared meta-distribution.

Per-task parameters are bundled as

$$
\boldsymbol{\theta}_t = \big(\boldsymbol{\mu}_t,\, \boldsymbol{\nu}_t,\, a_t,\, b_t,\, \gamma_t\big),\qquad \boldsymbol{\mu}_t,\boldsymbol{\nu}_t \in \mathbb{R}^2,\;\, a_t,b_t,\gamma_t > 0.
$$

These five quantities ($\boldsymbol{\mu}_t$ and $\boldsymbol{\nu}_t$
are 2-vectors; $a_t,b_t,\gamma_t$ are scalars) are sampled once at
construction time (Section 3) and remain fixed for the lifetime of the
benchmark.

---

## 3. Task generation

### 3.1 Objective

The per-task objective is the **shifted Ackley function**

$$
f_t(\mathbf{x}) \;=\; \mathrm{Ackley}\!\big(\mathbf{x} - \boldsymbol{\mu}_t\big),
$$

where the standard 2-D Ackley function is

$$
\mathrm{Ackley}(\mathbf{u}) \;=\; -a\,\exp\!\left(-b\,\sqrt{\tfrac{1}{2}(u_1^2 + u_2^2)}\right) \;-\; \exp\!\left(\tfrac{1}{2}\big(\cos(c u_1) + \cos(c u_2)\big)\right) \;+\; a \;+\; e
$$

with constants $a = 20$, $b = 0.2$, $c = 2\pi$. Ackley has a **single**
global minimum at the origin with $f^\star = 0$, surrounded by a dense
lattice of shallow local minima from the cosine term. Shifting by
$\boldsymbol{\mu}_t$ translates the global minimum so that
$f_t(\boldsymbol{\mu}_t) = 0$.

### 3.2 Feasibility

The smooth **feasibility score** is

$$
h_t(\mathbf{x}) \;=\; 1 \;-\; \left(\frac{x_1 - \nu_{t,1}}{a_t}\right)^2 \;-\; \left(\frac{x_2 - \nu_{t,2}}{b_t}\right)^2 \;+\; \gamma_t\,\sin\!\left(\frac{2\pi (x_1 + x_2)}{T_p}\right),
$$

with $T_p$ a fixed spatial period (default $T_p = 2.5$). The binary
feasibility label is

$$
c_t(\mathbf{x}) \;=\; \mathbb{I}\!\left[\,h_t(\mathbf{x}) \ge 0\,\right] \;\in\; \{0, 1\}.
$$

Geometrically, the feasible set $\{\mathbf{x} : h_t(\mathbf{x}) \ge 0\}$
is an axis-aligned ellipsoid centred at $\boldsymbol{\nu}_t$ with
semi-axes $(a_t, b_t)$, perturbed by a sinusoidal ripple of amplitude
$\gamma_t$. The ripple makes the boundary non-convex and the feasible
region non-simply-connected for large $\gamma_t$, which prevents the
problem from collapsing into a trivially separable structure.

### 3.3 Meta-distribution

Task parameters are drawn i.i.d. from

$$
\boldsymbol{\mu}_t \;\sim\; \mathcal{N}(\bar{\mathbf{x}},\; \sigma_\mu^2 I_2),
$$

$$
\boldsymbol{\nu}_t \;\mid\; \boldsymbol{\mu}_t \;\sim\; \mathcal{N}(\boldsymbol{\mu}_t,\; \sigma_\nu^2 I_2),
$$

$$
a_t \sim \mathcal{U}(a_{\min}, a_{\max}),\quad b_t \sim \mathcal{U}(b_{\min}, b_{\max}),\quad \gamma_t \sim \mathcal{U}(\gamma_{\min}, \gamma_{\max}),
$$

where $\bar{\mathbf{x}} = (0, 0)$ is the centre of the domain. The
**conditional** distribution of $\boldsymbol{\nu}_t$ given
$\boldsymbol{\mu}_t$ is the key TL design decision: it makes
feasibility centres co-vary with objective optima, so source tasks
informative about $\boldsymbol{\mu}_t$ are also informative about
$\boldsymbol{\nu}_t$.

The two scalar knobs that drive the **transfer-difficulty axis** are

| symbol | meaning | small value | large value |
|---|---|---|---|
| $\sigma_\mu$ | spread of objective optima | tasks share an optimum | optima spread across domain |
| $\sigma_\nu$ | conditional spread of feasibility centres | feasibility tracks objective tightly | feasibility decorrelates from objective |

For source tasks to be useful, $\sigma_\mu, \sigma_\nu$ should be
**small enough that the optima cluster** but **large enough that the
target's optimum is not exactly at any source's**. The defaults
$\sigma_\mu = 1.0$, $\sigma_\nu = 0.7$ achieve this on the $[-5, 5]^2$
domain (~10 %/7 % of the domain width).

---

## 4. Default parameters

| symbol | default | comment |
|---|---|---|
| $T$ | $10$ | number of tasks |
| $\bar{\mathbf{x}}$ | $(0, 0)$ | domain centre |
| $\sigma_\mu$ | $1.0$ | objective-shift std-dev |
| $\sigma_\nu$ | $0.7$ | feasibility-centre conditional std-dev |
| $a_{\min}, a_{\max}$ | $2.0,\; 3.5$ | ellipsoid semi-axis range (x1) |
| $b_{\min}, b_{\max}$ | $2.0,\; 3.5$ | ellipsoid semi-axis range (x2) |
| $\gamma_{\min}, \gamma_{\max}$ | $0.1,\; 0.4$ | boundary-ripple amplitude range |
| $T_p$ | $2.5$ | ripple spatial period |
| seed | $42$ | RNG seed (deterministic across runs) |

These defaults yield a feasibility coverage of roughly 30–50 % of the
domain and put the constrained minimum well inside the feasible region
on most tasks.

---

## 5. API

```python
from synthetic_benchmark import SyntheticTLBenchmark
import numpy as np

bench = SyntheticTLBenchmark(n_tasks=10, seed=42)

# Single-point oracle
y, c = bench.query(task_idx=0, x=np.array([1.0, 4.0]))

# Batched oracle
X = bench.sample_inputs(50, method="sobol")  # 50 Sobol points
y, c = bench.query(0, X)

# Source datasets for TL-BO (skips target)
sources = bench.make_source_datasets(target_idx=9, n_per_source=50)
# sources: dict {task_idx: (X, y, c)}

# Ground truth
f_star, x_star = bench.true_constrained_minimum(task_idx=9, n_grid=401)
f_un, x_un    = bench.true_unconstrained_minimum(task_idx=9)

# Visualisation
bench.plot_task(task_idx=0)
```

The class has no I/O dependencies; it can be imported in any test or
notebook.

---

## 6. Properties useful for unit-testing

### 6.1 Known optima

Because the objective is a shifted Ackley, the **unconstrained**
minimum value is the constant $f^\star_{\rm un} = 0$ and its location
is exactly $\boldsymbol{\mu}_t$. The **constrained** minimum is found
by gridding the domain and taking the lowest objective value over
feasible cells; the helper `true_constrained_minimum(task_idx, n_grid)`
does this with a $401 \times 401$ grid by default (gives $\sim 0.025$
unit resolution on $[-5, 5]^2$).

### 6.2 Feasibility coverage

For each task, the proportion of the domain that is feasible is

$$
\rho_t \;=\; \frac{|\{\mathbf{x}\in\mathcal{X} : h_t(\mathbf{x}) \ge 0\}|}{|\mathcal{X}|}.
$$

With the default ellipsoid axes $a_t, b_t \sim \mathcal{U}(2, 3.5)$ on
the $10\times 10$ box, $\rho_t$ ranges over roughly $[0.20, 0.60]$.
Sources of feasibility-rate variation match the laser-cutting setting
where $\rho$ varies from $\sim 12\%$ to $\sim 26\%$ across tasks (see
`constrained_bo_transfer.md`, §8).

### 6.3 Transfer signal: rank correlation across tasks

A useful sanity check is the **objective rank correlation** between
two tasks:

$$
r_{t,t'} \;=\; \mathrm{Spearman}\!\big(f_t(X_{\rm test}),\; f_{t'}(X_{\rm test})\big)
$$

for a shared random test set $X_{\rm test}$. Under the default meta-distribution this should be high ($r \approx 0.7$–$0.9$) for most
task pairs, dropping to $r \approx 0.3$–$0.5$ between the two
tasks whose $\boldsymbol{\mu}_t$ are farthest apart. RGPE rank-loss
weights should reflect this ordering when used on this benchmark.

---

## 7. Suggested experiments

### 7.1 Sanity check: oracle behaviour

Inspect each task with `bench.plot_task(t)`. The constrained minimum
(white star) should sit inside the red feasibility contour; the
unconstrained minimum (orange plus) may or may not.

### 7.2 Transfer-vs-no-transfer

Compare on each task as target:

- Vanilla BO (target GP only)
- RGPE
- Constrained BO with target classifier
- Constrained BO with RGPE-weighted vote
- Constrained BO with equal-weight vote

Report median final regret $f^\star_{\rm method} - f^\star_{\rm true}$
and number of infeasible queries. Defaults that match the laser-cutting
study: $N_{\rm init} = 5$, $N_{\rm BO\,iter} = 25$, $N_{\rm seeds} = 5$.

### 7.3 Task-similarity sweep

Vary $\sigma_\mu \in \{0.5, 1.0, 1.5, 2.5, 4.0\}$ at fixed $\sigma_\nu$.
The transfer benefit should peak at small-to-moderate $\sigma_\mu$
(homogeneous task family) and vanish at large $\sigma_\mu$ (no useful
prior) -- a textbook negative-transfer curve.

### 7.4 Source-data-quantity sweep

For a fixed task family, vary $N_{\rm per\,source} \in \{10, 25, 50,
100, 200\}$. The PID study (`pressure_simulation/tl_pid_bayopt.py`)
showed an interior optimum around $50$; this benchmark should
reproduce that pattern with no need to wait for ChamberEnv physics.

### 7.5 Constraint-sharpness sweep

Hold the surrogate fixed (RGPE) and vary the constraint exponent
$\beta$ in $\alpha_\beta = \alpha_{\rm EI} \cdot p_{\rm feas}^\beta$ over
$\{0, 0.25, 0.5, 1, 2, 4, 8\}$. This benchmark provides a closed-form
ground truth, so the Pareto plot of mean infeasibility vs final regret
becomes an exact rather than empirical curve. Compare with the same
sweep on the laser-cutting data (see
`bayopt_beta_sweep.png`).

---

## 8. Limitations and extensions

The 2-D Ackley family is deliberately small. Three extensions are
straightforward in the same code:

1. **Higher dimension.** `_ackley` reads $d$ from `x.shape[-1]` and so
   generalises directly; widen `DOMAIN_LO/HI` to the desired $d$-vectors.
   For a different landscape, swap in a Hartmann-3 / Hartmann-6 function
   and shift it by a $d$-vector $\boldsymbol{\mu}_t$.
   The feasibility score generalises to $h_t(\mathbf{x}) = 1 -
   \sum_{i=1}^d ((x_i - \nu_{t,i})/a_{t,i})^2 + \gamma_t\,\sin(\cdot)$.

2. **Heteroscedastic noise.** Wrap `query()` in a Gaussian-noise layer
   to study robustness to observation noise; also add Bernoulli flips
   to $c_t$ for studying classifier label noise (cf. the
   noise-robustness experiment in `tl_experiments.py`).

3. **Adversarial source.** Insert a task with anti-correlated $\mu_t$
   to test whether RGPE-weighted voting downweights it appropriately
   relative to equal voting. This is the diagnostic that distinguishes
   the two voting schemes most sharply.

The canonical reference implementation is
`synthetic_benchmark.py`; the demo at the bottom of that module
generates a 10-panel figure (`synthetic_benchmark_tasks.png`) showing
all default tasks side-by-side, with constrained and unconstrained
optima marked.
