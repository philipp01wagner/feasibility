"""Synthetic transfer-learning constrained-BO benchmark.

A family of related 2-D minimisation tasks with a binary feasibility label.
Each task is a shifted Branin objective with a perturbed-ellipsoid feasibility
region. Task parameters are sampled from a meta-distribution so the resulting
tasks are similar but not identical -- the canonical TL setting.

Default oracle returns ``(y, c)`` for a task index ``t`` and an input
``x in [-5, 10] x [0, 15]`` (Branin's standard domain). All evaluations are
deterministic and closed-form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskParams:
    """Per-task parameter bundle.

    mu       -- shift of the objective (translates Branin minima)
    nu       -- centre of the feasibility ellipsoid
    a, b     -- semi-axes of the feasibility ellipsoid
    gamma    -- amplitude of the sinusoidal feasibility perturbation
    """

    mu: np.ndarray   # shape (2,)
    nu: np.ndarray   # shape (2,)
    a: float
    b: float
    gamma: float


class SyntheticTLBenchmark:
    """Transfer-learning constrained-BO benchmark.

    The objective on task t is

        f_t(x) = Branin(x - mu_t)

    and the feasibility indicator is

        c_t(x) = 1  iff  h_t(x) >= 0

    where the smooth feasibility score is

        h_t(x) = 1 - ((x1 - nu_{t,1}) / a_t)^2
                   - ((x2 - nu_{t,2}) / b_t)^2
                   + gamma_t * sin(2*pi*(x1 + x2) / period)

    (an ellipsoidal disk with sinusoidal perturbations on its boundary).

    Parameters
    ----------
    n_tasks : int
        Number of tasks to generate.
    sigma_mu : float
        Std-dev of objective-shift Gaussian.
    sigma_nu : float
        Std-dev of feasibility-centre noise around the objective shift.
    a_range, b_range : tuple
        Min/max ellipsoid semi-axes drawn uniformly per task.
    gamma_range : tuple
        Perturbation amplitudes drawn uniformly per task. Larger -> rougher
        feasibility boundary.
    period : float
        Spatial period of the boundary perturbation.
    seed : int
        RNG seed for task generation. Reproducible.
    """

    # Branin's standard domain.
    DOMAIN_LO = np.array([-5.0, 0.0])
    DOMAIN_HI = np.array([10.0, 15.0])
    DIM = 2

    def __init__(
        self,
        n_tasks: int = 10,
        sigma_mu: float = 1.5,
        sigma_nu: float = 1.0,
        a_range: tuple[float, float] = (3.0, 5.5),
        b_range: tuple[float, float] = (3.0, 5.5),
        gamma_range: tuple[float, float] = (0.1, 0.4),
        period: float = 4.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.n_tasks = n_tasks
        self.period = float(period)

        domain_centre = 0.5 * (self.DOMAIN_LO + self.DOMAIN_HI)
        mu = domain_centre[None, :] + sigma_mu * rng.standard_normal((n_tasks, 2))
        nu = mu + sigma_nu * rng.standard_normal((n_tasks, 2))
        a = rng.uniform(a_range[0], a_range[1], size=n_tasks)
        b = rng.uniform(b_range[0], b_range[1], size=n_tasks)
        gamma = rng.uniform(gamma_range[0], gamma_range[1], size=n_tasks)

        self.tasks: list[TaskParams] = [
            TaskParams(mu=mu[t], nu=nu[t], a=float(a[t]),
                       b=float(b[t]), gamma=float(gamma[t]))
            for t in range(n_tasks)
        ]

    # ----- core math -------------------------------------------------------
    @staticmethod
    def _branin(x: np.ndarray) -> np.ndarray:
        """Standard 2-D Branin function. x shape (..., 2)."""
        a, b = 1.0, 5.1 / (4.0 * np.pi ** 2)
        c, r = 5.0 / np.pi, 6.0
        s, t = 10.0, 1.0 / (8.0 * np.pi)
        x1 = x[..., 0]
        x2 = x[..., 1]
        return (
            a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
            + s * (1.0 - t) * np.cos(x1)
            + s
        )

    def evaluate(self, task_idx: int, x: np.ndarray) -> np.ndarray:
        """Objective value f_t(x). x can be (2,) or (N, 2)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        p = self.tasks[task_idx]
        shifted = x - p.mu[None, :]
        out = self._branin(shifted)
        return out if x.shape[0] > 1 else float(out[0])

    def feasibility_score(self, task_idx: int, x: np.ndarray) -> np.ndarray:
        """Continuous feasibility score h_t(x); >= 0 means feasible."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        p = self.tasks[task_idx]
        dx = x - p.nu[None, :]
        base = 1.0 - (dx[..., 0] / p.a) ** 2 - (dx[..., 1] / p.b) ** 2
        ripple = p.gamma * np.sin(
            2.0 * np.pi * (x[..., 0] + x[..., 1]) / self.period
        )
        out = base + ripple
        return out if x.shape[0] > 1 else float(out[0])

    def is_feasible(self, task_idx: int, x: np.ndarray) -> np.ndarray:
        """Boolean feasibility c_t(x). Same shape as evaluate."""
        s = self.feasibility_score(task_idx, x)
        return np.asarray(s) >= 0.0

    def query(self, task_idx: int, x: np.ndarray):
        """Return (y, c) for x on task t. Shape-preserving."""
        return self.evaluate(task_idx, x), self.is_feasible(task_idx, x)

    # ----- helpers for BO loops -------------------------------------------
    def domain(self):
        """(LO, HI) box bounds, shape (2,) each."""
        return self.DOMAIN_LO.copy(), self.DOMAIN_HI.copy()

    def sample_inputs(self, n: int, rng: np.random.Generator | None = None,
                      method: str = "uniform") -> np.ndarray:
        """Draw n inputs from the domain. method in {"uniform", "sobol"}."""
        if rng is None:
            rng = np.random.default_rng()
        if method == "uniform":
            return rng.uniform(self.DOMAIN_LO, self.DOMAIN_HI, size=(n, 2))
        if method == "sobol":
            from scipy.stats.qmc import Sobol
            seed = int(rng.integers(0, 2 ** 31 - 1))
            unit = Sobol(d=2, seed=seed).random(n)
            return self.DOMAIN_LO + unit * (self.DOMAIN_HI - self.DOMAIN_LO)
        raise ValueError(f"unknown method: {method}")

    def make_source_datasets(
        self,
        target_idx: int,
        n_per_source: int = 50,
        seed: int = 0,
        method: str = "sobol",
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Pre-sample (X, y, c) for every task except target_idx."""
        rng = np.random.default_rng(seed)
        out = {}
        for t in range(self.n_tasks):
            if t == target_idx:
                continue
            X = self.sample_inputs(n_per_source, rng=rng, method=method)
            y, c = self.query(t, X)
            out[t] = (X, np.asarray(y), np.asarray(c))
        return out

    # ----- ground truth ---------------------------------------------------
    def true_constrained_minimum(
        self, task_idx: int, n_grid: int = 401
    ) -> tuple[float, np.ndarray] | None:
        """Brute-force grid search for the constrained global minimum.

        Returns (f_star, x_star) or None if no feasible grid point is found.
        """
        x1 = np.linspace(self.DOMAIN_LO[0], self.DOMAIN_HI[0], n_grid)
        x2 = np.linspace(self.DOMAIN_LO[1], self.DOMAIN_HI[1], n_grid)
        XX, YY = np.meshgrid(x1, x2, indexing="xy")
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        y = self.evaluate(task_idx, pts)
        c = self.is_feasible(task_idx, pts)
        feas_idx = np.where(c)[0]
        if not feas_idx.size:
            return None
        best = feas_idx[np.argmin(y[feas_idx])]
        return float(y[best]), pts[best].copy()

    def true_unconstrained_minimum(self, task_idx: int) -> tuple[float, np.ndarray]:
        """Branin's three minima all at f=0.397887. Return one shifted."""
        # one of Branin's analytic minima, shifted by mu_t
        x_star_unshifted = np.array([np.pi, 2.275])
        return 0.39788735772973816, x_star_unshifted + self.tasks[task_idx].mu

    # ----- visualisation --------------------------------------------------
    def plot_task(self, task_idx: int, ax=None, n_grid: int = 200,
                  show_feasibility: bool = True, show_optima: bool = True,
                  log_objective: bool = True):
        """Contour plot of objective + feasibility region for one task."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))
        x1 = np.linspace(self.DOMAIN_LO[0], self.DOMAIN_HI[0], n_grid)
        x2 = np.linspace(self.DOMAIN_LO[1], self.DOMAIN_HI[1], n_grid)
        XX, YY = np.meshgrid(x1, x2)
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        y = self.evaluate(task_idx, pts).reshape(XX.shape)
        h = self.feasibility_score(task_idx, pts).reshape(XX.shape)

        Z = np.log10(y + 1.0) if log_objective else y
        cs = ax.contourf(XX, YY, Z, levels=30, cmap="viridis")
        ax.contour(XX, YY, Z, levels=10, colors="white",
                   linewidths=0.5, alpha=0.4)
        plt.colorbar(cs, ax=ax,
                     label="log10(f + 1)" if log_objective else "f")

        if show_feasibility:
            ax.contourf(XX, YY, (h >= 0).astype(float),
                        levels=[0.5, 1.5], colors="none",
                        hatches=["//"], alpha=0.0)
            ax.contour(XX, YY, h, levels=[0.0], colors="red",
                       linewidths=2.0)

        if show_optima:
            f_star, x_star = self.true_constrained_minimum(task_idx)
            ax.plot(x_star[0], x_star[1], "*", color="white",
                    markeredgecolor="black", markersize=18,
                    label=f"constrained min (f={f_star:.3f})")
            f_un, x_un = self.true_unconstrained_minimum(task_idx)
            ax.plot(x_un[0], x_un[1], "+", color="orange",
                    markeredgecolor="black", markersize=14,
                    label=f"unconstrained min (f={f_un:.3f})")
            ax.legend(loc="upper right", fontsize=9)

        p = self.tasks[task_idx]
        ax.set_title(
            f"Task {task_idx}: mu=({p.mu[0]:.2f},{p.mu[1]:.2f}), "
            f"nu=({p.nu[0]:.2f},{p.nu[1]:.2f}), "
            f"a={p.a:.2f}, b={p.b:.2f}, gamma={p.gamma:.2f}"
        )
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        return ax


# ---- demo --------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bench = SyntheticTLBenchmark(n_tasks=10, seed=42)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for t in range(10):
        bench.plot_task(t, ax=axes.ravel()[t])
    fig.suptitle(
        "Synthetic constrained TL-BO benchmark - 10 related tasks "
        "(red = feasibility boundary, white star = constrained min)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("synthetic_benchmark_tasks.png", dpi=120,
                bbox_inches="tight")
    print("Saved synthetic_benchmark_tasks.png")
