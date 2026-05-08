"""Run/cache the PID 5-scheme sweep capturing per-iteration histories,
then build aggregate + per-target convergence plots inside a timestamped
experiment folder.

The PID bo_loop uses the standard Gardner-style constrained EI
(``acq = ei * p_feas``) for both RGPE-vote and equal-vote constraints.
There is no constraint-sharpness exponent here, unlike the laser-cutting
``tl_bayopt.bo_loop`` which exposes a ``beta`` argument.

Usage:
    python pid_convergence_plot.py [experiment_name]
"""

from __future__ import annotations

import datetime
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from tl_pid_bayopt import (
    COLORS,
    N_BO_ITER,
    N_TARGET_INIT,
    N_TASKS,
    SCHEMES,
    SEED,
    fit_gp,
    fit_source_classifiers,
    generate_tasks,
    sample_all_task_data,
    bo_loop,
)
import numpy as np  # noqa: F401  (kept after re-import for clarity)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HIST_PICKLE = "pid_convergence_histories.pkl"


def make_experiment_dir(name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("experiments") / f"{name}_{ts}"
    (root / "per_target").mkdir(parents=True, exist_ok=True)
    return root


def run_for_target(target_idx, tasks, all_X, all_y, all_stable, n_seeds):
    target_V, target_qm = tasks[target_idx]
    src_indices = [i for i in range(N_TASKS) if i != target_idx]
    src_X = [all_X[i] for i in src_indices]
    src_y = [all_y[i] for i in src_indices]
    src_stab = [all_stable[i] for i in src_indices]
    source_gps = [fit_gp(Xs, np.log1p(ys)) for Xs, ys in zip(src_X, src_y)]
    source_clfs = fit_source_classifiers(src_X, src_stab)

    out = {s: {"hist": [], "unstable": []} for s in SCHEMES}
    for s_idx in range(n_seeds):
        seed = SEED + 100 + s_idx
        for s in SCHEMES:
            hist, unstable = bo_loop(
                s, target_V, target_qm, source_gps, source_clfs, seed
            )
            out[s]["hist"].append(hist)
            out[s]["unstable"].append(unstable)
    return out


def collect(n_seeds=3):
    tasks = generate_tasks()
    print(f"Generated {len(tasks)} plant tasks (LHS in log-(V, qm))")
    for i, (V, qm) in enumerate(tasks):
        print(f"  Task {i:>2}: V = {V:7.2f} L,  qm = {qm:.4f} g/s")
    all_X, all_y, all_stable = sample_all_task_data(tasks)

    sweep = {}
    t_start = time.time()
    for ti in range(N_TASKS):
        t0 = time.time()
        sweep[ti] = run_for_target(ti, tasks, all_X, all_y, all_stable, n_seeds)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (N_TASKS - ti - 1)
        print(f"[{ti + 1:>2}/{N_TASKS}] target {ti} "
              f"(V={tasks[ti][0]:6.2f}L qm={tasks[ti][1]:.3f})  "
              f"{elapsed:5.0f}s  total {total / 60:5.1f}m  eta {eta / 60:5.1f}m",
              flush=True)
    with open(HIST_PICKLE, "wb") as f:
        pickle.dump({"tasks": tasks, "sweep": sweep, "n_seeds": n_seeds}, f)
    print(f"Saved {HIST_PICKLE}")
    return tasks, sweep, n_seeds


def plot_per_target(target_idx, target_V, target_qm, runs_by_scheme,
                    n_seeds, out_path):
    n_iters = N_BO_ITER + 1
    iters = np.arange(n_iters)
    fig, (ax_b, ax_u) = plt.subplots(1, 2, figsize=(13, 5))

    final_burrs = []
    final_unstables = []
    for s, name in SCHEMES.items():
        H = np.array(runs_by_scheme[s]["hist"])
        U = np.array(runs_by_scheme[s]["unstable"])
        mean_b = H.mean(axis=0); se_b = H.std(axis=0) / np.sqrt(max(H.shape[0], 1))
        mean_u = U.mean(axis=0); se_u = U.std(axis=0) / np.sqrt(max(U.shape[0], 1))
        ax_b.plot(iters, mean_b, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_b.fill_between(iters, mean_b - se_b, mean_b + se_b,
                          color=COLORS[s - 1], alpha=0.18)
        ax_u.plot(iters, mean_u, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_u.fill_between(iters, mean_u - se_u, mean_u + se_u,
                          color=COLORS[s - 1], alpha=0.18)
        final_burrs.append(f"{name}: {mean_b[-1]:.3f}")
        final_unstables.append(f"{name}: {mean_u[-1]:.1f}")

    ax_b.set_xlabel("BO iteration")
    ax_b.set_ylabel("Best ITAE so far")
    ax_b.set_yscale("log")
    ax_b.set_title("ITAE convergence")
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.grid(alpha=0.3, which="both")

    ax_u.set_xlabel("BO iteration")
    ax_u.set_ylabel("Cumulative unstable evals")
    ax_u.set_title("Unstable convergence")
    ax_u.legend(loc="upper left", fontsize=9)
    ax_u.grid(alpha=0.3)

    fig.suptitle(
        f"Target task {target_idx}  |  V={target_V:.2f} L  qm={target_qm:.4f} g/s\n"
        f"{n_seeds} seeds, {N_TARGET_INIT} init + {N_BO_ITER} BO iters",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(tasks, sweep, n_seeds, out_path):
    n_iters = N_BO_ITER + 1
    iters = np.arange(n_iters)

    # per-target normalize by per-target median initial best for ITAE
    burr_norm = {s: np.zeros(n_iters) for s in SCHEMES}
    burr_norm_se = {s: np.zeros(n_iters) for s in SCHEMES}
    unstable_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    unstable_se = {s: np.zeros(n_iters) for s in SCHEMES}

    target_means_burr = {s: [] for s in SCHEMES}
    target_means_un = {s: [] for s in SCHEMES}
    for ti in range(len(tasks)):
        baselines = []
        for s in SCHEMES:
            H = np.array(sweep[ti][s]["hist"])
            baselines.append(H[:, 0])
        baseline = float(np.median(np.concatenate(baselines)))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        for s in SCHEMES:
            H = np.array(sweep[ti][s]["hist"])
            U = np.array(sweep[ti][s]["unstable"])
            target_means_burr[s].append((H / baseline).mean(axis=0))
            target_means_un[s].append(U.mean(axis=0))

    for s in SCHEMES:
        TM_b = np.array(target_means_burr[s])
        TM_u = np.array(target_means_un[s])
        burr_norm[s] = TM_b.mean(axis=0)
        burr_norm_se[s] = TM_b.std(axis=0) / np.sqrt(len(tasks))
        unstable_curves[s] = TM_u.mean(axis=0)
        unstable_se[s] = TM_u.std(axis=0) / np.sqrt(len(tasks))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_n, ax_log, ax_un, ax_layout = axes.ravel()

    for s, name in SCHEMES.items():
        for ax in (ax_n, ax_log):
            ax.plot(iters, burr_norm[s], color=COLORS[s - 1],
                    linewidth=2.2, label=name)
            ax.fill_between(iters, burr_norm[s] - burr_norm_se[s],
                            burr_norm[s] + burr_norm_se[s],
                            color=COLORS[s - 1], alpha=0.18)
        ax_un.plot(iters, unstable_curves[s], color=COLORS[s - 1],
                   linewidth=2.2, label=name)
        ax_un.fill_between(iters, unstable_curves[s] - unstable_se[s],
                           unstable_curves[s] + unstable_se[s],
                           color=COLORS[s - 1], alpha=0.18)

    ax_n.set_xlabel("BO iteration")
    ax_n.set_ylabel("Best ITAE / median initial ITAE")
    ax_n.set_title(f"Aggregated normalized ITAE convergence "
                   f"({len(tasks)} targets)")
    ax_n.legend(loc="upper right", fontsize=9)
    ax_n.grid(alpha=0.3)

    ax_log.set_xlabel("BO iteration")
    ax_log.set_ylabel("Best ITAE / median initial ITAE (log)")
    ax_log.set_yscale("log")
    ax_log.set_title("Same, log-y axis")
    ax_log.legend(loc="upper right", fontsize=9)
    ax_log.grid(alpha=0.3, which="both")

    ax_un.set_xlabel("BO iteration")
    ax_un.set_ylabel("Cumulative unstable evals (mean across targets)")
    ax_un.set_title("Aggregated unstable convergence")
    ax_un.legend(loc="upper left", fontsize=9)
    ax_un.grid(alpha=0.3)

    # plant-task layout
    for i, (V, qm) in enumerate(tasks):
        ax_layout.scatter(V, qm, color="#444", s=70, edgecolors="black",
                          linewidths=0.7, zorder=3)
        ax_layout.annotate(str(i), (V, qm), textcoords="offset points",
                           xytext=(6, 6), fontsize=10)
    ax_layout.set_xscale("log"); ax_layout.set_yscale("log")
    ax_layout.set_xlabel("Chamber volume V (L)")
    ax_layout.set_ylabel("Mass inflow qm (g/s)")
    ax_layout.set_title(f"{len(tasks)} plant tasks (LHS in log-(V, qm))")
    ax_layout.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"PID best-ITAE per BO iteration - 5-method comparison, "
        f"{n_seeds} seeds, {N_TARGET_INIT} init + {N_BO_ITER} BO iters",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "pid_5method"
    exp_dir = make_experiment_dir(exp_name)
    print(f"Experiment folder: {exp_dir}")

    if Path(HIST_PICKLE).exists():
        print(f"Loading cached histories from {HIST_PICKLE}")
        with open(HIST_PICKLE, "rb") as f:
            d = pickle.load(f)
        tasks, sweep, n_seeds = d["tasks"], d["sweep"], d.get("n_seeds", 3)
    else:
        tasks, sweep, n_seeds = collect()
    if Path(HIST_PICKLE).exists():
        shutil.copy(HIST_PICKLE, exp_dir / "histories.pkl")

    plot_aggregate(tasks, sweep, n_seeds, exp_dir / "convergence_aggregate.png")
    print(f"Saved {exp_dir / 'convergence_aggregate.png'}")

    print(f"Writing per-target plots into {exp_dir / 'per_target'}/...")
    for ti in range(len(tasks)):
        V, qm = tasks[ti]
        out = (exp_dir / "per_target"
               / f"target_{ti:02d}_V{V:06.2f}_qm{qm:.4f}.png")
        plot_per_target(ti, V, qm, sweep[ti], n_seeds, out)
    print(f"  Wrote {len(tasks)} per-target plots.")
    print(f"\nAll outputs in: {exp_dir}")


if __name__ == "__main__":
    main()
