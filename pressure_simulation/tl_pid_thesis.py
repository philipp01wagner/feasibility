"""Thesis-grade PID sweep: main + sigma-equivalent + N_per_source sweeps.

Mirrors ``tl_synthetic_bayopt.py`` for the synthetic Ackley case. Builds an
experiment folder per run, saves histories, regret/infeasibility CSVs with
95 % bootstrap CIs, pairwise paired-Wilcoxon stats, and per-target plots.

Usage:
    python tl_pid_thesis.py [experiment_name]

Tunable via env vars (so the same code drives multiple settings):
    N_SEEDS         : number of paired BO runs per (target, scheme)
    N_PER_SOURCE    : LHS samples per source task
    TASK_SPREAD     : log-(V, qm) box scale around the geometric centre
                      (1.0 = full physical range)
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
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from tl_pid_bayopt import (
    COLORS,
    N_BO_ITER,
    N_PER_SOURCE,
    N_SEEDS,
    N_TARGET_INIT,
    N_TASKS,
    SCHEMES,
    SEED,
    TASK_SPREAD,
    bo_loop,
    fit_gp,
    fit_source_classifiers,
    generate_tasks,
    sample_all_task_data,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------- experiment folder -------------------------------------
def make_experiment_dir(name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("experiments") / f"{name}_{ts}"
    (root / "per_target").mkdir(parents=True, exist_ok=True)
    return root


# ----------------- per-target driver -------------------------------------
def run_for_target(target_idx, tasks, all_X, all_y, all_stable):
    target_V, target_qm = tasks[target_idx]
    src_indices = [i for i in range(N_TASKS) if i != target_idx]
    src_X = [all_X[i] for i in src_indices]
    src_y = [all_y[i] for i in src_indices]
    src_stab = [all_stable[i] for i in src_indices]
    source_gps = [fit_gp(Xs, np.log1p(ys)) for Xs, ys in zip(src_X, src_y)]
    source_clfs = fit_source_classifiers(src_X, src_stab)
    out = {s: {"hist": [], "unstable": []} for s in SCHEMES}
    for run in range(N_SEEDS):
        seed = SEED + 100 + run
        for s in SCHEMES:
            hist, unstable = bo_loop(
                s, target_V, target_qm, source_gps, source_clfs, seed
            )
            out[s]["hist"].append(hist)
            out[s]["unstable"].append(unstable)
    return out


# ----------------- statistical analysis (paired) -------------------------
def _bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, rng_seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng_b = np.random.default_rng(rng_seed)
    idx = rng_b.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def statistical_analysis(sweep, target_best_itae, exp_dir):
    """Paired Wilcoxon + bootstrap CI over (target, seed) on best ITAE
    and on cumulative unstable count.

    Regret-like metric: best_ITAE - per_target_floor, where the floor is the
    best ITAE any (scheme, seed) achieved on that target (proxy for true min
    since closed-form is unavailable).
    """
    from scipy.stats import wilcoxon

    scheme_keys = list(SCHEMES.keys())
    scheme_names = [SCHEMES[s] for s in scheme_keys]
    n_schemes = len(scheme_keys)

    final_regret = np.zeros((N_TASKS, N_SEEDS, n_schemes), dtype=float)
    final_infeas = np.zeros((N_TASKS, N_SEEDS, n_schemes), dtype=float)
    for ti in range(N_TASKS):
        floor = target_best_itae[ti]
        for si, s in enumerate(scheme_keys):
            H = np.array(sweep[ti][s]["hist"])
            U = np.array(sweep[ti][s]["unstable"])
            best = H[:, -1]
            best = np.where(np.isfinite(best), best, 1e6)  # clip inf
            final_regret[ti, :, si] = best - floor
            final_infeas[ti, :, si] = U[:, -1]

    pooled_regret = final_regret.reshape(-1, n_schemes)
    pooled_infeas = final_infeas.reshape(-1, n_schemes)
    n_paired = pooled_regret.shape[0]

    lines = []
    lines.append("\n-- Mean final regret (ITAE - target floor) across all "
                 f"targets x seeds (N={n_paired} paired observations) --")
    lines.append(f"  {'scheme':<22} {'mean regret':>11}  "
                 f"{'95% bootstrap CI':>22}  {'mean unstable':>13}  "
                 f"{'95% CI':>16}")
    regret_summary, infeas_summary = [], []
    for si, name in enumerate(scheme_names):
        m_r, lo_r, hi_r = _bootstrap_mean_ci(pooled_regret[:, si])
        m_i, lo_i, hi_i = _bootstrap_mean_ci(pooled_infeas[:, si])
        regret_summary.append((name, m_r, lo_r, hi_r))
        infeas_summary.append((name, m_i, lo_i, hi_i))
        lines.append(f"  {name:<22} {m_r:>11.3f}  "
                     f"[{lo_r:>8.3f}, {hi_r:>8.3f}]  "
                     f"{m_i:>13.2f}  [{lo_i:>5.2f},{hi_i:>5.2f}]")

    # Pairwise paired Wilcoxon on pooled regret + Holm correction.
    lines.append("\n-- Pairwise paired Wilcoxon signed-rank tests on regret "
                 "(Holm-Bonferroni corrected) --")
    pairs, raw_p, diffs = [], [], []
    for i in range(n_schemes):
        for j in range(i + 1, n_schemes):
            d = pooled_regret[:, i] - pooled_regret[:, j]
            d = d[np.isfinite(d)]
            if d.size < 5 or np.all(d == 0):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(d)
                except ValueError:
                    p = 1.0
            pairs.append((i, j)); raw_p.append(p); diffs.append(d)
    raw_p = np.array(raw_p)
    order = np.argsort(raw_p)
    n_tests = len(raw_p)
    adjusted_p = np.empty_like(raw_p)
    max_so_far = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, raw_p[idx] * (n_tests - rank))
        max_so_far = max(max_so_far, adj)
        adjusted_p[idx] = max_so_far

    lines.append(f"  {'scheme A':<22} {'scheme B':<22} "
                 f"{'mean A-B':>10} {'median A-B':>11} {'p (Holm)':>10} {'sig':>4}")
    pair_rows = []
    for (i, j), p_adj, d in zip(pairs, adjusted_p, diffs):
        mean_d = float(d.mean()) if d.size else float("nan")
        med_d = float(np.median(d)) if d.size else float("nan")
        sig = (
            "***" if p_adj < 0.001 else
            "**" if p_adj < 0.01 else
            "*" if p_adj < 0.05 else
            "ns"
        )
        lines.append(f"  {scheme_names[i]:<22} {scheme_names[j]:<22} "
                     f"{mean_d:>10.4f} {med_d:>11.4f} {p_adj:>10.2e} {sig:>4}")
        pair_rows.append({
            "scheme_A": scheme_names[i],
            "scheme_B": scheme_names[j],
            "mean_diff": mean_d,
            "median_diff": med_d,
            "p_raw": raw_p[pairs.index((i, j))],
            "p_holm": p_adj,
            "sig": sig,
        })

    pd.DataFrame(regret_summary,
                 columns=["scheme", "mean_regret", "ci_lo", "ci_hi"]
                 ).to_csv(exp_dir / "regret_summary.csv", index=False)
    pd.DataFrame(infeas_summary,
                 columns=["scheme", "mean_infeas", "ci_lo", "ci_hi"]
                 ).to_csv(exp_dir / "infeas_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(exp_dir / "pairwise_wilcoxon.csv", index=False)

    return "\n".join(lines)


# ----------------- per-target plot ---------------------------------------
def plot_per_target(ti, tasks, runs_by_scheme, target_floor, out_path):
    n_iters = N_BO_ITER + 1
    iters = np.arange(n_iters)
    fig, (ax_b, ax_u) = plt.subplots(1, 2, figsize=(13, 5))
    V, qm = tasks[ti]
    for s, name in SCHEMES.items():
        H = np.array(runs_by_scheme[s]["hist"])
        H = np.where(np.isfinite(H), H, 1e6)
        U = np.array(runs_by_scheme[s]["unstable"])
        mean_b = H.mean(axis=0); se_b = H.std(axis=0) / np.sqrt(max(H.shape[0], 1))
        mean_u = U.mean(axis=0); se_u = U.std(axis=0) / np.sqrt(max(U.shape[0], 1))
        ax_b.plot(iters, mean_b, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_b.fill_between(iters, mean_b - se_b, mean_b + se_b,
                          color=COLORS[s - 1], alpha=0.18)
        ax_u.plot(iters, mean_u, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_u.fill_between(iters, mean_u - se_u, mean_u + se_u,
                          color=COLORS[s - 1], alpha=0.18)
    ax_b.axhline(target_floor, color="black", linestyle="--", alpha=0.5,
                 label=f"empirical floor = {target_floor:.3f}")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("BO iteration"); ax_b.set_ylabel("Best ITAE so far")
    ax_b.set_title(f"ITAE convergence (target {ti})")
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.grid(alpha=0.3, which="both")
    ax_u.set_xlabel("BO iteration")
    ax_u.set_ylabel("Cumulative unstable evals")
    ax_u.set_title("Unstable-eval convergence")
    ax_u.legend(loc="upper left", fontsize=9)
    ax_u.grid(alpha=0.3)
    fig.suptitle(f"Target {ti}  |  V={V:.2f}L  qm={qm:.4f}g/s  |  "
                 f"N_seeds={N_SEEDS}, N_per_source={N_PER_SOURCE}, "
                 f"spread={TASK_SPREAD}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_aggregate(tasks, sweep, out_path):
    n_iters = N_BO_ITER + 1
    iters = np.arange(n_iters)
    burr_norm = {s: np.zeros(n_iters) for s in SCHEMES}
    burr_se = {s: np.zeros(n_iters) for s in SCHEMES}
    unstable_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    unstable_se = {s: np.zeros(n_iters) for s in SCHEMES}
    means_per_target_b = {s: [] for s in SCHEMES}
    means_per_target_u = {s: [] for s in SCHEMES}
    for ti in range(N_TASKS):
        baselines = []
        for s in SCHEMES:
            H = np.array(sweep[ti][s]["hist"])
            baselines.append(H[:, 0])
        baseline = float(np.median(np.concatenate(baselines)))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        for s in SCHEMES:
            H = np.array(sweep[ti][s]["hist"])
            H = np.where(np.isfinite(H), H, 1e6)
            U = np.array(sweep[ti][s]["unstable"])
            means_per_target_b[s].append((H / baseline).mean(axis=0))
            means_per_target_u[s].append(U.mean(axis=0))
    for s in SCHEMES:
        TM_b = np.array(means_per_target_b[s])
        TM_u = np.array(means_per_target_u[s])
        burr_norm[s] = TM_b.mean(axis=0); burr_se[s] = TM_b.std(axis=0) / np.sqrt(N_TASKS)
        unstable_curves[s] = TM_u.mean(axis=0)
        unstable_se[s] = TM_u.std(axis=0) / np.sqrt(N_TASKS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_n, ax_log, ax_u, ax_lay = axes.ravel()
    for s, name in SCHEMES.items():
        for ax in (ax_n, ax_log):
            ax.plot(iters, burr_norm[s], color=COLORS[s - 1],
                    linewidth=2.2, label=name)
            ax.fill_between(iters, burr_norm[s] - burr_se[s],
                            burr_norm[s] + burr_se[s],
                            color=COLORS[s - 1], alpha=0.18)
        ax_u.plot(iters, unstable_curves[s], color=COLORS[s - 1],
                  linewidth=2.2, label=name)
        ax_u.fill_between(iters, unstable_curves[s] - unstable_se[s],
                          unstable_curves[s] + unstable_se[s],
                          color=COLORS[s - 1], alpha=0.18)
    ax_n.set_xlabel("BO iteration"); ax_n.set_ylabel("Best ITAE / median initial ITAE")
    ax_n.set_title(f"Aggregated normalized convergence ({N_TASKS} targets)")
    ax_n.legend(loc="upper right", fontsize=9); ax_n.grid(alpha=0.3)
    ax_log.set_xlabel("BO iteration"); ax_log.set_ylabel("Best ITAE / median initial ITAE (log)")
    ax_log.set_yscale("log"); ax_log.set_title("Same, log-y axis")
    ax_log.legend(loc="upper right", fontsize=9); ax_log.grid(alpha=0.3, which="both")
    ax_u.set_xlabel("BO iteration"); ax_u.set_ylabel("Cumulative unstable evals")
    ax_u.set_title("Aggregated unstable convergence")
    ax_u.legend(loc="upper left", fontsize=9); ax_u.grid(alpha=0.3)
    for ti, (V, qm) in enumerate(tasks):
        ax_lay.scatter(V, qm, color="#444", s=80, edgecolors="black",
                       linewidths=0.7)
        ax_lay.annotate(str(ti), (V, qm), textcoords="offset points",
                        xytext=(6, 6), fontsize=10)
    ax_lay.set_xscale("log"); ax_lay.set_yscale("log")
    ax_lay.set_xlabel("V (L)"); ax_lay.set_ylabel("qm (g/s)")
    ax_lay.set_title(f"{N_TASKS} plant tasks (spread={TASK_SPREAD})")
    ax_lay.grid(alpha=0.3, which="both")
    fig.suptitle(f"PID thesis sweep | spread={TASK_SPREAD}  "
                 f"N_per_source={N_PER_SOURCE}  N_seeds={N_SEEDS}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ----------------- main --------------------------------------------------
def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "pid_thesis"
    exp_dir = make_experiment_dir(exp_name)
    print(f"Experiment folder: {exp_dir}")
    print(f"Config: N_SEEDS={N_SEEDS}, N_PER_SOURCE={N_PER_SOURCE}, "
          f"TASK_SPREAD={TASK_SPREAD}")

    tasks = generate_tasks(spread=TASK_SPREAD)
    print(f"\nGenerated {N_TASKS} plant tasks (spread={TASK_SPREAD}):")
    for ti, (V, qm) in enumerate(tasks):
        print(f"  Task {ti}: V = {V:7.2f} L,  qm = {qm:.4f} g/s")

    all_X, all_y, all_stable = sample_all_task_data(tasks)

    print(f"\nSweep: {N_TASKS} targets x {len(SCHEMES)} schemes x "
          f"{N_SEEDS} runs x {N_BO_ITER} BO iters")
    sweep = {}
    t_start = time.time()
    for ti in range(N_TASKS):
        t0 = time.time()
        sweep[ti] = run_for_target(ti, tasks, all_X, all_y, all_stable)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (N_TASKS - ti - 1)
        finals = {s: float(np.mean(np.array(sweep[ti][s]["hist"])[:, -1]))
                  for s in SCHEMES}
        unstables = {s: float(np.mean(np.array(sweep[ti][s]["unstable"])[:, -1]))
                     for s in SCHEMES}
        print(f"[{ti + 1:>2}/{N_TASKS}] target {ti}  {elapsed:5.0f}s  "
              f"total {total / 60:5.1f}m  eta {eta / 60:5.1f}m", flush=True)
        for s in SCHEMES:
            print(f"        {SCHEMES[s]:<22} ITAE={finals[s]:8.4f}  "
                  f"unstable={unstables[s]:5.2f}", flush=True)

    # Save histories.
    with open(exp_dir / "histories.pkl", "wb") as f:
        pickle.dump({"sweep": sweep, "tasks": tasks,
                     "n_tasks": N_TASKS, "n_seeds": N_SEEDS,
                     "n_per_source": N_PER_SOURCE,
                     "task_spread": TASK_SPREAD}, f)
    print(f"\nSaved {exp_dir / 'histories.pkl'}")

    # Empirical per-target floor for regret computation
    target_floor = np.zeros(N_TASKS)
    for ti in range(N_TASKS):
        all_finals = []
        for s in SCHEMES:
            H = np.array(sweep[ti][s]["hist"])
            H = np.where(np.isfinite(H), H, 1e6)
            all_finals.append(H[:, -1])
        target_floor[ti] = float(np.concatenate(all_finals).min())

    # Statistical analysis
    print("\n=== Statistical analysis (paired across targets x seeds) ===")
    stat_report = statistical_analysis(sweep, target_floor, exp_dir)
    print(stat_report)

    # Plots
    plot_aggregate(tasks, sweep, exp_dir / "convergence_aggregate.png")
    print(f"Saved {exp_dir / 'convergence_aggregate.png'}")
    print(f"Writing per-target plots...")
    for ti in range(N_TASKS):
        plot_per_target(ti, tasks, sweep[ti], target_floor[ti],
                        exp_dir / "per_target" / f"target_{ti:02d}.png")
    print(f"  Wrote {N_TASKS} per-target plots.")
    print(f"\nAll outputs in: {exp_dir}")


if __name__ == "__main__":
    main()
