"""Re-run the 5-scheme + beta=1 sweep capturing full per-iteration histories,
then build dedicated burr-convergence plots (aggregate + one per target)
inside a timestamped experiment folder.

Usage:
    python convergence_plot.py [experiment_name]
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

from cutting_simulator import create_task_simulator
from tl_bayopt import (
    FEATURE_COLS,
    INFEASIBLE_PENALTY,
    N_INIT,
    N_ITER,
    OUTLIER_TARGETS,
    SEED,
    bo_loop,
    fit_all_task_models,
    task_df_dict,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_RUNS = 3
SCHEMES = {
    1: "Vanilla BO",
    2: "RGPE",
    3: "cBO target-clf",
    4: "cBO RGPE-vote (beta=1)",
    5: "cBO equal-vote",
}
COLORS = ["#2E86AB", "#E63946", "#06A77D", "#FFB400", "#7B2CBF"]
HIST_PICKLE = "convergence_histories.pkl"


def run_target(target_name, all_gps, all_clfs):
    df = task_df_dict[target_name]
    lo = df[FEATURE_COLS].min().values
    hi = df[FEATURE_COLS].max().values
    bounds = np.column_stack([lo, hi])
    sim = create_task_simulator(target_name, task_df_dict)
    src_gps = {k: v for k, v in all_gps.items() if k != target_name}
    src_clfs = {k: v for k, v in all_clfs.items() if k != target_name}

    out = {s: {"hist": []} for s in SCHEMES}
    for run in range(N_RUNS):
        seed = SEED + run
        for s in SCHEMES:
            h, _ = bo_loop(s, seed, sim, bounds, src_gps, src_clfs)
            out[s]["hist"].append(h)
    return out


def collect():
    targets = []
    for name, df in task_df_dict.items():
        if name in OUTLIER_TARGETS:
            continue
        has_pos = ((df["burr_evaluated"] >= 0) & (df["roughness_z_evaluated"] >= 0)).any()
        has_neg = ((df["burr_evaluated"] < 0) | (df["roughness_z_evaluated"] < 0)).any()
        if has_pos and has_neg and len(df) >= 8:
            targets.append(name)
    print(f"Found {len(targets)} valid targets.")

    all_gps, all_clfs = fit_all_task_models()
    print(f"Pre-fit: {len(all_gps)} GPs, {len(all_clfs)} classifiers.")

    sweep = {}
    t_start = time.time()
    for ti, target in enumerate(targets):
        t0 = time.time()
        sweep[target] = run_target(target, all_gps, all_clfs)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (len(targets) - ti - 1)
        print(f"[{ti + 1:>2}/{len(targets)}] {target[:50]:<50} "
              f"{elapsed:5.0f}s  total {total / 60:5.1f}m  eta {eta / 60:5.1f}m",
              flush=True)
    with open(HIST_PICKLE, "wb") as f:
        pickle.dump({"targets": targets, "sweep": sweep}, f)
    print(f"Saved {HIST_PICKLE}")
    return targets, sweep


def make_experiment_dir(name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("experiments") / f"{name}_{ts}"
    (root / "per_target").mkdir(parents=True, exist_ok=True)
    return root


def plot_per_target(target, runs_by_scheme, out_path):
    """Single-target convergence plot, all 5 schemes, mean +/- SE."""
    n_iters = N_ITER + 1
    iters = np.arange(n_iters)
    fig, ax = plt.subplots(figsize=(9, 6))

    final_text = []
    for s, name in SCHEMES.items():
        H = np.array(runs_by_scheme[s]["hist"])
        H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
        mean = H.mean(axis=0)
        se = H.std(axis=0) / np.sqrt(max(H.shape[0], 1))
        ax.plot(iters, mean, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax.fill_between(iters, mean - se, mean + se,
                        color=COLORS[s - 1], alpha=0.18)
        final_text.append(f"{name}: {mean[-1]:.1f}")

    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best feasible burr (absolute)")
    ax.set_title(f"Target task: {target}\n"
                 f"final mean burr  |  " + "  |  ".join(final_text),
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot(targets, sweep, exp_dir: Path):
    n_iters = N_ITER + 1
    iters = np.arange(n_iters)

    # Panel A: aggregated *normalized* convergence (per-target divided by median initial best)
    norm_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    norm_se = {s: np.zeros(n_iters) for s in SCHEMES}
    raw_per_target = {s: [] for s in SCHEMES}
    for target in targets:
        baselines = []
        for s in SCHEMES:
            H = np.array(sweep[target][s]["hist"])
            baselines.append(H[:, 0])
        baseline = float(np.median(np.concatenate(baselines)))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        for s in SCHEMES:
            H = np.array(sweep[target][s]["hist"])
            H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
            raw_per_target[s].append((target, H / baseline))

    # mean and SE across targets of the per-target run-mean curves
    for s in SCHEMES:
        target_means = np.array([rh.mean(axis=0) for _, rh in raw_per_target[s]])
        norm_curves[s] = target_means.mean(axis=0)
        norm_se[s] = target_means.std(axis=0) / np.sqrt(len(targets))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_norm, ax_log, ax_spaghetti, ax_h450 = axes.ravel()

    # Panel A: aggregated normalized
    for s, name in SCHEMES.items():
        ax_norm.plot(iters, norm_curves[s], color=COLORS[s - 1], linewidth=2.2,
                     label=name)
        ax_norm.fill_between(iters, norm_curves[s] - norm_se[s],
                             norm_curves[s] + norm_se[s],
                             color=COLORS[s - 1], alpha=0.18)
    ax_norm.set_xlabel("BO iteration")
    ax_norm.set_ylabel("Best burr / median initial burr")
    ax_norm.set_title(f"Aggregated normalized convergence ({len(targets)} targets, "
                      f"{N_RUNS} runs/target)")
    ax_norm.legend(loc="upper right", fontsize=10)
    ax_norm.grid(alpha=0.3)

    # Panel B: same but log-y
    for s, name in SCHEMES.items():
        ax_log.plot(iters, norm_curves[s], color=COLORS[s - 1], linewidth=2.2,
                    label=name)
        ax_log.fill_between(iters, norm_curves[s] - norm_se[s],
                            norm_curves[s] + norm_se[s],
                            color=COLORS[s - 1], alpha=0.18)
    ax_log.set_xlabel("BO iteration")
    ax_log.set_ylabel("Best burr / median initial burr (log scale)")
    ax_log.set_yscale("log")
    ax_log.set_title("Same, log-y axis")
    ax_log.legend(loc="upper right", fontsize=10)
    ax_log.grid(alpha=0.3, which="both")

    # Panel C: per-target spaghetti for vanilla vs RGPE-vote (beta=1)
    # Show every target's mean curve as a faint line, plus the aggregate
    for s in (1, 4):
        for target, rh in raw_per_target[s]:
            ax_spaghetti.plot(iters, rh.mean(axis=0), color=COLORS[s - 1],
                              linewidth=0.7, alpha=0.25)
        ax_spaghetti.plot(iters, norm_curves[s], color=COLORS[s - 1],
                          linewidth=2.5, label=SCHEMES[s])
    ax_spaghetti.set_xlabel("BO iteration")
    ax_spaghetti.set_ylabel("Best burr / median initial burr")
    ax_spaghetti.set_title("Per-target spaghetti: Vanilla vs RGPE-vote (beta=1)")
    ax_spaghetti.legend(loc="upper right", fontsize=10)
    ax_spaghetti.grid(alpha=0.3)

    # Panel D: a single hard target (H450) with absolute burr
    h450 = "150_ST150MD0-N2H0-30-2_L76_0.4_10000_H450"
    if h450 in sweep:
        for s, name in SCHEMES.items():
            H = np.array(sweep[h450][s]["hist"])
            H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
            mean = H.mean(axis=0); se = H.std(axis=0) / np.sqrt(N_RUNS)
            ax_h450.plot(iters, mean, color=COLORS[s - 1], linewidth=2.2, label=name)
            ax_h450.fill_between(iters, mean - se, mean + se,
                                 color=COLORS[s - 1], alpha=0.18)
        ax_h450.set_xlabel("BO iteration")
        ax_h450.set_ylabel("Best feasible burr (absolute)")
        ax_h450.set_title(f"Single target: {h450[-25:]}")
        ax_h450.legend(loc="upper right", fontsize=9)
        ax_h450.grid(alpha=0.3)

    fig.suptitle("Best burr per BO iteration - 5-method comparison "
                 "(beta=1 for RGPE-vote AND equal-vote constraints)",
                 fontsize=13)
    fig.tight_layout()
    aggregate_out = exp_dir / "convergence_aggregate.png"
    fig.savefig(aggregate_out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {aggregate_out}")

    # --- per-target plots ---
    print(f"Writing per-target plots into {exp_dir / 'per_target'}/...")
    for ti, target in enumerate(targets):
        # filename-safe with index for ordering
        safe = target.replace("/", "_").replace(" ", "_")
        out = exp_dir / "per_target" / f"target_{ti:02d}_{safe}.png"
        plot_per_target(target, sweep[target], out)
    print(f"  Wrote {len(targets)} per-target plots.")


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "bayopt_5method"
    exp_dir = make_experiment_dir(exp_name)
    print(f"Experiment folder: {exp_dir}")

    import os
    if os.path.exists(HIST_PICKLE):
        print(f"Loading cached histories from {HIST_PICKLE}")
        with open(HIST_PICKLE, "rb") as f:
            d = pickle.load(f)
        targets, sweep = d["targets"], d["sweep"]
    else:
        targets, sweep = collect()
    # copy the histories pickle into the experiment folder for reproducibility
    if os.path.exists(HIST_PICKLE):
        shutil.copy(HIST_PICKLE, exp_dir / "histories.pkl")

    plot(targets, sweep, exp_dir)
    print(f"\nAll outputs in: {exp_dir}")


if __name__ == "__main__":
    main()
