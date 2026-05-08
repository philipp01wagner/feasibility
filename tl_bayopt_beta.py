"""Beta sweep for the laser-cutting constrained-BO across all targets.

Same setup as tl_bayopt.py's all-target sweep, but instead of comparing five
schemes we fix the surrogate (RGPE) and the constraint source (RGPE-weighted
vote of source feasibility classifiers, scheme 4) and tune the constraint
sharpness via the exponent

    acq = EI * p_feas ** beta

beta = 0 reduces to plain EI (no constraint == scheme 2). Larger beta sharpens
the multiplier toward a hard mask. We sweep a small grid and aggregate
final-burr / miscut-count statistics across the same valid targets used in
tl_bayopt.py.

Run from project root:
    python tl_bayopt_beta.py
"""

from __future__ import annotations

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Tighter for the sweep — beta grid * targets * runs is already large
N_RUNS = 3
BETA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def run_target(target_name, all_gps, all_clfs):
    """Run scheme=4 (RGPE-vote constraint) for each beta + the two baselines."""
    df = task_df_dict[target_name]
    lo = df[FEATURE_COLS].min().values
    hi = df[FEATURE_COLS].max().values
    bounds = np.column_stack([lo, hi])
    sim = create_task_simulator(target_name, task_df_dict)
    src_gps = {k: v for k, v in all_gps.items() if k != target_name}
    src_clfs = {k: v for k, v in all_clfs.items() if k != target_name}

    out = {("baseline", "vanilla"): {"hist": [], "miscuts": []},
           ("baseline", "rgpe"): {"hist": [], "miscuts": []}}
    for b in BETA_VALUES:
        out[("beta", b)] = {"hist": [], "miscuts": []}

    for run in range(N_RUNS):
        seed = SEED + run
        # Baselines (cheap, share the same seed init for paired comparison)
        h, m = bo_loop(1, seed, sim, bounds, src_gps, src_clfs)
        out[("baseline", "vanilla")]["hist"].append(h)
        out[("baseline", "vanilla")]["miscuts"].append(m)
        h, m = bo_loop(2, seed, sim, bounds, src_gps, src_clfs)
        out[("baseline", "rgpe")]["hist"].append(h)
        out[("baseline", "rgpe")]["miscuts"].append(m)
        # Beta sweep on scheme 4
        for b in BETA_VALUES:
            h, m = bo_loop(4, seed, sim, bounds, src_gps, src_clfs, beta=b)
            out[("beta", b)]["hist"].append(h)
            out[("beta", b)]["miscuts"].append(m)
    return out


def main():
    targets = []
    for name, df in task_df_dict.items():
        if name in OUTLIER_TARGETS:
            continue
        has_pos = ((df["burr_evaluated"] >= 0) & (df["roughness_z_evaluated"] >= 0)).any()
        has_neg = ((df["burr_evaluated"] < 0) | (df["roughness_z_evaluated"] < 0)).any()
        if has_pos and has_neg and len(df) >= 8:
            targets.append(name)
    print(f"Found {len(targets)} valid target tasks "
          f"(both classes, n>=8, {len(OUTLIER_TARGETS)} OOD outliers excluded).")

    print("Pre-fitting GPs and classifiers for all tasks...")
    t0 = time.time()
    all_gps, all_clfs = fit_all_task_models()
    print(f"  {len(all_gps)} GPs, {len(all_clfs)} classifiers ({time.time() - t0:.1f}s)")

    print(f"\nBeta sweep: {len(targets)} targets x "
          f"({len(BETA_VALUES)} betas + 2 baselines) x "
          f"{N_RUNS} runs x {N_ITER} iters")
    print(f"Beta grid: {BETA_VALUES}\n")

    all_results = {}
    t_start = time.time()
    for ti, target in enumerate(targets):
        t0 = time.time()
        all_results[target] = run_target(target, all_gps, all_clfs)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (len(targets) - ti - 1)
        print(f"[{ti + 1:>2}/{len(targets)}] {target[:50]:<50} "
              f"{elapsed:5.0f}s  total {total / 60:5.1f}m  eta {eta / 60:5.1f}m")

    # ---- Aggregate ----
    schemes = (
        [("baseline", "vanilla"), ("baseline", "rgpe")]
        + [("beta", b) for b in BETA_VALUES]
    )
    labels = (
        ["Vanilla BO", "RGPE"]
        + [f"beta={b:g}" for b in BETA_VALUES]
    )

    burr_table = {lab: [] for lab in labels}
    miscut_table = {lab: [] for lab in labels}
    for target in targets:
        for s, lab in zip(schemes, labels):
            H = np.array(all_results[target][s]["hist"])
            M = np.array(all_results[target][s]["miscuts"])
            burr_table[lab].append(float(H[:, -1].mean()))
            miscut_table[lab].append(float(M[:, -1].mean()))

    target_short = [t[:48] for t in targets]
    burr_df = pd.DataFrame(burr_table, index=target_short)
    miscut_df = pd.DataFrame(miscut_table, index=target_short)
    burr_df.to_csv("bayopt_beta_sweep_burr.csv")
    miscut_df.to_csv("bayopt_beta_sweep_miscuts.csv")
    print("\nSaved bayopt_beta_sweep_burr.csv, bayopt_beta_sweep_miscuts.csv")

    print("\n=== Final best burr per target (mean over runs) ===")
    print(burr_df.round(1).to_string())
    print("\n=== Cumulative miscuts per target (mean over runs) ===")
    print(miscut_df.round(2).to_string())

    print("\n=== Aggregate (mean across targets) ===")
    print(f"  {'scheme':<14} {'mean burr':>12} {'mean miscuts':>14}")
    for lab in labels:
        mb = float(np.mean(burr_table[lab]))
        mm = float(np.mean(miscut_table[lab]))
        print(f"  {lab:<14} {mb:>12.1f} {mm:>14.2f}")

    # ---- Plot: Pareto + trade-off vs beta ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mean_burr = np.array([np.mean(burr_table[l]) for l in labels])
    mean_miscuts = np.array([np.mean(miscut_table[l]) for l in labels])

    # Pareto scatter
    ax = axes[0]
    # baseline points
    ax.scatter(mean_miscuts[0], mean_burr[0], s=140, marker="s",
               color="#2E86AB", edgecolors="black", linewidths=0.7,
               label="Vanilla BO")
    ax.scatter(mean_miscuts[1], mean_burr[1], s=140, marker="D",
               color="#E63946", edgecolors="black", linewidths=0.7,
               label="RGPE (no constraint)")
    # beta sweep
    beta_burr = mean_burr[2:]
    beta_miscuts = mean_miscuts[2:]
    ax.plot(beta_miscuts, beta_burr, "-", color="#444", alpha=0.4, zorder=1)
    sc = ax.scatter(beta_miscuts, beta_burr,
                    c=BETA_VALUES, cmap="viridis",
                    s=140, edgecolors="black", linewidths=0.7,
                    zorder=2, label="cBO RGPE-vote (beta sweep)")
    for b, mc, bu in zip(BETA_VALUES, beta_miscuts, beta_burr):
        ax.annotate(f"  beta={b:g}", (mc, bu), fontsize=9)
    cbar = plt.colorbar(sc, ax=ax, label="beta")
    ax.set_xlabel("Mean cumulative miscuts (across targets)")
    ax.set_ylabel("Mean final best burr (across targets)")
    ax.set_title(f"Pareto: constraint strength (beta) trade-off, {len(targets)} targets")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Trade-off curves vs beta (twin axis)
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(BETA_VALUES, beta_burr, "o-", color="#E63946", linewidth=2,
            label="mean burr")
    ax2.plot(BETA_VALUES, beta_miscuts, "s-", color="#06A77D", linewidth=2,
             label="mean miscuts")
    # Reference lines for vanilla and RGPE-no-constraint
    ax.axhline(mean_burr[0], color="#2E86AB", linestyle="--",
               alpha=0.5, label="Vanilla BO burr")
    ax.axhline(mean_burr[1], color="#E63946", linestyle=":",
               alpha=0.4, label="RGPE burr (beta=0 ref)")
    ax2.axhline(mean_miscuts[0], color="#2E86AB", linestyle="--", alpha=0.3)
    ax2.axhline(mean_miscuts[1], color="#06A77D", linestyle=":", alpha=0.3)
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlabel("beta (constraint sharpness exponent)")
    ax.set_ylabel("Mean final best burr", color="#E63946")
    ax2.set_ylabel("Mean cumulative miscuts", color="#06A77D")
    ax.tick_params(axis="y", labelcolor="#E63946")
    ax2.tick_params(axis="y", labelcolor="#06A77D")
    ax.set_title("Trade-off vs beta")
    ax.grid(alpha=0.3)
    # combined legend
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="best", fontsize=9)

    fig.suptitle(
        f"Constraint-strength sweep on cBO RGPE-vote, {len(targets)} targets, "
        f"{N_RUNS} runs x {N_ITER} iters",
        fontsize=12,
    )
    fig.tight_layout()
    out = "bayopt_beta_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
