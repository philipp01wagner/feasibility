"""Combine the 5-scheme sweep and the beta sweep into a unified comparison.

The 5-scheme sweep (bayopt_all_targets_*.csv) has Vanilla, RGPE, cBO target-clf,
cBO RGPE-vote (beta=1), cBO equal-vote. The beta sweep
(bayopt_beta_sweep_*.csv) varies the constraint sharpness for the RGPE-vote
constraint scheme. The shared point is "cBO RGPE-vote" == "beta=1".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

methods_burr = pd.read_csv("bayopt_all_targets_burr.csv", index_col=0)
methods_miscuts = pd.read_csv("bayopt_all_targets_miscuts.csv", index_col=0)
beta_burr = pd.read_csv("bayopt_beta_sweep_burr.csv", index_col=0)
beta_miscuts = pd.read_csv("bayopt_beta_sweep_miscuts.csv", index_col=0)

# Sanity: cBO RGPE-vote in the methods sweep should be ~ beta=1 in beta sweep.
diff = (methods_burr["cBO RGPE-vote"].values - beta_burr["beta=1"].values)
print("Sanity check (5-scheme cBO RGPE-vote vs beta=1):")
print(f"  burr: max abs diff = {np.max(np.abs(diff)):.2f}")
print(f"  miscuts: max abs diff = "
      f"{np.max(np.abs(methods_miscuts['cBO RGPE-vote'].values - beta_miscuts['beta=1'].values)):.2f}")

# Build unified table.
unified_burr = methods_burr.copy()
unified_miscuts = methods_miscuts.copy()
unified_burr.rename(columns={"cBO RGPE-vote": "cBO RGPE-vote (beta=1)"}, inplace=True)
unified_miscuts.rename(columns={"cBO RGPE-vote": "cBO RGPE-vote (beta=1)"}, inplace=True)

method_order = [
    "Vanilla BO",
    "RGPE",
    "cBO target-clf",
    "cBO RGPE-vote (beta=1)",
    "cBO equal-vote",
]
colors = ["#2E86AB", "#E63946", "#06A77D", "#FFB400", "#7B2CBF"]

print("\n=== Aggregate (mean across 21 targets) ===")
print(f"  {'method':<28} {'mean burr':>10} {'mean miscuts':>14}")
for m in method_order:
    print(f"  {m:<28} {unified_burr[m].mean():>10.1f} "
          f"{unified_miscuts[m].mean():>14.2f}")

# Win counts.
def share_wins(df):
    v = df.values
    is_min = v == v.min(axis=1, keepdims=True)
    return (is_min / is_min.sum(axis=1, keepdims=True)).sum(axis=0)


burr_wins = share_wins(unified_burr[method_order])
miscut_wins = share_wins(unified_miscuts[method_order])

print("\n=== Win counts (tied wins shared, out of 21 targets) ===")
print(f"  {'method':<28} {'burr wins':>10} {'miscut wins':>13}")
for m, bw, mw in zip(method_order, burr_wins, miscut_wins):
    print(f"  {m:<28} {bw:>10.2f} {mw:>13.2f}")

# Build plot: 4 panels.
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
ax_pareto, ax_box, ax_bw, ax_mw = axes.ravel()

# ---- (1) Pareto: 5 methods + beta sweep ----
mean_burr = np.array([unified_burr[m].mean() for m in method_order])
mean_miscuts = np.array([unified_miscuts[m].mean() for m in method_order])
markers = ["s", "D", "^", "*", "o"]
sizes = [140, 140, 140, 200, 140]
for m, c, mk, s, mb, mc in zip(
    method_order, colors, markers, sizes, mean_burr, mean_miscuts
):
    ax_pareto.scatter(mc, mb, color=c, marker=mk, s=s,
                      edgecolors="black", linewidths=0.7, label=m, zorder=3)
    ax_pareto.annotate(f"  {m}", (mc, mb), fontsize=8)

# Overlay the beta sweep curve to show where beta=1 sits on the Pareto front.
betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
beta_mb = np.array([beta_burr[f"beta={b:g}"].mean() for b in betas])
beta_mc = np.array([beta_miscuts[f"beta={b:g}"].mean() for b in betas])
ax_pareto.plot(beta_mc, beta_mb, "--", color="gray", alpha=0.5,
               linewidth=1.5, zorder=1, label="beta sweep on RGPE-vote")
sc = ax_pareto.scatter(beta_mc, beta_mb, c=betas, cmap="viridis",
                       s=60, edgecolors="black", linewidths=0.4, zorder=2)
plt.colorbar(sc, ax=ax_pareto, label="beta", shrink=0.7)
ax_pareto.set_xlabel("Mean cumulative miscuts (across targets)")
ax_pareto.set_ylabel("Mean final best burr (across targets)")
ax_pareto.set_title("Pareto: 5 methods + beta sweep on RGPE-vote constraint")
ax_pareto.legend(loc="upper right", fontsize=8)
ax_pareto.grid(alpha=0.3)

# ---- (2) Per-target burr distribution ----
data = [unified_burr[m].values for m in method_order]
bp = ax_box.boxplot(data, tick_labels=method_order, patch_artist=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax_box.set_yscale("log")
ax_box.set_ylabel("Final best burr (per target, log scale)")
ax_box.set_title("Per-target final-burr distribution (21 targets)")
plt.setp(ax_box.get_xticklabels(), rotation=20, ha="right", fontsize=9)
ax_box.grid(alpha=0.3, axis="y", which="both")

# ---- (3) Burr win count ----
ax_bw.bar(range(len(method_order)), burr_wins, color=colors)
ax_bw.set_xticks(range(len(method_order)))
ax_bw.set_xticklabels(method_order, rotation=20, ha="right", fontsize=9)
ax_bw.set_ylabel("# targets won (ties shared)")
ax_bw.set_title("Burr win count")
ax_bw.grid(alpha=0.3, axis="y")

# ---- (4) Miscut win count ----
ax_mw.bar(range(len(method_order)), miscut_wins, color=colors)
ax_mw.set_xticks(range(len(method_order)))
ax_mw.set_xticklabels(method_order, rotation=20, ha="right", fontsize=9)
ax_mw.set_ylabel("# targets won (ties shared)")
ax_mw.set_title("Miscut win count")
ax_mw.grid(alpha=0.3, axis="y")

fig.suptitle(
    f"All-methods comparison (beta=1 for RGPE-vote constraint), 21 targets",
    fontsize=13,
)
fig.tight_layout()
out = "compare_all_methods.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved {out}")
