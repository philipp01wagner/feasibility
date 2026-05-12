"""Aggregate the synthetic thesis sweeps into two summary plots:
- mean regret + 95 % CI vs sigma_mu (robustness to task heterogeneity)
- mean regret + 95 % CI vs N_per_source (robustness to source-data budget)

Reads the regret_summary.csv / infeas_summary.csv files from each sweep
experiment folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEME_ORDER = [
    "Vanilla BO", "RGPE",
    "cBO target-clf", "cBO RGPE-vote", "cBO equal-vote",
]
COLORS = ["#2E86AB", "#E63946", "#06A77D", "#FFB400", "#7B2CBF"]

EXP_ROOT = Path("experiments")

# (label, x_value, folder_prefix) tuples
SIGMA_MU_SETTINGS = [
    (0.3, "sweep_sigma_0.3"),
    (0.5, "synthetic_thesis_main"),  # main run at sigma_mu=0.5
    (0.7, "sweep_sigma_0.7"),
    (1.0, "sweep_sigma_1.0"),
    (1.5, "sweep_sigma_1.5"),
]

N_PER_SOURCE_SETTINGS = [
    (10, "sweep_nsrc_10"),
    (25, "sweep_nsrc_25"),
    (50, "synthetic_thesis_main"),   # main run at N_PER_SOURCE=50
    (100, "sweep_nsrc_100"),
    (200, "sweep_nsrc_200"),
]


def _latest_folder(prefix: str) -> Path | None:
    candidates = sorted(EXP_ROOT.glob(f"{prefix}_*"))
    return candidates[-1] if candidates else None


def _read_setting(prefix: str) -> dict[str, dict[str, float]]:
    """Return {scheme -> {mean_regret, ci_lo, ci_hi, mean_infeas, infeas_lo, infeas_hi}}."""
    folder = _latest_folder(prefix)
    if folder is None:
        raise FileNotFoundError(f"No experiment folder matches {prefix}_*")
    regret = pd.read_csv(folder / "regret_summary.csv").set_index("scheme")
    infeas = pd.read_csv(folder / "infeas_summary.csv").set_index("scheme")
    out = {}
    for s in SCHEME_ORDER:
        out[s] = {
            "mean_regret": float(regret.loc[s, "mean_regret"]),
            "ci_lo": float(regret.loc[s, "ci_lo"]),
            "ci_hi": float(regret.loc[s, "ci_hi"]),
            "mean_infeas": float(infeas.loc[s, "mean_infeas"]),
            "infeas_lo": float(infeas.loc[s, "ci_lo"]),
            "infeas_hi": float(infeas.loc[s, "ci_hi"]),
        }
    return out


def plot_sweep(settings, x_label, fname_prefix, x_log=False):
    """Build a 2-panel figure: mean regret + mean infeasibility vs x."""
    x_vals = [s[0] for s in settings]
    data = {x: _read_setting(prefix) for x, prefix in settings}

    fig, (ax_r, ax_i) = plt.subplots(1, 2, figsize=(14, 5.5))

    for s, color in zip(SCHEME_ORDER, COLORS):
        regret_means = [data[x][s]["mean_regret"] for x in x_vals]
        regret_lo = [data[x][s]["ci_lo"] for x in x_vals]
        regret_hi = [data[x][s]["ci_hi"] for x in x_vals]
        infeas_means = [data[x][s]["mean_infeas"] for x in x_vals]
        infeas_lo = [data[x][s]["infeas_lo"] for x in x_vals]
        infeas_hi = [data[x][s]["infeas_hi"] for x in x_vals]

        ax_r.plot(x_vals, regret_means, "o-", color=color, linewidth=2.0,
                  markersize=7, label=s)
        ax_r.fill_between(x_vals, regret_lo, regret_hi,
                          color=color, alpha=0.18)

        ax_i.plot(x_vals, infeas_means, "o-", color=color, linewidth=2.0,
                  markersize=7, label=s)
        ax_i.fill_between(x_vals, infeas_lo, infeas_hi,
                          color=color, alpha=0.18)

    if x_log:
        ax_r.set_xscale("log")
        ax_i.set_xscale("log")
    ax_r.set_yscale("symlog", linthresh=1.0)
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel("Mean final regret (95 % bootstrap CI)")
    ax_r.set_title("Regret vs " + x_label)
    ax_r.legend(loc="best", fontsize=9)
    ax_r.grid(alpha=0.3, which="both")

    ax_i.set_xlabel(x_label)
    ax_i.set_ylabel("Mean cumulative infeasibles (95 % bootstrap CI)")
    ax_i.set_title("Infeasibility vs " + x_label)
    ax_i.legend(loc="best", fontsize=9)
    ax_i.grid(alpha=0.3)

    fig.suptitle(
        f"Synthetic Ackley TL-BO sweep: {x_label} "
        f"(10 targets x 10-20 seeds per setting)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{fname_prefix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname_prefix}.png")

    # also save the aggregated CSV
    rows = []
    for x in x_vals:
        for s in SCHEME_ORDER:
            d = data[x][s]
            rows.append({x_label.replace(" ", "_"): x, "scheme": s, **d})
    df = pd.DataFrame(rows)
    df.to_csv(f"{fname_prefix}.csv", index=False)
    print(f"Saved {fname_prefix}.csv")
    return df


def main():
    print("=== sigma_mu sweep ===")
    df_sigma = plot_sweep(
        SIGMA_MU_SETTINGS,
        x_label="sigma_mu",
        fname_prefix="experiments/aggregate_sigma_mu",
    )
    print(df_sigma.round(3).to_string(index=False))

    print("\n=== N_per_source sweep ===")
    df_nsrc = plot_sweep(
        N_PER_SOURCE_SETTINGS,
        x_label="N_per_source",
        fname_prefix="experiments/aggregate_n_per_source",
        x_log=True,
    )
    print(df_nsrc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
