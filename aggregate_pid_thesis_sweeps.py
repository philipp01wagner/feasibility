"""Aggregate the PID thesis sweeps into two summary plots.

- regret vs TASK_SPREAD (robustness to chamber-family heterogeneity)
- regret vs N_per_source (source-data budget sensitivity)

Reads regret_summary.csv / infeas_summary.csv from each experiment folder.
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

SPREAD_SETTINGS = [
    (0.3, "pid_sweep_spread_0.3"),
    (0.7, "pid_sweep_spread_0.7"),
    (1.0, "pid_thesis_main"),       # main run at spread=1.0
]

NSRC_SETTINGS = [
    (10, "pid_sweep_nsrc_10"),
    (30, "pid_thesis_main"),         # main run at N=30
    (50, "pid_sweep_nsrc_50"),
    (100, "pid_sweep_nsrc_100"),
]


def _latest_folder(prefix: str) -> Path | None:
    candidates = sorted(EXP_ROOT.glob(f"{prefix}_*"))
    return candidates[-1] if candidates else None


def _read_setting(prefix: str) -> dict[str, dict[str, float]]:
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
    x_vals = [s[0] for s in settings]
    data = {x: _read_setting(prefix) for x, prefix in settings}

    fig, (ax_r, ax_i) = plt.subplots(1, 2, figsize=(14, 5.5))
    for s, color in zip(SCHEME_ORDER, COLORS):
        rm = [data[x][s]["mean_regret"] for x in x_vals]
        rlo = [data[x][s]["ci_lo"] for x in x_vals]
        rhi = [data[x][s]["ci_hi"] for x in x_vals]
        im = [data[x][s]["mean_infeas"] for x in x_vals]
        ilo = [data[x][s]["infeas_lo"] for x in x_vals]
        ihi = [data[x][s]["infeas_hi"] for x in x_vals]
        ax_r.plot(x_vals, rm, "o-", color=color, lw=2.0, ms=7, label=s)
        ax_r.fill_between(x_vals, rlo, rhi, color=color, alpha=0.18)
        ax_i.plot(x_vals, im, "o-", color=color, lw=2.0, ms=7, label=s)
        ax_i.fill_between(x_vals, ilo, ihi, color=color, alpha=0.18)
    if x_log:
        ax_r.set_xscale("log"); ax_i.set_xscale("log")
    ax_r.set_yscale("symlog", linthresh=0.1)
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel("Mean ITAE regret (95 % bootstrap CI)")
    ax_r.set_title("Regret (best ITAE - empirical floor) vs " + x_label)
    ax_r.legend(fontsize=9); ax_r.grid(alpha=0.3, which="both")
    ax_i.set_xlabel(x_label)
    ax_i.set_ylabel("Mean cumulative unstable (95 % bootstrap CI)")
    ax_i.set_title("Unstable count vs " + x_label)
    ax_i.legend(fontsize=9); ax_i.grid(alpha=0.3)
    fig.suptitle(f"PID thesis sweep: {x_label} "
                 "(10 plant tasks x 10-20 seeds per setting)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{fname_prefix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname_prefix}.png")

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
    print("=== TASK_SPREAD sweep ===")
    df_s = plot_sweep(SPREAD_SETTINGS, "TASK_SPREAD",
                      "experiments/aggregate_pid_spread")
    print(df_s.round(3).to_string(index=False))
    print("\n=== N_per_source sweep ===")
    df_n = plot_sweep(NSRC_SETTINGS, "N_per_source",
                      "experiments/aggregate_pid_n_per_source", x_log=True)
    print(df_n.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
