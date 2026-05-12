"""5-scheme constrained TL-BO sweep on the synthetic benchmark.

Mirrors ``tl_bayopt.py`` and ``pressure_simulation/tl_pid_bayopt.py`` but
operates on ``synthetic_benchmark.SyntheticTLBenchmark`` (closed-form
shifted-Ackley + perturbed-ellipsoid feasibility). Results land in a
timestamped experiment folder with aggregate + per-target convergence
plots and CSVs.

Usage:
    python tl_synthetic_bayopt.py [experiment_name]
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
from scipy.stats.qmc import Sobol
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from synthetic_benchmark import SyntheticTLBenchmark
from tl_bayopt import (
    EI_XI,
    expected_improvement,
    fit_gp,
    gp_predict,
    rgpe_predict,
    rgpe_weights,
    voted_feasibility,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------- config --------------------------------------
N_TASKS = 10
import os
N_PER_SOURCE = int(os.environ.get("N_PER_SOURCE", 50))
N_INIT = int(os.environ.get("N_INIT", 5))
N_ITER = int(os.environ.get("N_ITER", 25))
N_RUNS = int(os.environ.get("N_RUNS", 20))
N_CANDIDATES = 1500
INFEASIBLE_PENALTY = 50.0     # > max Ackley over [-5, 5]^2 (~13)
SEED = 42

# Meta-distribution spread: smaller -> more similar tasks ->
# stronger positive-transfer regime. Benchmark defaults (sigma_mu=1.0,
# sigma_nu=0.7 on [-5,5]^2) produce moderate spread; tightening below
# keeps source priors useful while still distinguishing target from sources.
SIGMA_MU = float(os.environ.get("SIGMA_MU", 0.5))
SIGMA_NU = float(os.environ.get("SIGMA_NU", 0.4))

SCHEMES = {
    1: "Vanilla BO",
    2: "RGPE",
    3: "cBO target-clf",
    4: "cBO RGPE-vote",
    5: "cBO equal-vote",
}
COLORS = [
    "#2E86AB", "#E63946", "#06A77D",
    "#FFB400", "#7B2CBF",            # static-source vote (4, 5)
]
HIST_PICKLE = "synthetic_convergence_histories.pkl"


# ---------------------------- source models -------------------------------
def fit_source_models(bench, source_data, seed):
    """Source GPs on feasible-only objective + SVCs on full feasibility data.

    source_data: {task_idx -> (X, y, c)}.
    """
    gps = {}
    clfs = {}
    for t, (X, y, c) in source_data.items():
        c_int = c.astype(int)
        if int(c_int.sum()) >= 5:
            gps[t] = fit_gp(X[c], y[c], seed=seed)
        if len(np.unique(c_int)) >= 2:
            sc = StandardScaler().fit(X)
            clf = SVC(kernel="rbf", gamma="scale",
                      probability=True, random_state=seed)
            clf.fit(sc.transform(X), c_int)
            clfs[t] = (clf, sc)
    return gps, clfs


# ---------------------------- BO loop -------------------------------------
def _initial_points(bench, seed):
    """Sobol init shared across schemes for paired comparison."""
    sobol = Sobol(d=2, seed=seed)
    unit = sobol.random(n=N_INIT)
    lo, hi = bench.domain()
    return lo + unit * (hi - lo)


def bo_loop(scheme, seed, bench, target_idx, source_gps, source_clfs,
            beta=1.0):
    """One BO run. Returns (best-feas-y history, cumulative-infeas history)."""
    rng = np.random.default_rng(seed)
    lo, hi = bench.domain()

    X = _initial_points(bench, seed)
    y_arr, c_arr = bench.query(target_idx, X)
    y = np.asarray(y_arr, dtype=float)
    c = np.asarray(c_arr, dtype=bool)

    def best_feas():
        return float(y[c].min()) if c.any() else float("inf")

    history = [best_feas()]
    infeas_history = [int((~c).sum())]

    for _ in range(N_ITER):
        X_cand = rng.uniform(lo, hi, size=(N_CANDIDATES, 2))

        if scheme == 1:  # Vanilla BO (no source, infeasibles penalised)
            y_train = np.where(c, y, INFEASIBLE_PENALTY)
            target_gp = fit_gp(X, y_train, seed=seed)
            mean, std = gp_predict(target_gp, X_cand)
            ei = expected_improvement(mean, std, y_train.min(), xi=EI_XI)
            x_next = X_cand[np.argmax(ei)]

        elif scheme == 2:  # RGPE (no constraint, infeasibles penalised)
            y_train = np.where(c, y, INFEASIBLE_PENALTY)
            target_gp = fit_gp(X, y_train, seed=seed)
            keys, src_w, tgt_w = rgpe_weights(source_gps, X, y_train, rng)
            mean, std = rgpe_predict(source_gps, target_gp, keys, src_w,
                                     tgt_w, X_cand)
            ei = expected_improvement(mean, std, y_train.min(), xi=EI_XI)
            x_next = X_cand[np.argmax(ei)]

        elif scheme in (8, 9):
            # Constrained with penalty in surrogate: target GP fits ALL data
            # (feasible + penalty). Source vote provides the constraint, no
            # target-classifier multiplier. The penalty bumps in the GP
            # supply the self-correcting feedback that the static source
            # vote alone lacks.
            y_train = np.where(c, y, INFEASIBLE_PENALTY)
            target_gp = fit_gp(X, y_train, seed=seed)
            keys, src_w, tgt_w = rgpe_weights(source_gps, X, y_train, rng)
            mean, std = rgpe_predict(
                source_gps, target_gp, keys, src_w, tgt_w, X_cand
            )
            ei = expected_improvement(
                mean, std, y_train.min(), xi=EI_XI
            )
            if scheme == 8:
                p_feas = voted_feasibility(
                    source_clfs, keys, src_w, X_cand
                )
            else:  # scheme == 9
                eq_w = (np.full(len(keys), 1.0 / len(keys))
                        if len(keys) > 0 else np.array([]))
                p_feas = voted_feasibility(
                    source_clfs, keys, eq_w, X_cand
                )
            if beta == 0.0:
                acq = ei
            else:
                acq = ei * np.power(np.clip(p_feas, 0.0, 1.0), beta)
            x_next = X_cand[np.argmax(acq)]

        elif scheme in (3, 4, 5, 6, 7):
            # Surrogate on feasible-only target; constraint multiplier
            if int(c.sum()) < 2:
                x_next = X_cand[rng.integers(len(X_cand))]
            else:
                X_feas = X[c]
                y_feas = y[c]
                target_gp = fit_gp(X_feas, y_feas, seed=seed)
                keys, src_w, tgt_w = rgpe_weights(
                    source_gps, X_feas, y_feas, rng
                )
                mean, std = rgpe_predict(
                    source_gps, target_gp, keys, src_w, tgt_w, X_cand
                )
                ei = expected_improvement(
                    mean, std, y_feas.min(), xi=EI_XI
                )

                # --- target-classifier probability (used in 3, 6, 7) ---
                def _p_target():
                    c_int = c.astype(int)
                    if len(np.unique(c_int)) < 2:
                        return np.ones(len(X_cand))
                    sc = StandardScaler().fit(X)
                    clf = SVC(kernel="rbf", gamma="scale",
                              probability=True, random_state=seed)
                    clf.fit(sc.transform(X), c_int)
                    return clf.predict_proba(sc.transform(X_cand))[:, 1]

                # --- source-vote probabilities (used in 4, 5, 6, 7) ---
                def _p_source_rgpe():
                    return voted_feasibility(
                        source_clfs, keys, src_w, X_cand
                    )

                def _p_source_equal():
                    eq_w = (np.full(len(keys), 1.0 / len(keys))
                            if len(keys) > 0 else np.array([]))
                    return voted_feasibility(
                        source_clfs, keys, eq_w, X_cand
                    )

                if scheme == 3:
                    p_feas = _p_target()
                elif scheme == 4:
                    p_feas = _p_source_rgpe()
                elif scheme == 5:
                    p_feas = _p_source_equal()
                elif scheme == 6:
                    # hybrid: source vote (RGPE-weighted) gated by target clf
                    p_feas = _p_source_rgpe() * _p_target()
                else:  # scheme == 7
                    # hybrid: source vote (equal-weighted) gated by target clf
                    p_feas = _p_source_equal() * _p_target()

                if beta == 0.0:
                    acq = ei
                else:
                    acq = ei * np.power(np.clip(p_feas, 0.0, 1.0), beta)
                x_next = X_cand[np.argmax(acq)]
        else:
            raise ValueError(f"unknown scheme {scheme}")

        y_next_arr, c_next_arr = bench.query(
            target_idx, x_next.reshape(1, -1)
        )
        X = np.vstack([X, x_next])
        y = np.append(y, float(np.asarray(y_next_arr).flatten()[0]))
        c = np.append(c, bool(np.asarray(c_next_arr).flatten()[0]))
        history.append(best_feas())
        infeas_history.append(int((~c).sum()))

    return np.array(history), np.array(infeas_history), X, c


# ---------------------------- per-target driver ---------------------------
def run_for_target(target_idx, bench, source_data):
    """Run all 5 schemes x N_RUNS for target_idx; return per-scheme histories."""
    out = {s: {"hist": [], "infeas": [], "X": [], "c": []} for s in SCHEMES}
    for run in range(N_RUNS):
        seed = SEED + 100 + run
        # Source models depend on the seed only via SVC's random_state;
        # we fit them once per seed for paired comparison across schemes.
        source_gps, source_clfs = fit_source_models(bench, source_data, seed)
        for s in SCHEMES:
            hist, infeas, X_all, c_all = bo_loop(
                s, seed, bench, target_idx, source_gps, source_clfs
            )
            out[s]["hist"].append(hist)
            out[s]["infeas"].append(infeas)
            out[s]["X"].append(X_all)
            out[s]["c"].append(c_all)
    return out


# ---------------------------- experiment folder ---------------------------
def make_experiment_dir(name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("experiments") / f"{name}_{ts}"
    (root / "per_target").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------- main ----------------------------------------
def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "synthetic_5method"
    exp_dir = make_experiment_dir(exp_name)
    print(f"Experiment folder: {exp_dir}")

    bench = SyntheticTLBenchmark(
        n_tasks=N_TASKS, sigma_mu=SIGMA_MU, sigma_nu=SIGMA_NU, seed=SEED
    )
    print(f"Generated {N_TASKS} synthetic tasks "
          f"(seed={SEED}, sigma_mu={SIGMA_MU}, sigma_nu={SIGMA_NU})")
    for t in range(N_TASKS):
        p = bench.tasks[t]
        f_star, _ = bench.true_constrained_minimum(t, n_grid=251)
        print(f"  Task {t}: mu=({p.mu[0]:+.2f},{p.mu[1]:+.2f}), "
              f"nu=({p.nu[0]:+.2f},{p.nu[1]:+.2f}), "
              f"a={p.a:.2f} b={p.b:.2f} gamma={p.gamma:.2f}, "
              f"f*={f_star:.3f}")

    # --- pre-sample source data for every task once (re-used per target) --
    print(f"\nPre-sampling {N_PER_SOURCE} LHS source samples per task...")
    rng = np.random.default_rng(SEED + 1)
    source_data_all = {}
    for t in range(N_TASKS):
        X = bench.sample_inputs(N_PER_SOURCE, rng=rng, method="sobol")
        y, c = bench.query(t, X)
        source_data_all[t] = (X, np.asarray(y), np.asarray(c))
    feas_rates = [src[2].mean() for src in source_data_all.values()]
    print(f"  Feasibility rate per task (min/median/max): "
          f"{min(feas_rates):.2f} / {np.median(feas_rates):.2f} / "
          f"{max(feas_rates):.2f}")

    # --- sweep over targets ----------------------------------------------
    print(f"\nSweep: {N_TASKS} targets x {len(SCHEMES)} schemes x "
          f"{N_RUNS} runs x {N_ITER} iters")
    sweep = {}
    t_start = time.time()
    for ti in range(N_TASKS):
        t0 = time.time()
        # source data = everything except target ti
        source_data = {k: v for k, v in source_data_all.items() if k != ti}
        sweep[ti] = run_for_target(ti, bench, source_data)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (N_TASKS - ti - 1)
        finals = {s: float(np.mean(np.array(sweep[ti][s]["hist"])[:, -1]))
                  for s in SCHEMES}
        infs = {s: float(np.mean(np.array(sweep[ti][s]["infeas"])[:, -1]))
                for s in SCHEMES}
        print(f"[{ti + 1:>2}/{N_TASKS}] target {ti}  {elapsed:5.0f}s  "
              f"total {total / 60:5.1f}m  eta {eta / 60:5.1f}m", flush=True)
        for s in SCHEMES:
            print(f"        {SCHEMES[s]:<22} y={finals[s]:8.4f}  "
                  f"infeas={infs[s]:5.2f}", flush=True)

    # --- cache ------------------------------------------------------------
    with open(HIST_PICKLE, "wb") as f:
        pickle.dump({"sweep": sweep, "tasks": bench.tasks,
                     "n_tasks": N_TASKS, "n_runs": N_RUNS}, f)
    shutil.copy(HIST_PICKLE, exp_dir / "histories.pkl")
    print(f"\nSaved {HIST_PICKLE}")

    # --- tables -----------------------------------------------------------
    target_idx_label = [f"task{ti:02d}" for ti in range(N_TASKS)]
    burr_cols = {SCHEMES[s]: [
        float(np.array(sweep[ti][s]["hist"])[:, -1].mean())
        for ti in range(N_TASKS)
    ] for s in SCHEMES}
    inf_cols = {SCHEMES[s]: [
        float(np.array(sweep[ti][s]["infeas"])[:, -1].mean())
        for ti in range(N_TASKS)
    ] for s in SCHEMES}
    burr_df = pd.DataFrame(burr_cols, index=target_idx_label)
    inf_df = pd.DataFrame(inf_cols, index=target_idx_label)
    burr_df.to_csv(exp_dir / "final_obj.csv")
    inf_df.to_csv(exp_dir / "final_infeas.csv")

    # also dump regret table (final - true min)
    truths = []
    for ti in range(N_TASKS):
        f_star, _ = bench.true_constrained_minimum(ti, n_grid=401)
        truths.append(f_star)
    truths = np.array(truths)
    regret_df = burr_df.copy()
    for col in regret_df.columns:
        regret_df[col] = regret_df[col].values - truths
    regret_df.to_csv(exp_dir / "final_regret.csv")

    print("\n=== Final best feasible objective per target (mean over runs) ===")
    print(burr_df.round(3).to_string())
    print("\n=== Final regret = best - true_min per target ===")
    print(regret_df.round(3).to_string())
    print("\n=== Cumulative infeasible evals per target (mean over runs) ===")
    print(inf_df.round(2).to_string())

    # win counts
    def share_wins(df):
        v = df.values
        is_min = v == v.min(axis=1, keepdims=True)
        return (is_min / is_min.sum(axis=1, keepdims=True)).sum(axis=0)

    obj_wins = share_wins(burr_df)
    inf_wins = share_wins(inf_df)
    print("\n=== Win counts (tied wins shared, out of 10 targets) ===")
    print(f"  {'scheme':<22} {'obj wins':>9} {'infeas wins':>13}")
    for i, s in enumerate(SCHEMES):
        print(f"  {SCHEMES[s]:<22} {obj_wins[i]:>9.2f} {inf_wins[i]:>13.2f}")

    # --- statistical analysis --------------------------------------------
    print("\n=== Statistical analysis (paired across targets x seeds) ===")
    stat_report = statistical_analysis(sweep, truths, exp_dir)
    print(stat_report)

    # --- aggregate plot ---------------------------------------------------
    plot_aggregate(bench, sweep, truths, exp_dir / "convergence_aggregate.png")
    print(f"Saved {exp_dir / 'convergence_aggregate.png'}")

    # --- per-target plots -------------------------------------------------
    print(f"Writing per-target plots into {exp_dir / 'per_target'}/...")
    for ti in range(N_TASKS):
        f_star = truths[ti]
        out = (exp_dir / "per_target" / f"target_{ti:02d}.png")
        plot_per_target(ti, bench, sweep[ti], f_star, out)
    print(f"  Wrote {N_TASKS} per-target plots.")

    # --- per-method query overlays ---------------------------------------
    queries_dir = exp_dir / "queries"
    queries_dir.mkdir(exist_ok=True)
    print(f"Writing per-method query overlays into {queries_dir}/...")
    for s, name in SCHEMES.items():
        out = queries_dir / f"scheme_{s}_{name.replace(' ', '_')}.png"
        plot_method_queries(s, name, bench, sweep, out)
    print(f"  Wrote {len(SCHEMES)} per-method query overlays.")

    print(f"\nAll outputs in: {exp_dir}")


def _bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, rng_seed=0):
    """Percentile-bootstrap (1-alpha) CI on the mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng_b = np.random.default_rng(rng_seed)
    idx = rng_b.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def statistical_analysis(sweep, truths, exp_dir):
    """Pooled and per-target stats on final regret and infeasibility.

    Reports:
    - Mean regret + 95 % bootstrap CI per scheme.
    - Pairwise paired Wilcoxon (Holm-Bonferroni corrected).
    - Per-target win-rate over seeds (more robust to magnitude scaling).
    """
    from scipy.stats import wilcoxon

    n_schemes = len(SCHEMES)
    scheme_keys = list(SCHEMES.keys())
    scheme_names = [SCHEMES[s] for s in scheme_keys]

    # Build full (target, seed) x scheme matrices for final regret and infeas
    final_regret = np.zeros((N_TASKS, N_RUNS, n_schemes), dtype=float)
    final_infeas = np.zeros((N_TASKS, N_RUNS, n_schemes), dtype=float)
    for ti in range(N_TASKS):
        for si, s in enumerate(scheme_keys):
            H = np.array(sweep[ti][s]["hist"])
            H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
            I = np.array(sweep[ti][s]["infeas"])
            final_regret[ti, :, si] = H[:, -1] - truths[ti]
            final_infeas[ti, :, si] = I[:, -1]

    pooled_regret = final_regret.reshape(-1, n_schemes)   # (N_TASKS*N_RUNS, S)
    pooled_infeas = final_infeas.reshape(-1, n_schemes)
    n_paired = pooled_regret.shape[0]

    # --- 1) Mean regret + 95 % bootstrap CI per scheme ------------------
    lines = []
    lines.append("\n-- Mean final regret across all targets x seeds "
                 f"(N={n_paired} paired observations) --")
    lines.append(f"  {'scheme':<22} {'mean regret':>11}  "
                 f"{'95% bootstrap CI':>22}  {'mean miscuts':>13}  "
                 f"{'95% CI':>16}")
    regret_summary = []
    infeas_summary = []
    for si, name in enumerate(scheme_names):
        m_r, lo_r, hi_r = _bootstrap_mean_ci(pooled_regret[:, si])
        m_i, lo_i, hi_i = _bootstrap_mean_ci(pooled_infeas[:, si])
        regret_summary.append((name, m_r, lo_r, hi_r))
        infeas_summary.append((name, m_i, lo_i, hi_i))
        lines.append(f"  {name:<22} {m_r:>11.3f}  "
                     f"[{lo_r:>8.3f}, {hi_r:>8.3f}]  "
                     f"{m_i:>13.2f}  [{lo_i:>5.2f},{hi_i:>5.2f}]")

    # --- 2) Pairwise paired Wilcoxon on pooled regret --------------------
    lines.append("\n-- Pairwise paired Wilcoxon signed-rank tests on regret "
                 "(Holm-Bonferroni corrected p-values) --")
    pairs = []
    raw_p = []
    diffs = []
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
            pairs.append((i, j))
            raw_p.append(p)
            diffs.append(d)

    # Holm-Bonferroni correction
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
                     f"{mean_d:>10.3f} {med_d:>11.3f} {p_adj:>10.2e} {sig:>4}")
        pair_rows.append({
            "scheme_A": scheme_names[i],
            "scheme_B": scheme_names[j],
            "mean_diff": mean_d,
            "median_diff": med_d,
            "p_raw": raw_p[pairs.index((i, j))],
            "p_holm": p_adj,
            "sig": sig,
        })

    # --- 3) Per-target win rates over seeds ------------------------------
    lines.append("\n-- Per-target win rates over seeds "
                 "(fraction of seeds where scheme is strictly best on regret) --")
    win_table = np.zeros((N_TASKS, n_schemes))
    for ti in range(N_TASKS):
        for run in range(N_RUNS):
            row = final_regret[ti, run]
            best = np.where(row == row.min())[0]
            for b in best:
                win_table[ti, b] += 1.0 / len(best)
    win_rate = win_table / N_RUNS
    lines.append(f"  {'target':<10} " + "  ".join(f"{n[:18]:>18}"
                                                  for n in scheme_names))
    for ti in range(N_TASKS):
        lines.append(f"  task{ti:02d}     " +
                     "  ".join(f"{win_rate[ti, si]:>18.3f}"
                               for si in range(n_schemes)))
    lines.append(f"  {'mean':<10} " +
                 "  ".join(f"{win_rate[:, si].mean():>18.3f}"
                          for si in range(n_schemes)))

    # --- save the analysis to disk --------------------------------------
    pd.DataFrame(regret_summary,
                 columns=["scheme", "mean_regret", "ci_lo", "ci_hi"]
                 ).to_csv(exp_dir / "regret_summary.csv", index=False)
    pd.DataFrame(infeas_summary,
                 columns=["scheme", "mean_infeas", "ci_lo", "ci_hi"]
                 ).to_csv(exp_dir / "infeas_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(exp_dir / "pairwise_wilcoxon.csv", index=False)
    pd.DataFrame(win_rate, index=[f"task{i:02d}" for i in range(N_TASKS)],
                 columns=scheme_names).to_csv(exp_dir / "win_rates.csv")

    return "\n".join(lines)


def plot_per_target(ti, bench, runs_by_scheme, f_star, out_path):
    n_iters = N_ITER + 1
    iters = np.arange(n_iters)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_b, ax_i, ax_t = axes

    for s, name in SCHEMES.items():
        H = np.array(runs_by_scheme[s]["hist"])
        I = np.array(runs_by_scheme[s]["infeas"])
        # clip inf for plotting
        H_clip = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
        mean_b = H_clip.mean(axis=0)
        se_b = H_clip.std(axis=0) / np.sqrt(max(H.shape[0], 1))
        mean_i = I.mean(axis=0)
        se_i = I.std(axis=0) / np.sqrt(max(I.shape[0], 1))
        ax_b.plot(iters, mean_b, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_b.fill_between(iters, mean_b - se_b, mean_b + se_b,
                          color=COLORS[s - 1], alpha=0.18)
        ax_i.plot(iters, mean_i, color=COLORS[s - 1], linewidth=2.2, label=name)
        ax_i.fill_between(iters, mean_i - se_i, mean_i + se_i,
                          color=COLORS[s - 1], alpha=0.18)

    ax_b.axhline(f_star, color="black", linestyle="--", alpha=0.5,
                 label=f"true min = {f_star:.3f}")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("BO iteration"); ax_b.set_ylabel("Best feasible f")
    ax_b.set_title(f"Objective convergence (target task {ti})")
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.grid(alpha=0.3, which="both")

    ax_i.set_xlabel("BO iteration")
    ax_i.set_ylabel("Cumulative infeasible evals")
    ax_i.set_title("Infeasibility convergence")
    ax_i.legend(loc="upper left", fontsize=9)
    ax_i.grid(alpha=0.3)

    bench.plot_task(ti, ax=ax_t, n_grid=120, log_objective=True)
    ax_t.set_title(f"Task {ti} layout")

    p = bench.tasks[ti]
    fig.suptitle(
        f"Synthetic target task {ti}  |  "
        f"mu=({p.mu[0]:+.2f},{p.mu[1]:+.2f})  "
        f"nu=({p.nu[0]:+.2f},{p.nu[1]:+.2f})  "
        f"true min={f_star:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_method_queries(scheme, scheme_name, bench, sweep, out_path):
    """For one method: a 2x5 grid showing every target's task landscape with
    that method's queried points scattered over all runs. Feasible points are
    drawn as green dots, infeasible points as red crosses."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        f"Method: {scheme_name}  — query points across {N_RUNS} runs\n"
        f"green dot = feasible eval, red x = infeasible eval, "
        f"white star = constrained min",
        fontsize=12,
    )
    for ti, ax in enumerate(axes.ravel()):
        bench.plot_task(ti, ax=ax, n_grid=120, log_objective=False,
                        show_optima=False)

        # Stack queries from every run, distinguish initial Sobol vs BO picks
        all_X = np.vstack(sweep[ti][scheme]["X"])
        all_c = np.concatenate(sweep[ti][scheme]["c"]).astype(bool)
        feas = all_c
        infeas = ~all_c

        ax.scatter(
            all_X[feas, 0], all_X[feas, 1],
            s=22, c="#1aff66", edgecolors="#0d6b2a", linewidths=0.6,
            zorder=5, label=f"feasible ({int(feas.sum())})",
        )
        ax.scatter(
            all_X[infeas, 0], all_X[infeas, 1],
            s=42, c="red", marker="x", linewidths=1.4,
            zorder=6, label=f"infeasible ({int(infeas.sum())})",
        )

        # Mark the constrained optimum on top so it's visible.
        f_star_xy = bench.true_constrained_minimum(ti, n_grid=251)
        if f_star_xy is not None:
            _, x_star = f_star_xy
            ax.plot(x_star[0], x_star[1], marker="*", markersize=16,
                    markerfacecolor="white", markeredgecolor="black",
                    zorder=7)

        ax.set_title(f"task {ti}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_xlabel(""); ax.set_ylabel("")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(bench, sweep, truths, out_path):
    n_iters = N_ITER + 1
    iters = np.arange(n_iters)

    # per-target normalize convergence by per-target median initial best
    obj_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    obj_se = {s: np.zeros(n_iters) for s in SCHEMES}
    inf_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    inf_se = {s: np.zeros(n_iters) for s in SCHEMES}
    target_means_obj = {s: [] for s in SCHEMES}
    target_means_inf = {s: [] for s in SCHEMES}

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
            H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
            target_means_obj[s].append((H / baseline).mean(axis=0))
            I = np.array(sweep[ti][s]["infeas"])
            target_means_inf[s].append(I.mean(axis=0))

    for s in SCHEMES:
        TM_o = np.array(target_means_obj[s])
        TM_i = np.array(target_means_inf[s])
        obj_curves[s] = TM_o.mean(axis=0)
        obj_se[s] = TM_o.std(axis=0) / np.sqrt(N_TASKS)
        inf_curves[s] = TM_i.mean(axis=0)
        inf_se[s] = TM_i.std(axis=0) / np.sqrt(N_TASKS)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_n, ax_log, ax_in, ax_lay = axes.ravel()

    for s, name in SCHEMES.items():
        for ax in (ax_n, ax_log):
            ax.plot(iters, obj_curves[s], color=COLORS[s - 1],
                    linewidth=2.2, label=name)
            ax.fill_between(iters, obj_curves[s] - obj_se[s],
                            obj_curves[s] + obj_se[s],
                            color=COLORS[s - 1], alpha=0.18)
        ax_in.plot(iters, inf_curves[s], color=COLORS[s - 1],
                   linewidth=2.2, label=name)
        ax_in.fill_between(iters, inf_curves[s] - inf_se[s],
                           inf_curves[s] + inf_se[s],
                           color=COLORS[s - 1], alpha=0.18)

    ax_n.set_xlabel("BO iteration")
    ax_n.set_ylabel("Best f / median initial f")
    ax_n.set_title(f"Aggregated normalized convergence ({N_TASKS} targets)")
    ax_n.legend(loc="upper right", fontsize=9)
    ax_n.grid(alpha=0.3)

    ax_log.set_xlabel("BO iteration")
    ax_log.set_ylabel("Best f / median initial f (log)")
    ax_log.set_yscale("log")
    ax_log.set_title("Same, log-y axis")
    ax_log.legend(loc="upper right", fontsize=9)
    ax_log.grid(alpha=0.3, which="both")

    ax_in.set_xlabel("BO iteration")
    ax_in.set_ylabel("Cumulative infeasible evals (mean across targets)")
    ax_in.set_title("Aggregated infeasibility convergence")
    ax_in.legend(loc="upper left", fontsize=9)
    ax_in.grid(alpha=0.3)

    # task layout
    for t in range(N_TASKS):
        p = bench.tasks[t]
        ax_lay.scatter(p.mu[0], p.mu[1], color="#666", s=80,
                       edgecolors="black", linewidths=0.7,
                       marker="o")
        ax_lay.annotate(str(t), (p.mu[0], p.mu[1]),
                        textcoords="offset points",
                        xytext=(7, 7), fontsize=10)
    ax_lay.set_xlim(bench.DOMAIN_LO[0], bench.DOMAIN_HI[0])
    ax_lay.set_ylim(bench.DOMAIN_LO[1], bench.DOMAIN_HI[1])
    ax_lay.set_xlabel("x1 (Ackley shift mu_1)")
    ax_lay.set_ylabel("x2 (Ackley shift mu_2)")
    ax_lay.set_title("Task shifts mu_t in input space")
    ax_lay.grid(alpha=0.3)

    fig.suptitle(
        f"Synthetic constrained TL-BO sweep, {N_TASKS} targets, "
        f"{N_RUNS} runs x {N_INIT} init + {N_ITER} BO iters",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
