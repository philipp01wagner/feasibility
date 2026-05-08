"""Bayesian-optimization comparison across ALL target tasks.

Goal: minimize burr. For each task that has both feasible and infeasible
measurements, treat that task as the target and run five BO schemes:

  1. Vanilla BO      - single GP, no source data. Miscut -> high penalty.
  2. RGPE            - Ranking-Weighted GP Ensemble + target GP. Same penalty.
  3. RGPE + cBO      - RGPE surrogate (feasible-only target GP), constraint
                       = SVC trained on the target's collected labels.
  4. RGPE + cBO      - like 3, but constraint = weighted vote of per-source
     (RGPE-vote)       feasibility classifiers using the RGPE source weights.
  5. RGPE + cBO      - like 4, but vote weights are equal across sources.
     (equal-vote)

Outputs:
  - bayopt_all_targets_burr.csv     final best burr per (target, scheme)
  - bayopt_all_targets_miscuts.csv  cumulative miscuts per (target, scheme)
  - bayopt_all_targets.png          aggregated convergence + win counts
"""

import pickle
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cutting_simulator import create_task_simulator

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="ABNORMAL_TERMINATION_IN_LNSRCH")

# ---------------------------- config --------------------------------------
with open("./data/task_df_dict.pkl", "rb") as f:
    task_df_dict = pickle.load(f)

FEATURE_COLS = ["feedrate", "gas_pressure", "focal_position"]
DIM = 3

SEED = 42
N_INIT = 5
N_ITER = 30
N_RUNS = 3
N_CANDIDATES = 1500
INFEASIBLE_PENALTY = 5000.0
RGPE_SAMPLES = 30
EI_XI = 0.01

SCHEMES = {
    1: "Vanilla BO",
    2: "RGPE",
    3: "cBO target-clf",
    4: "cBO RGPE-vote",
    5: "cBO equal-vote",
}
COLORS = ["#2E86AB", "#E63946", "#06A77D", "#FFB400", "#7B2CBF"]

# Targets where the source classifiers systematically mispredict the target's
# feasibility boundary, blowing up scheme 4/5 miscut counts. Excluded from
# the cross-target sweep so the aggregate isn't dominated by these tails.
OUTLIER_TARGETS = {
    "100_ST100MD0-N2H0-30-2_L81_0.7_10000_S235JR",
    "80_ST080MD0-N2H0-30-2_L95_0.4_6000_S355J2",
    "150_ST150MD0-N2H0-30-2_L76_0.4_10000_S355J2sand",
}


# ---------------------------- helpers -------------------------------------
def make_kernel():
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )


def fit_gp(X, y, seed=SEED):
    sc = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(
        kernel=make_kernel(),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )
    gp.fit(sc.transform(X), y)
    return gp, sc


def gp_predict(pair, X):
    gp, sc = pair
    return gp.predict(sc.transform(X), return_std=True)


def expected_improvement(mean, std, y_best, xi=EI_XI):
    imp = y_best - mean - xi
    Z = imp / np.where(std > 0, std, 1e-9)
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return np.where(std > 0, np.maximum(ei, 0.0), 0.0)


# ---------------------------- per-task models (built once) ----------------
def fit_all_task_models():
    """Fit a GP and SVC for every task. We later filter out the target."""
    gps, clfs = {}, {}
    for name, df in task_df_dict.items():
        feas = (df["burr_evaluated"].values >= 0) & (
            df["roughness_z_evaluated"].values >= 0
        )
        if feas.sum() >= 5:
            gps[name] = fit_gp(
                df.loc[feas, FEATURE_COLS].values,
                df.loc[feas, "burr_evaluated"].values,
            )
        if len(np.unique(feas.astype(int))) == 2:
            X = df[FEATURE_COLS].values
            sc = StandardScaler().fit(X)
            clf = SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
            clf.fit(sc.transform(X), feas.astype(int))
            clfs[name] = (clf, sc)
    return gps, clfs


# ---------------------------- RGPE ----------------------------------------
def _pairwise_rank_loss(samples, y_obs):
    actual = np.sign(y_obs[:, None] - y_obs[None, :])
    n = len(y_obs)
    n_pairs = max(1, n * (n - 1) // 2)
    total = 0.0
    for s in samples:
        pred = np.sign(s[:, None] - s[None, :])
        total += ((actual * pred) < 0).sum() / 2
    return total / (len(samples) * n_pairs)


def rgpe_weights(source_gps, X_obs, y_obs, rng):
    keys = list(source_gps.keys())
    n_obs = len(X_obs)
    if len(keys) == 0:
        return [], np.array([]), 0.0
    if n_obs < 2:
        return keys, np.full(len(keys), 1.0 / len(keys)), 0.0
    losses = []
    for k in keys:
        m, s = gp_predict(source_gps[k], X_obs)
        smp = rng.normal(m, s + 1e-6, size=(RGPE_SAMPLES, n_obs))
        losses.append(_pairwise_rank_loss(smp, y_obs))
    losses = np.array(losses)
    scale = max(np.median(losses), 1e-3)
    src_w = np.exp(-losses / scale)
    src_w /= src_w.sum()
    target_w = float(np.clip(1.0 / (1.0 + np.exp(-(n_obs - 5) / 5.0)), 0.0, 0.5))
    src_w *= 1.0 - target_w
    return keys, src_w, target_w


def rgpe_predict(source_gps, target_gp, keys, src_w, target_w, X):
    means, vars_, ws = [], [], []
    for k, w in zip(keys, src_w):
        m, s = gp_predict(source_gps[k], X)
        means.append(m); vars_.append(s ** 2); ws.append(w)
    if target_gp is not None and target_w > 0:
        m, s = gp_predict(target_gp, X)
        means.append(m); vars_.append(s ** 2); ws.append(target_w)
    if not means:
        return np.zeros(X.shape[0]), np.ones(X.shape[0])
    means = np.array(means); vars_ = np.array(vars_); ws = np.array(ws)
    ws = ws / max(ws.sum(), 1e-12)
    mean = (ws[:, None] * means).sum(axis=0)
    var = (ws[:, None] * (vars_ + (means - mean) ** 2)).sum(axis=0)
    return mean, np.sqrt(np.maximum(var, 1e-12))


# ---------------------------- voted feasibility ---------------------------
def voted_feasibility(source_clfs, keys, weights, X):
    use_keys = [k for k in keys if k in source_clfs]
    use_w = np.array([w for k, w in zip(keys, weights) if k in source_clfs])
    if use_w.sum() <= 0 or len(use_keys) == 0:
        return np.full(X.shape[0], 1.0)
    use_w = use_w / use_w.sum()
    p = np.array([
        source_clfs[k][0].predict_proba(source_clfs[k][1].transform(X))[:, 1]
        for k in use_keys
    ])
    return use_w @ p


# ---------------------------- BO loop -------------------------------------
def bo_loop(scheme, seed, simulator, bounds, source_gps, source_clfs, beta=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_INIT, DIM))
    y = np.atleast_1d(simulator(X[:, 0], X[:, 1], X[:, 2]))
    history, miscut_history = [], []

    def best_feas():
        f = y >= 0
        return float(y[f].min()) if f.any() else float("inf")

    history.append(best_feas())
    miscut_history.append(int((y < 0).sum()))

    for _ in range(N_ITER):
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_CANDIDATES, DIM))

        if scheme == 1:
            y_train = np.where(y >= 0, y, INFEASIBLE_PENALTY)
            target_gp = fit_gp(X, y_train, seed=seed)
            mean, std = gp_predict(target_gp, X_cand)
            ei = expected_improvement(mean, std, y_train.min())
            x_next = X_cand[np.argmax(ei)]

        elif scheme == 2:
            y_train = np.where(y >= 0, y, INFEASIBLE_PENALTY)
            target_gp = fit_gp(X, y_train, seed=seed)
            keys, src_w, tgt_w = rgpe_weights(source_gps, X, y_train, rng)
            mean, std = rgpe_predict(source_gps, target_gp, keys, src_w, tgt_w, X_cand)
            ei = expected_improvement(mean, std, y_train.min())
            x_next = X_cand[np.argmax(ei)]

        elif scheme in (3, 4, 5):
            feas_mask = y >= 0
            if feas_mask.sum() < 2:
                x_next = X_cand[rng.integers(len(X_cand))]
            else:
                X_feas = X[feas_mask]
                y_feas = y[feas_mask]
                target_gp = fit_gp(X_feas, y_feas, seed=seed)
                keys, src_w, tgt_w = rgpe_weights(source_gps, X_feas, y_feas, rng)
                mean, std = rgpe_predict(
                    source_gps, target_gp, keys, src_w, tgt_w, X_cand
                )
                ei = expected_improvement(mean, std, y_feas.min())

                if scheme == 3:
                    fi = feas_mask.astype(int)
                    if len(np.unique(fi)) < 2:
                        p_feas = np.ones(len(X_cand))
                    else:
                        sc = StandardScaler().fit(X)
                        clf = SVC(
                            kernel="rbf",
                            gamma="scale",
                            probability=True,
                            random_state=seed,
                        )
                        clf.fit(sc.transform(X), fi)
                        p_feas = clf.predict_proba(sc.transform(X_cand))[:, 1]
                elif scheme == 4:
                    p_feas = voted_feasibility(source_clfs, keys, src_w, X_cand)
                else:  # scheme == 5
                    eq_w = (
                        np.full(len(keys), 1.0 / len(keys))
                        if len(keys) > 0 else np.array([])
                    )
                    p_feas = voted_feasibility(source_clfs, keys, eq_w, X_cand)

                # beta tunes constraint sharpness:
                # beta=0 -> no constraint (p_feas**0 = 1 everywhere)
                # beta=1 -> standard Gardner-style constrained EI
                # beta -> infinity -> hard mask (only p_feas=1 survives)
                if beta == 0.0:
                    acq = ei
                else:
                    acq = ei * np.power(np.clip(p_feas, 0.0, 1.0), beta)
                x_next = X_cand[np.argmax(acq)]
        else:
            raise ValueError(f"unknown scheme {scheme}")

        y_next = float(np.atleast_1d(
            simulator(np.atleast_1d(x_next[0]), np.atleast_1d(x_next[1]),
                      np.atleast_1d(x_next[2]))
        )[0])
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        history.append(best_feas())
        miscut_history.append(int((y < 0).sum()))

    return np.array(history), np.array(miscut_history)


# ---------------------------- per-target driver ---------------------------
def run_for_target(target_name, all_gps, all_clfs):
    df = task_df_dict[target_name]
    lo = df[FEATURE_COLS].min().values
    hi = df[FEATURE_COLS].max().values
    bounds = np.column_stack([lo, hi])
    sim = create_task_simulator(target_name, task_df_dict)
    src_gps = {k: v for k, v in all_gps.items() if k != target_name}
    src_clfs = {k: v for k, v in all_clfs.items() if k != target_name}

    out = {s: {"hist": [], "miscuts": []} for s in SCHEMES}
    for run in range(N_RUNS):
        seed = SEED + run
        for s in SCHEMES:
            h, mh = bo_loop(s, seed, sim, bounds, src_gps, src_clfs)
            out[s]["hist"].append(h)
            out[s]["miscuts"].append(mh)
    return out


# ---------------------------- main ----------------------------------------
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
    print(f"  {len(all_gps)} GPs, {len(all_clfs)} classifiers ({time.time()-t0:.1f}s)")

    print(f"\nSweep: {len(targets)} targets x {len(SCHEMES)} schemes "
          f"x {N_RUNS} runs x {N_ITER} iterations")

    all_results = {}
    t_start = time.time()
    for ti, target in enumerate(targets):
        t0 = time.time()
        all_results[target] = run_for_target(target, all_gps, all_clfs)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (len(targets) - ti - 1)
        print(f"[{ti+1:>2}/{len(targets)}] {target[:50]:<50} "
              f"{elapsed:5.0f}s  total {total/60:5.1f}m  eta {eta/60:5.1f}m")

    # ---- summary tables ----
    targets_short = [t[:48] for t in targets]
    burr_table = {SCHEMES[s]: [] for s in SCHEMES}
    miscut_table = {SCHEMES[s]: [] for s in SCHEMES}

    for target in targets:
        for s in SCHEMES:
            H = np.array(all_results[target][s]["hist"])
            M = np.array(all_results[target][s]["miscuts"])
            burr_table[SCHEMES[s]].append(float(H[:, -1].mean()))
            miscut_table[SCHEMES[s]].append(float(M[:, -1].mean()))

    burr_df = pd.DataFrame(burr_table, index=targets_short)
    miscut_df = pd.DataFrame(miscut_table, index=targets_short)
    burr_df.to_csv("bayopt_all_targets_burr.csv")
    miscut_df.to_csv("bayopt_all_targets_miscuts.csv")
    print("\nSaved bayopt_all_targets_burr.csv, bayopt_all_targets_miscuts.csv")

    print("\n=== Final best burr per target (mean over runs) ===")
    print(burr_df.round(1).to_string())
    print("\n=== Cumulative miscuts per target (mean over runs) ===")
    print(miscut_df.round(2).to_string())

    # Win counts (ties shared)
    def share_wins(values):
        is_min = values == values.min(axis=1, keepdims=True)
        return (is_min / is_min.sum(axis=1, keepdims=True)).sum(axis=0)

    burr_wins = share_wins(burr_df.values)
    miscut_wins = share_wins(miscut_df.values)

    print("\n=== Win counts (tied wins shared) ===")
    print(f"  {'scheme':<22} {'burr wins':>10} {'miscut wins':>12}")
    for s in SCHEMES:
        idx = s - 1
        print(f"  {SCHEMES[s]:<22} {burr_wins[idx]:>10.2f} {miscut_wins[idx]:>12.2f}")

    # ---- aggregated convergence (per-target normalization) ----
    n_iters = N_ITER + 1
    burr_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    miscut_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    n_valid = 0
    for target in targets:
        # Per-target baseline = median initial best across schemes & runs
        inits = []
        for s in SCHEMES:
            H = np.array(all_results[target][s]["hist"])
            inits.append(H[:, 0])
        baseline = float(np.median(np.concatenate(inits)))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        for s in SCHEMES:
            H = np.array(all_results[target][s]["hist"])
            H = np.where(np.isfinite(H), H, INFEASIBLE_PENALTY)
            burr_curves[s] += (H / baseline).mean(axis=0)
            M = np.array(all_results[target][s]["miscuts"])
            miscut_curves[s] += M.mean(axis=0)
        n_valid += 1
    for s in SCHEMES:
        burr_curves[s] /= n_valid
        miscut_curves[s] /= n_valid

    # ---- plot ----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 0])
    ax_mw = fig.add_subplot(gs[1, 1])

    iters = np.arange(n_iters)
    for s, name in SCHEMES.items():
        ax_b.plot(iters, burr_curves[s], label=name, color=COLORS[s - 1], linewidth=2)
        ax_m.plot(iters, miscut_curves[s], label=name, color=COLORS[s - 1], linewidth=2)

    ax_b.set_xlabel("BO iteration")
    ax_b.set_ylabel("Best burr / median initial burr")
    ax_b.set_title(f"Aggregated burr convergence (across {len(targets)} targets)")
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3)

    ax_m.set_xlabel("BO iteration")
    ax_m.set_ylabel("Cumulative miscuts (mean across targets)")
    ax_m.set_title(f"Aggregated miscut convergence (across {len(targets)} targets)")
    ax_m.legend(fontsize=9)
    ax_m.grid(alpha=0.3)

    names = [SCHEMES[s] for s in SCHEMES]
    ax_bw.bar(range(len(SCHEMES)), burr_wins, color=COLORS)
    ax_bw.set_xticks(range(len(SCHEMES)))
    ax_bw.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_bw.set_ylabel("# targets won (tied wins shared)")
    ax_bw.set_title("Burr win count")
    ax_bw.grid(alpha=0.3, axis="y")

    ax_mw.bar(range(len(SCHEMES)), miscut_wins, color=COLORS)
    ax_mw.set_xticks(range(len(SCHEMES)))
    ax_mw.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_mw.set_ylabel("# targets won (tied wins shared)")
    ax_mw.set_title("Miscut win count")
    ax_mw.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"BO comparison across {len(targets)} target tasks "
        f"(N_RUNS={N_RUNS}, N_ITER={N_ITER})",
        fontsize=13,
    )
    out = "bayopt_all_targets.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
