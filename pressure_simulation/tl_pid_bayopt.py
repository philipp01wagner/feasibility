"""Transfer-learning BO for PID gains across chamber-plant variants.

10 plant tasks differ in (chamber volume, mass inflow). 9 are sources with
N pre-sampled (PID-gain, ITAE) points each. We compare vanilla BO vs
RGPE-BO on the held-out 10th plant.

Run from project root with the weldopt env:
    "C:/Users/phw/Anaconda3/envs/weldopt/python.exe" pressure_simulation/tl_pid_bayopt.py
"""

from __future__ import annotations

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube, Sobol
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ChamberEnv import ChamberEnv
from pid import PID

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------- config --------------------------------------
INITIAL_PRESSURE = 10.0
TARGET_PRESSURE = 5.0
SIM_DURATION = 5.0
SAMPLE_RATE = 0.05
TIME_STEP = 0.005

# PID search space (log-uniform on all three gains, same as optimize_pid_bo.py)
KP_BOUNDS = (1e-4, 5.0)
KI_BOUNDS = (1e-4, 5.0)
KD_BOUNDS = (1e-4, 0.5)
PID_BOUNDS = np.array([
    [np.log10(KP_BOUNDS[0]), np.log10(KP_BOUNDS[1])],
    [np.log10(KI_BOUNDS[0]), np.log10(KI_BOUNDS[1])],
    [np.log10(KD_BOUNDS[0]), np.log10(KD_BOUNDS[1])],
])
DIM = 3

# Plant variation
N_TASKS = 10
V_BOUNDS = (10.0, 200.0)   # chamber volume (L), log scale
QM_BOUNDS = (0.05, 0.5)    # mass inflow (g/s), log scale

# Experiment
N_PER_SOURCE = 30
N_TARGET_INIT = 5
N_BO_ITER = 25
N_SEEDS = 3
N_CANDIDATES = 1500
RGPE_SAMPLES = 30
SEED = 42


def to_pid_gains(x_log):
    return 10.0 ** np.atleast_1d(x_log).astype(float)


# ---------------------------- task generation -----------------------------
def generate_tasks(seed=SEED):
    """N_TASKS LHS samples in log-(V, qm) space."""
    sampler = LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=N_TASKS)
    log_V_lo, log_V_hi = np.log10(V_BOUNDS[0]), np.log10(V_BOUNDS[1])
    log_q_lo, log_q_hi = np.log10(QM_BOUNDS[0]), np.log10(QM_BOUNDS[1])
    log_V = log_V_lo + unit[:, 0] * (log_V_hi - log_V_lo)
    log_q = log_q_lo + unit[:, 1] * (log_q_hi - log_q_lo)
    return [(float(10 ** v), float(10 ** q)) for v, q in zip(log_V, log_q)]


# ---------------------------- objective -----------------------------------
# Stability thresholds: a controller is stable if its pressure has converged
# near the target by the end of the simulation AND isn't oscillating.
STABLE_FINAL_TOL = 0.1    # |p[-1] - target| in Torr
STABLE_BAND_TOL = 0.2     # peak-to-peak over the last 1 s, in Torr
STABLE_TAIL_S = 1.0       # length of the "settled" tail window, in s


def evaluate_pid(kp, ki, kd, V, qm, *, return_stable=False):
    env_config = {
        "p_goal": TARGET_PRESSURE,
        "sample_rate": SAMPLE_RATE,
        "time_step": TIME_STEP,
        "max_steps": int(SIM_DURATION / SAMPLE_RATE) + 5,
        "volume_chamb": float(V),
        "qm": float(qm),
    }
    env = ChamberEnv(env_config)
    env.reset(initial_pressure=INITIAL_PRESSURE, goal_pressure=TARGET_PRESSURE)
    pid = PID(
        dt=SAMPLE_RATE,
        kp=float(kp), ki=float(ki), kd=float(kd),
        u_bounds=(0.0, 1.0),
        use_antiwindup=True, clamping_antiwindup=True,
    )
    times = [0.0]
    pressures = [float(env.p)]
    n_steps = int(np.ceil(SIM_DURATION / SAMPLE_RATE))
    for _ in range(n_steps):
        measurement = float(env.p)
        error = measurement - TARGET_PRESSURE
        u, _ = pid.update(error, measurement, desired_value=TARGET_PRESSURE)
        action = pid.transform_control_variable(u)
        obs, _, _, info = env.step(action, duration=SAMPLE_RATE)
        times.append(float(info["time"]))
        pressures.append(float(obs[0]))
    times = np.array(times)
    pressures = np.array(pressures)
    abs_err = np.abs(pressures - TARGET_PRESSURE)
    itae = float(np.trapz(times * abs_err, times))

    if not return_stable:
        return itae

    tail = max(1, int(STABLE_TAIL_S / SAMPLE_RATE))
    final_err = abs(pressures[-1] - TARGET_PRESSURE)
    band = float(pressures[-tail:].max() - pressures[-tail:].min())
    stable = (final_err <= STABLE_FINAL_TOL) and (band <= STABLE_BAND_TOL)
    return itae, bool(stable)


def lhs_pid(n, seed):
    """LHS in PID search box (log-coords) for source-data sampling."""
    sampler = LatinHypercube(d=DIM, seed=seed)
    unit = sampler.random(n=n)
    return PID_BOUNDS[:, 0] + unit * (PID_BOUNDS[:, 1] - PID_BOUNDS[:, 0])


# ---------------------------- GP utilities --------------------------------
def make_kernel():
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )


def fit_gp(X, y_log, seed=SEED):
    sc = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(
        kernel=make_kernel(), normalize_y=True,
        n_restarts_optimizer=2, random_state=seed,
    )
    gp.fit(sc.transform(X), y_log)
    return gp, sc


def gp_predict(pair, X):
    gp, sc = pair
    return gp.predict(sc.transform(X), return_std=True)


def expected_improvement(mean, std, y_best, xi=0.01):
    imp = y_best - mean - xi
    Z = imp / np.where(std > 0, std, 1e-9)
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return np.where(std > 0, np.maximum(ei, 0.0), 0.0)


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


def rgpe_weights(source_gps, X_obs, y_obs_log, rng):
    n_obs = len(X_obs)
    if n_obs < 2:
        n = len(source_gps)
        return np.full(n, 1.0 / n), 0.0
    losses = []
    for gp_pair in source_gps:
        m, s = gp_predict(gp_pair, X_obs)
        samples = rng.normal(m, s + 1e-6, size=(RGPE_SAMPLES, n_obs))
        losses.append(_pairwise_rank_loss(samples, y_obs_log))
    losses = np.array(losses)
    scale = max(np.median(losses), 1e-3)
    src_w = np.exp(-losses / scale)
    src_w /= src_w.sum()
    target_w = float(np.clip(1.0 / (1.0 + np.exp(-(n_obs - 5) / 5.0)), 0.0, 0.5))
    src_w *= 1.0 - target_w
    return src_w, target_w


def rgpe_predict(source_gps, target_gp, src_w, target_w, X):
    means, vars_, ws = [], [], []
    for gp_pair, w in zip(source_gps, src_w):
        m, s = gp_predict(gp_pair, X)
        means.append(m); vars_.append(s ** 2); ws.append(w)
    if target_gp is not None and target_w > 0:
        m, s = gp_predict(target_gp, X)
        means.append(m); vars_.append(s ** 2); ws.append(target_w)
    means = np.array(means); vars_ = np.array(vars_); ws = np.array(ws)
    ws = ws / max(ws.sum(), 1e-12)
    mean = (ws[:, None] * means).sum(axis=0)
    var = (ws[:, None] * (vars_ + (means - mean) ** 2)).sum(axis=0)
    return mean, np.sqrt(np.maximum(var, 1e-12))


# ---------------------------- source classifiers --------------------------
def fit_source_classifiers(source_X_list, source_stable_list):
    """One SVC stability classifier per source task (probability=True)."""
    out = []
    for Xs, stab in zip(source_X_list, source_stable_list):
        if len(np.unique(stab.astype(int))) < 2:
            out.append(None)  # single-class source -> skip its vote
            continue
        sc = StandardScaler().fit(Xs)
        clf = SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
        clf.fit(sc.transform(Xs), stab.astype(int))
        out.append((clf, sc))
    return out


def voted_stable(source_clfs, weights, X):
    """Weighted vote of per-source SVC.predict_proba(stable=1)."""
    use = [(c, w) for c, w in zip(source_clfs, weights) if c is not None and w > 0]
    if not use:
        return np.full(X.shape[0], 1.0)
    ws = np.array([w for _, w in use])
    ws /= ws.sum()
    probas = np.array([
        c[0].predict_proba(c[1].transform(X))[:, 1] for c, _ in use
    ])
    return ws @ probas


# ---------------------------- BO loops ------------------------------------
SCHEMES = {
    1: "Vanilla BO",
    2: "RGPE",
    3: "cBO target-clf",
    4: "cBO RGPE-vote",
    5: "cBO equal-vote",
}
COLORS = ["#2E86AB", "#E63946", "#06A77D", "#FFB400", "#7B2CBF"]


def _initial_target_points(seed):
    """Sobol init shared across schemes for a paired comparison."""
    sobol = Sobol(d=DIM, seed=seed)
    unit = sobol.random(n=N_TARGET_INIT)
    return PID_BOUNDS[:, 0] + unit * (PID_BOUNDS[:, 1] - PID_BOUNDS[:, 0])


def bo_loop(scheme, target_V, target_qm, source_gps, source_clfs, seed):
    """Run BO for one scheme. Returns (best-ITAE history, cumulative-unstable history)."""
    rng = np.random.default_rng(seed)
    X = _initial_target_points(seed)
    y = np.empty(N_TARGET_INIT)
    stable = np.empty(N_TARGET_INIT, dtype=bool)
    for k, x in enumerate(X):
        y[k], stable[k] = evaluate_pid(
            *to_pid_gains(x), target_V, target_qm, return_stable=True
        )
    history = [float(y.min())]
    unstable_history = [int((~stable).sum())]

    for _ in range(N_BO_ITER):
        X_cand = rng.uniform(PID_BOUNDS[:, 0], PID_BOUNDS[:, 1],
                             size=(N_CANDIDATES, DIM))
        y_log = np.log1p(y)

        if scheme == 1:  # Vanilla BO
            target_gp = fit_gp(X, y_log, seed=seed)
            mean, std = gp_predict(target_gp, X_cand)
            ei = expected_improvement(mean, std, y_log.min())
            x_next = X_cand[np.argmax(ei)]

        elif scheme == 2:  # RGPE
            target_gp = fit_gp(X, y_log, seed=seed)
            src_w, tgt_w = rgpe_weights(source_gps, X, y_log, rng)
            mean, std = rgpe_predict(source_gps, target_gp, src_w, tgt_w, X_cand)
            ei = expected_improvement(mean, std, y_log.min())
            x_next = X_cand[np.argmax(ei)]

        elif scheme in (3, 4, 5):
            # Constrained BO: surrogate from RGPE on stable-only target data
            # when available, else from all data. Acquisition gated by P(stable).
            if stable.sum() >= 2:
                X_use = X[stable]
                y_use = np.log1p(y[stable])
            else:
                X_use = X
                y_use = y_log
            target_gp = fit_gp(X_use, y_use, seed=seed)
            src_w, tgt_w = rgpe_weights(source_gps, X_use, y_use, rng)
            mean, std = rgpe_predict(source_gps, target_gp, src_w, tgt_w, X_cand)
            ei = expected_improvement(mean, std, y_use.min())

            if scheme == 3:
                stab_int = stable.astype(int)
                if len(np.unique(stab_int)) < 2:
                    p_stable = np.ones(len(X_cand))
                else:
                    sc = StandardScaler().fit(X)
                    clf = SVC(kernel="rbf", gamma="scale",
                              probability=True, random_state=seed)
                    clf.fit(sc.transform(X), stab_int)
                    p_stable = clf.predict_proba(sc.transform(X_cand))[:, 1]
            elif scheme == 4:
                # RGPE-weighted vote of source classifiers (uses src_w directly)
                p_stable = voted_stable(source_clfs, src_w, X_cand)
            else:  # scheme == 5
                eq_w = np.full(len(source_clfs), 1.0 / len(source_clfs))
                p_stable = voted_stable(source_clfs, eq_w, X_cand)

            x_next = X_cand[np.argmax(ei * p_stable)]
        else:
            raise ValueError(f"unknown scheme {scheme}")

        cost, is_stable = evaluate_pid(
            *to_pid_gains(x_next), target_V, target_qm, return_stable=True
        )
        X = np.vstack([X, x_next])
        y = np.append(y, cost)
        stable = np.append(stable, is_stable)
        history.append(float(y.min()))
        unstable_history.append(int((~stable).sum()))

    return np.array(history), np.array(unstable_history)


# ---------------------------- main ----------------------------------------
def sample_all_task_data(tasks):
    """Sample N_PER_SOURCE LHS PID configs and evaluate ITAE+stability for every task."""
    all_X, all_y, all_stable = [], [], []
    print(f"Sampling {N_PER_SOURCE} LHS PID configs per task "
          f"(stability tol: |p[-1]-target|<={STABLE_FINAL_TOL}, "
          f"last-{STABLE_TAIL_S}s p2p<={STABLE_BAND_TOL})...\n")
    print(f"  {'task':>4}  {'V (L)':>7}  {'qm (g/s)':>9}  "
          f"{'best ITAE':>10}  {'median':>8}  {'stable':>9}")
    t0 = time.time()
    for i, (V, qm) in enumerate(tasks):
        Xs = lhs_pid(N_PER_SOURCE, seed=SEED + i + 1)
        ys = np.empty(N_PER_SOURCE)
        stab = np.empty(N_PER_SOURCE, dtype=bool)
        for k, x in enumerate(Xs):
            ys[k], stab[k] = evaluate_pid(
                *to_pid_gains(x), V, qm, return_stable=True
            )
        all_X.append(Xs); all_y.append(ys); all_stable.append(stab)
        print(f"  {i:>4}  {V:7.2f}  {qm:9.4f}  "
              f"{ys.min():10.3f}  {np.median(ys):8.3f}  "
              f"{int(stab.sum()):>3}/{N_PER_SOURCE}")
    print(f"  Total: {time.time() - t0:.1f}s")
    return all_X, all_y, all_stable


def run_for_target(target_idx, tasks, all_X, all_y, all_stable):
    """Run all 5 schemes x N_SEEDS for target_idx; return per-scheme histories."""
    target_V, target_qm = tasks[target_idx]
    src_indices = [i for i in range(N_TASKS) if i != target_idx]
    src_X = [all_X[i] for i in src_indices]
    src_y = [all_y[i] for i in src_indices]
    src_stab = [all_stable[i] for i in src_indices]

    source_gps = [fit_gp(Xs, np.log1p(ys)) for Xs, ys in zip(src_X, src_y)]
    source_clfs = fit_source_classifiers(src_X, src_stab)

    out = {s: {"hist": [], "unstable": []} for s in SCHEMES}
    for s_idx in range(N_SEEDS):
        seed = SEED + 100 + s_idx
        for s in SCHEMES:
            hist, unstable = bo_loop(
                s, target_V, target_qm, source_gps, source_clfs, seed
            )
            out[s]["hist"].append(hist)
            out[s]["unstable"].append(unstable)
    return out


def main():
    print("Generating 10 plant tasks (LHS in log-(V, qm))...")
    tasks = generate_tasks()
    for i, (V, qm) in enumerate(tasks):
        print(f"  Task {i:>2}: V = {V:7.2f} L,  qm = {qm:.4f} g/s")
    print()

    all_X, all_y, all_stable = sample_all_task_data(tasks)

    print(f"\nSweep: each of {N_TASKS} tasks held out as target in turn")
    print(f"  {N_SEEDS} seeds x ({N_TARGET_INIT} init + {N_BO_ITER} BO iters) "
          f"x {len(SCHEMES)} schemes per target\n")

    sweep_results = {}
    t_start = time.time()
    for ti in range(N_TASKS):
        t0 = time.time()
        sweep_results[ti] = run_for_target(ti, tasks, all_X, all_y, all_stable)
        elapsed = time.time() - t0
        total = time.time() - t_start
        eta = total / (ti + 1) * (N_TASKS - ti - 1)
        # quick per-target line
        finals = {s: float(np.mean(np.array(sweep_results[ti][s]["hist"])[:, -1]))
                  for s in SCHEMES}
        unstables = {s: float(np.mean(np.array(sweep_results[ti][s]["unstable"])[:, -1]))
                     for s in SCHEMES}
        print(f"[{ti + 1:>2}/{N_TASKS}] target task {ti} "
              f"(V={tasks[ti][0]:6.2f}L qm={tasks[ti][1]:.3f})  "
              f"{elapsed:5.0f}s  total {total / 60:5.1f}m  eta {eta / 60:5.1f}m")
        for s in SCHEMES:
            print(f"        {SCHEMES[s]:<22} "
                  f"ITAE={finals[s]:7.4f}  unstable={unstables[s]:5.2f}")

    # ---- summary tables ----
    burr_table = {SCHEMES[s]: [] for s in SCHEMES}
    miscut_table = {SCHEMES[s]: [] for s in SCHEMES}
    for ti in range(N_TASKS):
        for s in SCHEMES:
            H = np.array(sweep_results[ti][s]["hist"])
            U = np.array(sweep_results[ti][s]["unstable"])
            burr_table[SCHEMES[s]].append(float(H[:, -1].mean()))
            miscut_table[SCHEMES[s]].append(float(U[:, -1].mean()))

    target_labels = [f"task{i:02d}_V{tasks[i][0]:05.1f}_qm{tasks[i][1]:.3f}"
                     for i in range(N_TASKS)]
    itae_df = pd.DataFrame(burr_table, index=target_labels)
    unstable_df = pd.DataFrame(miscut_table, index=target_labels)
    itae_df.to_csv("tl_pid_sweep_itae.csv")
    unstable_df.to_csv("tl_pid_sweep_unstable.csv")
    print("\nSaved tl_pid_sweep_itae.csv, tl_pid_sweep_unstable.csv")

    print("\n=== Final best ITAE per target (mean over seeds) ===")
    print(itae_df.round(4).to_string())
    print("\n=== Cumulative unstable evals per target (mean over seeds) ===")
    print(unstable_df.round(2).to_string())

    # win counts (ties shared)
    def share_wins(values):
        is_min = values == values.min(axis=1, keepdims=True)
        return (is_min / is_min.sum(axis=1, keepdims=True)).sum(axis=0)

    itae_wins = share_wins(itae_df.values)
    unstable_wins = share_wins(unstable_df.values)
    print("\n=== Win counts (tied wins shared, out of 10 targets) ===")
    print(f"  {'scheme':<22} {'ITAE wins':>10} {'unstable wins':>14}")
    for s in SCHEMES:
        idx = s - 1
        print(f"  {SCHEMES[s]:<22} {itae_wins[idx]:>10.2f} {unstable_wins[idx]:>14.2f}")

    # ---- aggregated curves: per-target normalize ITAE by initial best ----
    n_iters = N_BO_ITER + 1
    burr_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    unstable_curves = {s: np.zeros(n_iters) for s in SCHEMES}
    for ti in range(N_TASKS):
        baselines = []
        for s in SCHEMES:
            H = np.array(sweep_results[ti][s]["hist"])
            baselines.append(H[:, 0])
        baseline = float(np.median(np.concatenate(baselines)))
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        for s in SCHEMES:
            H = np.array(sweep_results[ti][s]["hist"])
            burr_curves[s] += (H / baseline).mean(axis=0)
            U = np.array(sweep_results[ti][s]["unstable"])
            unstable_curves[s] += U.mean(axis=0)
    for s in SCHEMES:
        burr_curves[s] /= N_TASKS
        unstable_curves[s] /= N_TASKS

    # ---- plot 4-panel summary ----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 0])
    ax_mw = fig.add_subplot(gs[1, 1])

    iters = np.arange(n_iters)
    for s, name in SCHEMES.items():
        ax_b.plot(iters, burr_curves[s], label=name,
                  color=COLORS[s - 1], linewidth=2)
        ax_m.plot(iters, unstable_curves[s], label=name,
                  color=COLORS[s - 1], linewidth=2)

    ax_b.set_xlabel("BO iteration")
    ax_b.set_ylabel("Best ITAE / median initial ITAE")
    ax_b.set_yscale("log")
    ax_b.set_title(f"Aggregated ITAE convergence (across {N_TASKS} targets)")
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3, which="both")

    ax_m.set_xlabel("BO iteration")
    ax_m.set_ylabel("Cumulative unstable evals (mean across targets)")
    ax_m.set_title(f"Aggregated unstable convergence (across {N_TASKS} targets)")
    ax_m.legend(fontsize=9)
    ax_m.grid(alpha=0.3)

    names = [SCHEMES[s] for s in SCHEMES]
    ax_bw.bar(range(len(SCHEMES)), itae_wins, color=COLORS)
    ax_bw.set_xticks(range(len(SCHEMES)))
    ax_bw.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_bw.set_ylabel("# targets won (ties shared)")
    ax_bw.set_title("ITAE win count")
    ax_bw.grid(alpha=0.3, axis="y")

    ax_mw.bar(range(len(SCHEMES)), unstable_wins, color=COLORS)
    ax_mw.set_xticks(range(len(SCHEMES)))
    ax_mw.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_mw.set_ylabel("# targets won (ties shared)")
    ax_mw.set_title("Unstable-eval win count")
    ax_mw.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Constrained TL-BO for PID gains - sweep across {N_TASKS} target plants\n"
        f"{N_SEEDS} seeds x {N_TARGET_INIT} init + {N_BO_ITER} BO iters per target",
        fontsize=13,
    )
    fig.tight_layout()
    out = "tl_pid_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
