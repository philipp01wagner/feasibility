import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cutting_simulator import create_task_simulator

with open("./data/task_df_dict.pkl", "rb") as f:
    task_df_dict = pickle.load(f)

FEATURE_COLS = ["feedrate", "gas_pressure", "focal_position"]

# Choose the target task
TARGET_TASK = "150_ST150MD0-N2H0-30-2_L76_0.4_10000_H450"

# Source tasks: all others that have both feasible and infeasible points
source_tasks = []
for name, df in task_df_dict.items():
    if name == TARGET_TASK:
        continue
    has_neg = ((df["burr_evaluated"] < 0) | (df["roughness_z_evaluated"] < 0)).any()
    has_pos = ((df["burr_evaluated"] >= 0) & (df["roughness_z_evaluated"] >= 0)).any()
    if has_neg and has_pos:
        source_tasks.append(name)

print(f"Target task: {TARGET_TASK} ({len(task_df_dict[TARGET_TASK])} points)")
print(f"Source tasks: {len(source_tasks)}")
for s in source_tasks:
    print(f"  {s}: {len(task_df_dict[s])} points")

simulator = create_task_simulator(TARGET_TASK, task_df_dict)
print("Simulator ready.")


# === OT-weighted majority-vote experiment =================================
# 1. Train one feasibility classifier per source task.
# 2. Sample N points from TARGET's parameter range, label them with the
#    target simulator.
# 3. For each source task, compute the optimal-transport distance between
#    its data and the simulated samples in 5-D
#    (feedrate, gas_pressure, focal_position, burr, feasibility label).
# 4. Aggregate the per-task classifier predictions on the simulated samples
#    by (a) equal-weight majority vote and (b) inverse-OT-distance vote.
# 5. Compare both voting accuracies against the simulator's labels.

import ot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_SAMPLES = 1000
SEED = 42


def _feas_label(df):
    return (
        (df["burr_evaluated"].values >= 0)
        & (df["roughness_z_evaluated"].values >= 0)
    ).astype(int)


# --- 1. one classifier per source task ---
task_classifiers = {}
for name in source_tasks:
    df = task_df_dict[name]
    X = df[FEATURE_COLS].values
    y = _feas_label(df)
    if len(np.unique(y)) < 2:
        continue  # SVC needs both classes
    scaler = StandardScaler().fit(X)
    clf = SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    clf.fit(scaler.transform(X), y)
    task_classifiers[name] = (clf, scaler)
print(f"Trained {len(task_classifiers)} per-task classifiers.")

# --- 2. sample N datapoints from the simulator ---
rng = np.random.default_rng(SEED)
target_df = task_df_dict[TARGET_TASK]
lo = target_df[FEATURE_COLS].min().values
hi = target_df[FEATURE_COLS].max().values
X_sim = rng.uniform(lo, hi, size=(N_SAMPLES, 3))
burr_sim = simulator(X_sim[:, 0], X_sim[:, 1], X_sim[:, 2])
y_sim = (burr_sim >= 0).astype(int)
print(
    f"Sampled {N_SAMPLES} points from target simulator "
    f"({y_sim.sum()} feasible / {(1 - y_sim).sum()} miscut)."
)

# --- 3. OT distance per task in 5-D space, with shared min-max normalization ---
ordered = list(task_classifiers.keys())


def _to_5d(X3, burr, feas):
    return np.column_stack([X3, burr, feas])


sim_5d = _to_5d(X_sim, burr_sim, y_sim)
task_5d = {}
for name in ordered:
    df = task_df_dict[name]
    task_5d[name] = _to_5d(df[FEATURE_COLS].values, df["burr_evaluated"].values, _feas_label(df))

stacked = np.vstack([sim_5d] + list(task_5d.values()))
mins = stacked.min(axis=0)
maxs = stacked.max(axis=0)
spans = np.where(maxs > mins, maxs - mins, 1.0)


def _normalize(arr):
    return (arr - mins) / spans


sim_n = _normalize(sim_5d)
ot_distances = {}
for name in ordered:
    src_n = _normalize(task_5d[name])
    a = np.full(len(src_n), 1.0 / len(src_n))
    b = np.full(len(sim_n), 1.0 / len(sim_n))
    M = ot.dist(src_n, sim_n, metric="euclidean")
    ot_distances[name] = float(ot.emd2(a, b, M))

# --- 4. soft majority vote via predict_proba ---
# probas[s, n] = P(feasible | x_n) under classifier s
probas = np.array([
    task_classifiers[n][0].predict_proba(task_classifiers[n][1].transform(X_sim))[:, 1]
    for n in ordered
])  # shape (S, N)

p_eq = probas.mean(axis=0)
y_eq = (p_eq >= 0.5).astype(int)

distances = np.array([ot_distances[n] for n in ordered])
inv_d = 1.0 / (distances + 1e-12)
weights = inv_d / inv_d.sum()
p_ot = weights @ probas
y_ot = (p_ot >= 0.5).astype(int)

# --- 5. compare accuracies + per-task diagnostics ---
acc_eq = accuracy_score(y_sim, y_eq)
acc_ot = accuracy_score(y_sim, y_ot)

print()
print("Per-task OT distance to simulated samples and individual accuracy:")
print(f"  {'task':<54} {'OT dist':>10} {'weight':>8} {'acc':>8}")
for n in sorted(ordered, key=lambda k: ot_distances[k]):
    individual_acc = accuracy_score(y_sim, (probas[ordered.index(n)] >= 0.5).astype(int))
    w = weights[ordered.index(n)]
    print(f"  {n[:54]:<54} {ot_distances[n]:>10.4f} {w:>8.3f} {individual_acc:>8.3f}")

print()
print(f"Equal-weight soft vote accuracy:        {acc_eq:.3f}")
print(f"OT-weighted (1/d) soft vote accuracy:   {acc_ot:.3f}")
print(f"Improvement from OT weighting:          {acc_ot - acc_eq:+.3f}")


# === Noise-robustness experiment ==========================================
# Flip a fraction of the simulator labels used inside the OT 5-tuple,
# recompute distances/weights, soft-vote, and evaluate against the CLEAN
# labels. Equal-weight vote is independent of labels (constant baseline);
# OT-weighted vote is what we expect to degrade as label noise grows.
print("\n=== Noise-robustness experiment ===")
print(
    "Label noise is applied to the simulator labels feeding the OT 5-tuple.\n"
    "Accuracy is measured against the clean simulator labels.\n"
)

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_TRIALS = 10


def _ot_weights_for(sim_5d_arr):
    """Recompute OT distances and 1/d weights with a given simulator 5-D array."""
    stacked_n = np.vstack([sim_5d_arr] + list(task_5d.values()))
    mn = stacked_n.min(axis=0)
    mx = stacked_n.max(axis=0)
    sp = np.where(mx > mn, mx - mn, 1.0)
    sim_norm = (sim_5d_arr - mn) / sp
    a_sim = np.full(len(sim_norm), 1.0 / len(sim_norm))
    dists = []
    for nm in ordered:
        src_norm = (task_5d[nm] - mn) / sp
        a_src = np.full(len(src_norm), 1.0 / len(src_norm))
        M_loc = ot.dist(src_norm, sim_norm, metric="euclidean")
        dists.append(float(ot.emd2(a_src, a_sim, M_loc)))
    dists = np.array(dists)
    w = 1.0 / (dists + 1e-12)
    return dists, w / w.sum()


# equal vote is constant w.r.t. label noise — compute once
acc_eq_const = accuracy_score(y_sim, (probas.mean(axis=0) >= 0.5).astype(int))

print(f"  {'noise':>6}  {'equal-weight':>14}  {'OT-weighted (mean +/- std)':>28}  {'delta':>8}")
for p in NOISE_LEVELS:
    accs_ot = []
    for trial in range(N_TRIALS):
        rng_t = np.random.default_rng(SEED + 1 + trial)
        flips = rng_t.uniform(size=N_SAMPLES) < p
        y_noisy = np.where(flips, 1 - y_sim, y_sim)
        sim_5d_noisy = _to_5d(X_sim, burr_sim, y_noisy)
        _, w_t = _ot_weights_for(sim_5d_noisy)
        y_ot_t = ((w_t @ probas) >= 0.5).astype(int)
        accs_ot.append(accuracy_score(y_sim, y_ot_t))
    mean_ot = float(np.mean(accs_ot))
    std_ot = float(np.std(accs_ot))
    print(
        f"  {p:>6.2f}  {acc_eq_const:>14.3f}  "
        f"{mean_ot:>11.3f} +/- {std_ot:<11.3f}  "
        f"{mean_ot - acc_eq_const:>+8.3f}"
    )


# === exp(-d/tau) weighting sweep =========================================
# Sharper-than-1/d weighting concentrates weight on the closest few tasks.
# tau -> inf approaches equal weighting; tau -> 0 isolates the single
# closest task. Same noise grid as above; trial seeds are shared across
# weighting schemes so the comparison is paired.
print("\n=== exp(-d/tau) weighting under label noise ===")
print(f"Mean accuracy over {N_TRIALS} trials per cell.\n")

TAU_VALUES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

header_cells = ["noise", "equal", "1/d"] + [f"tau={t}" for t in TAU_VALUES]
print("  " + "".join(f"{c:>9}" for c in header_cells))

for p in NOISE_LEVELS:
    n_trials = 1 if p == 0.0 else N_TRIALS  # noise=0 is deterministic
    accs_invd = []
    accs_exp = {t: [] for t in TAU_VALUES}
    for trial in range(n_trials):
        rng_t = np.random.default_rng(SEED + 1 + trial)
        flips = rng_t.uniform(size=N_SAMPLES) < p
        y_noisy = np.where(flips, 1 - y_sim, y_sim)
        dists_t, w_invd = _ot_weights_for(_to_5d(X_sim, burr_sim, y_noisy))
        accs_invd.append(accuracy_score(y_sim, ((w_invd @ probas) >= 0.5).astype(int)))
        for tau in TAU_VALUES:
            w_exp = np.exp(-dists_t / tau)
            w_exp /= w_exp.sum()
            accs_exp[tau].append(
                accuracy_score(y_sim, ((w_exp @ probas) >= 0.5).astype(int))
            )

    row_vals = [f"{p:.2f}", f"{acc_eq_const:.3f}", f"{np.mean(accs_invd):.3f}"]
    row_vals += [f"{np.mean(accs_exp[t]):.3f}" for t in TAU_VALUES]
    print("  " + "".join(f"{c:>9}" for c in row_vals))
