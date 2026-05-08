"""
Data loading for the laser-cutting feasibility transfer-learning task.

Each "task" is a (thickness, ltt_name, machine, adb, laser_power, material) combination.
Features: feedrate, gas_pressure, focal_position (d=3).
Labels: 1 = feasible (burr & roughness >= 0), 0 = infeasible (n_classes=2).

Target task = a single chosen process; source tasks = all other process
combinations that contain both feasible and infeasible points.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)

# --- Defaults for the laser-cutting dataset ---
DEFAULT_PKL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "task_df_dict.pkl",
)
DEFAULT_TARGET_TASK = "150_ST150MD0-N2H0-30-2_L76_0.4_10000_H450"
FEATURE_COLS = ["feedrate", "gas_pressure", "focal_position"]


def _make_feasibility_label(df):
    return ((df["burr_evaluated"] >= 0) & (df["roughness_z_evaluated"] >= 0)).astype(int).values


def _make_one_task(n_samples, d, n_classes, shift, rotation_strength, seed):
    """Generate one classification task with a domain-specific shift + rotation."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=d,
        n_informative=max(2, d // 2),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    # Apply a random rotation + shift to simulate domain gap.
    rng = np.random.default_rng(seed)
    A = np.eye(d) + rotation_strength * rng.standard_normal((d, d))
    X = X @ A + shift
    return X.astype(np.float32), y.astype(np.int64)


def make_synthetic(n_sources=5, d=20, n_classes=4, n_per_source=500,
                   n_target_labeled=50, n_target_unlabeled=300, n_target_test=500):
    """Generate N source tasks + 1 target task."""
    Xs_list, ys_list = [], []
    for i in range(n_sources):
        # Each source has its own moderate shift.
        shift = 0.5 * RNG.standard_normal(d)
        X, y = _make_one_task(
            n_samples=n_per_source, d=d, n_classes=n_classes,
            shift=shift, rotation_strength=0.05, seed=100 + i,
        )
        Xs_list.append(X)
        ys_list.append(y)

    # Target has a larger shift -> nontrivial gap.
    target_shift = 1.5 * RNG.standard_normal(d)
    Xt_all, yt_all = _make_one_task(
        n_samples=n_target_labeled + n_target_unlabeled + n_target_test,
        d=d, n_classes=n_classes,
        shift=target_shift, rotation_strength=0.10, seed=999,
    )

    # Split target into labeled (small), unlabeled (for OT), test.
    Xt_lab, Xt_rest, yt_lab, yt_rest = train_test_split(
        Xt_all, yt_all, train_size=n_target_labeled, stratify=yt_all, random_state=0,
    )
    Xt_unl, Xt_test, _, yt_test = train_test_split(
        Xt_rest, yt_rest, train_size=n_target_unlabeled, stratify=yt_rest, random_state=0,
    )

    return {
        "Xs_list": Xs_list,
        "ys_list": ys_list,
        "Xt_train_labeled": Xt_lab.astype(np.float32),
        "yt_train_labeled": yt_lab.astype(np.int64),
        "Xt_train_unlabeled": Xt_unl.astype(np.float32),
        "Xt_test": Xt_test.astype(np.float32),
        "yt_test": yt_test.astype(np.int64),
        "n_classes": n_classes,
        "d": d,
    }


def load_data(
    pkl_path=DEFAULT_PKL_PATH,
    target_task=DEFAULT_TARGET_TASK,
    n_target_labeled=10,
    n_target_unlabeled=30,
    min_source_size=5,
    standardize=True,
    seed=42,
):
    """Load the laser-cutting feasibility dataset.

    Parameters
    ----------
    pkl_path : str
        Path to `task_df_dict.pkl` (dict[task_name -> DataFrame]).
    target_task : str
        Key of the target task in the dict.
    n_target_labeled : int
        Size of the small labeled target set.
    n_target_unlabeled : int
        Size of the unlabeled target set used for OT alignment.
    min_source_size : int
        Discard source tasks with fewer points than this (or only one class).
    standardize : bool
        If True, fit a StandardScaler on the labeled target features and
        apply it to all sources + the target sets (so OT distances are
        comparable across tasks).
    seed : int
        Random seed for the target train/unlabeled/test split.

    Returns
    -------
    dict with keys: Xs_list, ys_list, Xt_train_labeled, yt_train_labeled,
                    Xt_train_unlabeled, Xt_test, yt_test, n_classes, d,
                    source_task_names, scaler
    """
    with open(pkl_path, "rb") as f:
        task_df_dict = pickle.load(f)

    if target_task not in task_df_dict:
        raise KeyError(
            f"Target task {target_task!r} not in {pkl_path}. "
            f"Available: {list(task_df_dict.keys())[:5]}..."
        )

    # --- Target task: split into labeled / unlabeled / test ---
    df_target = task_df_dict[target_task]
    X_target_all = df_target[FEATURE_COLS].values.astype(np.float32)
    y_target_all = _make_feasibility_label(df_target).astype(np.int64)

    if len(np.unique(y_target_all)) < 2:
        raise ValueError(f"Target task {target_task!r} has only one class.")

    X_lab, X_rest, y_lab, y_rest = train_test_split(
        X_target_all, y_target_all,
        train_size=n_target_labeled,
        stratify=y_target_all,
        random_state=seed,
    )
    n_unlab = min(n_target_unlabeled, max(1, len(X_rest) - 1))
    X_unl, X_test, _, y_test = train_test_split(
        X_rest, y_rest,
        train_size=n_unlab,
        stratify=y_rest,
        random_state=seed,
    )

    # --- Standardize using the (small) labeled target set ---
    scaler = StandardScaler().fit(X_lab) if standardize else None

    def _maybe_scale(X):
        return scaler.transform(X).astype(np.float32) if scaler is not None else X.astype(np.float32)

    X_lab_s = _maybe_scale(X_lab)
    X_unl_s = _maybe_scale(X_unl)
    X_test_s = _maybe_scale(X_test)

    # --- Source tasks: every other task with both classes and enough points ---
    Xs_list, ys_list, source_task_names = [], [], []
    for name, df in task_df_dict.items():
        if name == target_task:
            continue
        if len(df) < min_source_size:
            continue
        X_src = df[FEATURE_COLS].values.astype(np.float32)
        y_src = _make_feasibility_label(df).astype(np.int64)
        if len(np.unique(y_src)) < 2:
            continue
        Xs_list.append(_maybe_scale(X_src))
        ys_list.append(y_src)
        source_task_names.append(name)

    return {
        "Xs_list": Xs_list,
        "ys_list": ys_list,
        "Xt_train_labeled": X_lab_s,
        "yt_train_labeled": y_lab.astype(np.int64),
        "Xt_train_unlabeled": X_unl_s,
        "Xt_test": X_test_s,
        "yt_test": y_test.astype(np.int64),
        "n_classes": 2,
        "d": X_lab_s.shape[1],
        "source_task_names": source_task_names,
        "target_task": target_task,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = load_data()
    print(f"N sources: {len(data['Xs_list'])}")
    print(f"Feature dim: {data['d']}, n_classes: {data['n_classes']}")
    for i, (X, y) in enumerate(zip(data["Xs_list"], data["ys_list"])):
        print(f"  Source {i}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
    print(f"Target labeled: {data['Xt_train_labeled'].shape}")
    print(f"Target unlabeled: {data['Xt_train_unlabeled'].shape}")
    print(f"Target test: {data['Xt_test'].shape}")
