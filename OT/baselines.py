"""Baselines (Step 2 of recipe). Run these FIRST. If they win, stop."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from data import load_data
from features import standardize_and_normalize


def _fit_eval(clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def run_baselines():
    data = load_data()
    Xs_list, Xt_lab, Xt_unl, Xt_test, _ = standardize_and_normalize(
        data["Xs_list"], data["Xt_train_labeled"],
        data["Xt_train_unlabeled"], data["Xt_test"],
    )
    ys_list = data["ys_list"]
    yt_lab = data["yt_train_labeled"]
    yt_test = data["yt_test"]

    results = {}

    # 1. Target-only
    clf = LogisticRegression(max_iter=1000, C=1.0)
    results["target_only_logreg"] = _fit_eval(clf, Xt_lab, yt_lab, Xt_test, yt_test)

    # 2. Pooled ERM (all sources + target labeled)
    X_pool = np.vstack(Xs_list + [Xt_lab])
    y_pool = np.concatenate(ys_list + [yt_lab])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    results["pooled_erm_logreg"] = _fit_eval(clf, X_pool, y_pool, Xt_test, yt_test)

    # 3. Pretrain on sources, fine-tune on target (MLP, since logreg can't really
    #    "fine-tune"; we use warm_start on MLP).
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=200,
        random_state=0, warm_start=True,
    )
    X_src = np.vstack(Xs_list)
    y_src = np.concatenate(ys_list)
    mlp.fit(X_src, y_src)
    # Continue training on target labeled data
    mlp.max_iter = 100
    mlp.fit(Xt_lab, yt_lab)
    results["pretrain_finetune_mlp"] = float(mlp.score(Xt_test, yt_test))

    # Also: pooled ERM with MLP
    mlp2 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
    mlp2.fit(X_pool, y_pool)
    results["pooled_erm_mlp"] = float(mlp2.score(Xt_test, yt_test))

    print("=== Baselines ===")
    for k, v in results.items():
        print(f"  {k:30s}  {v:.4f}")

    with open("results_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_baselines()
