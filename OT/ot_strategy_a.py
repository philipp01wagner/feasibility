"""
Strategy A from recipe Step 4:
  - For each source i, fit class-regularized OT from source_i to target.
  - Transport each source.
  - Weight each source by inverse Wasserstein distance to target.
  - Train final classifier on pooled transported sources + target labeled.
"""
import json
import numpy as np
import ot
from ot.da import SinkhornL1l2Transport
from sklearn.linear_model import LogisticRegression

from data import load_data
from features import standardize_and_normalize


def _wasserstein_distance(Xs, Xt, reg=0.1):
    """Sinkhorn divergence between two empirical distributions (uniform weights)."""
    a = np.ones(len(Xs)) / len(Xs)
    b = np.ones(len(Xt)) / len(Xt)
    M = ot.dist(Xs, Xt, metric="sqeuclidean")
    M = M / (M.max() + 1e-8)
    return float(ot.sinkhorn2(a, b, M, reg=reg))


def run_strategy_a(reg_e=0.1, reg_cl=1.0, temperature=1.0, use_target_labels=True):
    data = load_data()
    Xs_list, Xt_lab, Xt_unl, Xt_test, _ = standardize_and_normalize(
        data["Xs_list"], data["Xt_train_labeled"],
        data["Xt_train_unlabeled"], data["Xt_test"],
    )
    ys_list = data["ys_list"]
    yt_lab = data["yt_train_labeled"]
    yt_test = data["yt_test"]

    # OT alignment uses unlabeled target (richer) optionally combined with labeled.
    Xt_for_ot = np.vstack([Xt_unl, Xt_lab]) if use_target_labels else Xt_unl

    transported, distances, per_source_acc = [], [], []
    for i, (Xs, ys) in enumerate(zip(Xs_list, ys_list)):
        # Class-regularized OT (group-lasso per class).
        ot_da = SinkhornL1l2Transport(
            reg_e=reg_e, reg_cl=reg_cl, max_iter=20, tol=1e-6, verbose=False,
        )
        ot_da.fit(Xs=Xs, ys=ys, Xt=Xt_for_ot)
        Xs_t = ot_da.transform(Xs=Xs)
        transported.append(Xs_t)

        # Wasserstein distance source -> target (after transport, should be small).
        d_before = _wasserstein_distance(Xs, Xt_for_ot)
        distances.append(d_before)

        # Diagnostic: train classifier on this transported source alone
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xs_t, ys)
        per_source_acc.append(float(clf.score(Xt_test, yt_test)))
        print(f"  source {i}: W(s,t)={d_before:.4f}  solo_acc={per_source_acc[-1]:.4f}")

    # Source weights: softmax over -distance (closer sources get more weight)
    distances = np.array(distances)
    weights = np.exp(-distances / (temperature * distances.std() + 1e-8))
    weights = weights / weights.sum()
    print(f"Source weights: {np.round(weights, 3)}")

    # Pool transported sources + labeled target, with per-sample weights
    X_pool = np.vstack(transported + [Xt_lab])
    y_pool = np.concatenate(ys_list + [yt_lab])

    sample_weights = []
    for w, Xs_t in zip(weights, transported):
        sample_weights.append(np.full(len(Xs_t), w))
    # Give target labeled samples weight equal to max source weight (they're gold).
    sample_weights.append(np.full(len(Xt_lab), weights.max()))
    sample_weights = np.concatenate(sample_weights)

    # Final classifier
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_pool, y_pool, sample_weight=sample_weights)
    acc_logreg = float(clf.score(Xt_test, yt_test))

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
    mlp.fit(X_pool, y_pool)  # MLPClassifier doesn't accept sample_weight
    acc_mlp = float(mlp.score(Xt_test, yt_test))

    results = {
        "strategy_a_logreg": acc_logreg,
        "strategy_a_mlp": acc_mlp,
        "per_source_solo_acc": per_source_acc,
        "source_weights": weights.tolist(),
        "wasserstein_distances": distances.tolist(),
    }
    print("=== Strategy A (per-source OT + weighted pool) ===")
    print(f"  logreg: {acc_logreg:.4f}")
    print(f"  mlp:    {acc_mlp:.4f}")

    with open("results_strategy_a.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_strategy_a()
