"""
JCPOT (Redko, Courty, Flamary, Tuia, AISTATS 2019):
multi-source joint distribution OT under target shift.
Aligns class proportions across sources and target jointly.
"""
import json
import numpy as np
from ot.da import JCPOTTransport
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from data import load_data
from features import standardize_and_normalize


def run_jcpot(reg_e=0.1, max_iter=10):
    data = load_data()
    Xs_list, Xt_lab, Xt_unl, Xt_test, _ = standardize_and_normalize(
        data["Xs_list"], data["Xt_train_labeled"],
        data["Xt_train_unlabeled"], data["Xt_test"],
    )
    ys_list = data["ys_list"]
    yt_lab = data["yt_train_labeled"]
    yt_test = data["yt_test"]

    # Use unlabeled target for OT alignment (more samples = better estimate).
    Xt_for_ot = np.vstack([Xt_unl, Xt_lab])

    jcpot = JCPOTTransport(reg_e=reg_e, max_iter=max_iter, tol=1e-6, verbose=False)
    jcpot.fit(Xs=Xs_list, ys=ys_list, Xt=Xt_for_ot)

    # Transport each source to the target distribution.
    transported = [jcpot.transform(Xs=Xs, batch_size=128) for Xs in Xs_list]

    X_pool = np.vstack(transported + [Xt_lab])
    y_pool = np.concatenate(ys_list + [yt_lab])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_pool, y_pool)
    acc_logreg = float(clf.score(Xt_test, yt_test))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
    mlp.fit(X_pool, y_pool)
    acc_mlp = float(mlp.score(Xt_test, yt_test))

    results = {
        "jcpot_logreg": acc_logreg,
        "jcpot_mlp": acc_mlp,
    }
    print("=== JCPOT (multi-source joint OT) ===")
    print(f"  logreg: {acc_logreg:.4f}")
    print(f"  mlp:    {acc_mlp:.4f}")

    with open("results_jcpot.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_jcpot()
