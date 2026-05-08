"""Feature preprocessing. L2-normalization is critical before OT."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_and_normalize(Xs_list, Xt_labeled, Xt_unlabeled, Xt_test):
    """Fit StandardScaler on pooled source+target-unlabeled, then L2-normalize.

    Standardization on pooled data avoids per-domain scale artifacts; L2
    normalization puts everything on the unit sphere so OT distances are
    comparable across domains.
    """
    pooled = np.vstack(Xs_list + [Xt_unlabeled])
    scaler = StandardScaler().fit(pooled)

    def proc(X):
        Xs = scaler.transform(X)
        norms = np.linalg.norm(Xs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (Xs / norms).astype(np.float32)

    Xs_list_n = [proc(X) for X in Xs_list]
    return (
        Xs_list_n,
        proc(Xt_labeled),
        proc(Xt_unlabeled),
        proc(Xt_test),
        scaler,
    )
