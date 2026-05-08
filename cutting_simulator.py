import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_task_simulator(
    task_name, task_df_dict, feat_cols=["feedrate", "gas_pressure", "focal_position"]
):
    """Build a simulator that returns burr value (or -1 if infeasible).

    The returned ``simulator(feedrate, gas_pressure, focal_position)`` accepts
    scalars (returns ``float``) or array-likes of equal length (returns an
    ``ndarray``). Infeasible points are encoded as ``-1.0``.
    """
    df = task_df_dict[task_name]
    X = df[feat_cols].values
    burr = df["burr_evaluated"].values
    roughness = df["roughness_z_evaluated"].values

    # Feasibility label: 1 if both burr >= 0 and roughness >= 0
    y_feas = ((burr >= 0) & (roughness >= 0)).astype(int)

    # Classifier
    scaler_clf = StandardScaler().fit(X)
    clf = SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)
    clf.fit(scaler_clf.transform(X), y_feas)

    # Regressor on feasible points only
    feas_mask = y_feas == 1
    scaler_reg = StandardScaler().fit(X[feas_mask])
    reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
    reg.fit(scaler_reg.transform(X[feas_mask]), burr[feas_mask])

    def simulator(feedrate, gas_pressure, focal_position):
        fr = np.asarray(feedrate, dtype=float)
        gp = np.asarray(gas_pressure, dtype=float)
        fp = np.asarray(focal_position, dtype=float)
        scalar = fr.ndim == 0 and gp.ndim == 0 and fp.ndim == 0
        x = np.column_stack([fr.ravel(), gp.ravel(), fp.ravel()])
        is_feas = clf.predict(scaler_clf.transform(x))
        out = np.full(x.shape[0], -1.0)
        feas = is_feas == 1
        if feas.any():
            out[feas] = reg.predict(scaler_reg.transform(x[feas]))
        return float(out[0]) if scalar else out

    return simulator
