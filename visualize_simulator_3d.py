"""Build an interactive 3D HTML visualization of the cutting-task simulator.

For each selected task we render:
  * an isosurface of the simulator's feasibility boundary,
  * a volume rendering of the predicted burr inside the feasible region,
  * the underlying training points (feasible vs infeasible).

Model training and inference go through ``cutting_simulator.create_task_simulator``;
the visualization only handles the grid + Plotly traces. A dropdown switches
between tasks. Output: ``simulator_3d.html`` (standalone).
"""

import pickle

import numpy as np
import plotly.graph_objects as go

from cutting_simulator import create_task_simulator

FEATURE_COLS = ["feedrate", "gas_pressure", "focal_position"]
GRID_N = 30
TARGET_TASK = "150_ST150MD0-N2H0-30-2_L76_0.4_10000_H450"
INFEASIBLE_VALUE = -1.0  # convention used by create_task_simulator


def task_volume(task_name, task_df_dict):
    """Sample the simulator on a 3-D grid bounded by the task's data range."""
    df = task_df_dict[task_name]
    feas_actual = (df["burr_evaluated"].values >= 0) & (
        df["roughness_z_evaluated"].values >= 0
    )
    if feas_actual.sum() < 3 or (~feas_actual).sum() < 1:
        return None  # need both classes to make an interesting picture

    simulator = create_task_simulator(task_name, task_df_dict, feat_cols=FEATURE_COLS)

    fr = df["feedrate"].values
    gp = df["gas_pressure"].values
    fp = df["focal_position"].values

    def pad(lo, hi, frac=0.08):
        span = hi - lo if hi > lo else 1.0
        return lo - frac * span, hi + frac * span

    fr_lo, fr_hi = pad(fr.min(), fr.max())
    gp_lo, gp_hi = pad(gp.min(), gp.max())
    fp_lo, fp_hi = pad(fp.min(), fp.max())

    fr_g = np.linspace(fr_lo, fr_hi, GRID_N)
    gp_g = np.linspace(gp_lo, gp_hi, GRID_N)
    fp_g = np.linspace(fp_lo, fp_hi, GRID_N)
    F, G, P = np.meshgrid(fr_g, gp_g, fp_g, indexing="ij")

    # One vectorized call thanks to the simulator's batch support.
    field = simulator(F.ravel(), G.ravel(), P.ravel()).reshape(F.shape)

    return {
        "F": F,
        "G": G,
        "P": P,
        "field": field,
        "fr": fr,
        "gp": gp,
        "fp": fp,
        "burr_actual": df["burr_evaluated"].values,
        "feas_actual": feas_actual,
    }


def main():
    with open("./data/task_df_dict.pkl", "rb") as f:
        task_df_dict = pickle.load(f)

    tasks = []
    for name, df in task_df_dict.items():
        burr = df["burr_evaluated"].values
        rough = df["roughness_z_evaluated"].values
        has_pos = ((burr >= 0) & (rough >= 0)).any()
        has_neg = ((burr < 0) | (rough < 0)).any()
        if has_pos and has_neg and len(df) >= 8:
            tasks.append(name)
    if TARGET_TASK in tasks:
        tasks.remove(TARGET_TASK)
        tasks = [TARGET_TASK] + tasks

    print(f"Building visualization for {len(tasks)} tasks (grid={GRID_N}^3)...")

    fig = go.Figure()
    traces_per_task = 4  # boundary, volume, infeasible scatter, feasible scatter
    payloads = []

    for ti, name in enumerate(tasks):
        v = task_volume(name, task_df_dict)
        if v is None:
            continue
        payloads.append((name, v))
        visible = ti == 0

        field = v["field"]
        feas_field = field >= 0
        # Decision boundary: build a clipped {-1, 0} field so the isosurface
        # placement isn't pulled around by varying burr magnitudes inside the
        # feasible region.
        boundary_field = np.where(feas_field, 0.0, INFEASIBLE_VALUE)

        fig.add_trace(
            go.Isosurface(
                x=v["F"].ravel(),
                y=v["G"].ravel(),
                z=v["P"].ravel(),
                value=boundary_field.ravel(),
                isomin=-0.5,
                isomax=-0.5,
                surface_count=1,
                colorscale=[[0, "rgba(120,120,120,0.4)"], [1, "rgba(120,120,120,0.4)"]],
                showscale=False,
                opacity=0.35,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="feasibility boundary",
                visible=visible,
                hoverinfo="skip",
            )
        )

        # Burr volume inside the feasible region. plotly Volume can't take NaN,
        # so push infeasible cells far below the colormap range.
        if feas_field.any():
            vmin = float(field[feas_field].min())
            vmax = float(field[feas_field].max())
        else:
            vmin, vmax = 0.0, 1.0
        burr_for_vol = np.where(feas_field, field, vmin - 1e9)

        fig.add_trace(
            go.Volume(
                x=v["F"].ravel(),
                y=v["G"].ravel(),
                z=v["P"].ravel(),
                value=burr_for_vol.ravel(),
                isomin=vmin,
                isomax=vmax,
                opacity=0.18,
                surface_count=18,
                colorscale="Viridis",
                colorbar=dict(title="predicted burr", x=1.02, len=0.6),
                name="predicted burr",
                visible=visible,
                hoverinfo="skip",
            )
        )

        infeas = ~v["feas_actual"]
        fig.add_trace(
            go.Scatter3d(
                x=v["fr"][infeas],
                y=v["gp"][infeas],
                z=v["fp"][infeas],
                mode="markers",
                marker=dict(size=4, color="crimson", symbol="x", line=dict(width=0)),
                name="infeasible (measured)",
                visible=visible,
                hovertemplate=(
                    "feedrate=%{x:.2f}<br>gas=%{y:.2f}<br>focus=%{z:.2f}"
                    "<extra>infeasible</extra>"
                ),
            )
        )

        feas = v["feas_actual"]
        burr_a = v["burr_actual"][feas]
        fig.add_trace(
            go.Scatter3d(
                x=v["fr"][feas],
                y=v["gp"][feas],
                z=v["fp"][feas],
                mode="markers",
                marker=dict(
                    size=5,
                    color=burr_a,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    line=dict(width=1, color="black"),
                    showscale=False,
                ),
                name="feasible (measured)",
                visible=visible,
                customdata=burr_a,
                hovertemplate=(
                    "feedrate=%{x:.2f}<br>gas=%{y:.2f}<br>focus=%{z:.2f}"
                    "<br>burr=%{customdata:.1f}<extra>feasible</extra>"
                ),
            )
        )

    n_traces = len(fig.data)
    buttons = []
    for idx, (name, v) in enumerate(payloads):
        vis = [False] * n_traces
        for k in range(traces_per_task):
            vis[idx * traces_per_task + k] = True
        feas_share = float(v["feas_actual"].mean())
        buttons.append(
            dict(
                label=f"{name[:46]}  (n={len(v['fr'])}, feas={feas_share:.0%})",
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title.text": f"Task simulator — {name}<br>"
                        f"<sub>n={len(v['fr'])} measurements, "
                        f"feasible share={feas_share:.0%}, grid={GRID_N}^3</sub>"
                    },
                ],
            )
        )

    init_name, init_v = payloads[0]
    init_share = float(init_v["feas_actual"].mean())
    fig.update_layout(
        title=dict(
            text=(
                f"Task simulator — {init_name}<br>"
                f"<sub>n={len(init_v['fr'])} measurements, "
                f"feasible share={init_share:.0%}, grid={GRID_N}^3</sub>"
            )
        ),
        scene=dict(
            xaxis_title="feedrate [m/min]",
            yaxis_title="gas pressure [bar]",
            zaxis_title="focal position [mm]",
            aspectmode="cube",
            bgcolor="rgb(245,245,250)",
        ),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgb(180,180,180)",
                font=dict(size=11),
            )
        ],
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.7)"),
        height=820,
        annotations=[
            dict(
                text=(
                    "<b>How to read it</b>: grey shell = feasibility boundary "
                    "(simulator output crosses 0).<br>"
                    "Volume shading inside the shell = predicted burr (Viridis: "
                    "dark = low, yellow = high).<br>"
                    "× = infeasible measurements, dots = feasible measurements "
                    "(coloured by actual burr).<br>"
                    "Drag to rotate, scroll to zoom, double-click to reset, "
                    "use the dropdown to switch task."
                ),
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=11, color="rgb(80,80,80)"),
            )
        ],
    )

    out = "simulator_3d.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
