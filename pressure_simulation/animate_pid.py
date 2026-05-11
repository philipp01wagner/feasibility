"""Visual animation of the pressure-chamber simulation under PID control.

Renders a left-hand schematic of the chamber (gas inflow, pressure gauge,
butterfly valve at the angle the controller commands, pump exhaust) alongside
live time-series of pressure, valve angle, and the P/I/D contributions.

Usage
-----
    python animate_pid.py                 # save MP4 (falls back to GIF)
    python animate_pid.py --show          # display interactively
    python animate_pid.py --gif           # save GIF (no ffmpeg required)
    python animate_pid.py --out foo.mp4   # custom output path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D

from ChamberEnv import ChamberEnv
from pid import PID


def run_simulation(
    *,
    initial_pressure: float = 12.0,
    setpoint_schedule: list[tuple[float, float]] | None = None,
    sample_rate: float = 0.05,
    time_step: float = 0.005,
    kp: float = 0.007,
    ki: float = 0.122,
    kd: float = 0.015,
):
    """Drive ChamberEnv with a PID across a multi-step setpoint schedule.

    Parameters
    ----------
    setpoint_schedule
        list of (duration_s, setpoint) pairs. The simulation runs each
        segment in order, with the PID state carrying across segments.
    """
    if setpoint_schedule is None:
        setpoint_schedule = [(4.0, 5.0), (4.0, 3.0), (4.0, 7.0)]

    env = ChamberEnv({
        "p_goal": setpoint_schedule[0][1],
        "sample_rate": sample_rate,
        "time_step": time_step,
        "v_speed": 1000.0,
    })
    env.reset(initial_pressure=initial_pressure, goal_pressure=setpoint_schedule[0][1])

    pid = PID(
        dt=sample_rate,
        kp=kp, ki=ki, kd=kd,
        u_bounds=(0.0, 1.0),
        use_antiwindup=True,
        clamping_antiwindup=True,
    )

    times: list[float] = []
    pressures: list[float] = []
    targets: list[float] = []
    alphas: list[float] = []
    actions: list[float] = []
    u_p_hist: list[float] = []
    u_i_hist: list[float] = []
    u_d_hist: list[float] = []
    u_hist: list[float] = []

    t = 0.0
    p = float(env.p)
    alpha = float(env.alpha)
    error = p - setpoint_schedule[0][1]

    for duration, target in setpoint_schedule:
        n_steps = int(round(duration / sample_rate))
        env.p_goal = target
        for _ in range(n_steps):
            u, info = pid.update(error, p, desired_value=target)
            action = pid.transform_control_variable(u)
            obs, _, _, _ = env.step(action, duration=sample_rate)

            times.append(t)
            pressures.append(p)
            targets.append(target)
            alphas.append(alpha)
            actions.append(action)
            u_p_hist.append(info["u_p"])
            u_i_hist.append(info["u_i"])
            u_d_hist.append(info["u_d"])
            u_hist.append(info["u"])

            p = float(obs[0])
            alpha = float(obs[1])
            error = float(obs[2])
            t += sample_rate

    return {
        "t": np.asarray(times),
        "p": np.asarray(pressures),
        "target": np.asarray(targets),
        "alpha": np.asarray(alphas),
        "action": np.asarray(actions),
        "u_p": np.asarray(u_p_hist),
        "u_i": np.asarray(u_i_hist),
        "u_d": np.asarray(u_d_hist),
        "u": np.asarray(u_hist),
    }


def _draw_chamber_schematic(ax):
    """Draw the static chamber/pump/inflow background. Returns dynamic artists."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Gas inflow arrow at the top
    ax.add_patch(FancyArrowPatch(
        (-1.0, 2.2), (-1.0, 1.45),
        arrowstyle="-|>", mutation_scale=18,
        color="#2a7fbe", lw=2,
    ))
    ax.text(-1.0, 2.32, "gas in (q$_m$)", ha="center", fontsize=9, color="#2a7fbe")

    # Chamber body (filled translucent rectangle, color shows pressure)
    chamber = FancyBboxPatch(
        (-1.05, -0.25), 2.1, 1.7,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2, edgecolor="#333",
        facecolor="#cfe2ff",
    )
    ax.add_patch(chamber)
    ax.text(0, 1.55, "chamber (V = 34 L)", ha="center", fontsize=9, color="#333")

    # Inflow pipe cap
    ax.plot([-1.0, -1.0], [1.45, 1.45 - 0.05], color="#333", lw=2)

    # Pressure gauge: circle on the chamber's right wall
    gauge_center = (0.55, 0.95)
    ax.add_patch(patches.Circle(
        gauge_center, 0.32, facecolor="#fffaf0",
        edgecolor="#333", lw=1.5, zorder=3,
    ))
    # gauge tick marks (semicircle from 180deg left to 0deg right, sweeping over the top)
    for frac in np.linspace(0, 1, 7):
        ang = np.pi - frac * np.pi
        x0 = gauge_center[0] + 0.28 * np.cos(ang)
        y0 = gauge_center[1] + 0.28 * np.sin(ang)
        x1 = gauge_center[0] + 0.32 * np.cos(ang)
        y1 = gauge_center[1] + 0.32 * np.sin(ang)
        ax.plot([x0, x1], [y0, y1], color="#333", lw=1, zorder=4)
    ax.text(gauge_center[0], gauge_center[1] - 0.45, "pressure", ha="center", fontsize=8)

    # Setpoint marker (red triangle) and needle (blue) — both dynamic
    setpoint_marker, = ax.plot(
        [], [], marker=(3, 0, 0), markersize=10,
        markerfacecolor="#cc2233", markeredgecolor="#990000", zorder=5,
    )
    needle, = ax.plot([], [], color="#1f3b73", lw=2.5, solid_capstyle="round", zorder=5)

    # Outlet pipe between chamber and valve
    ax.add_patch(patches.Rectangle(
        (-0.18, -0.7), 0.36, 0.5, facecolor="#dcdcdc", edgecolor="#333", lw=1.5,
    ))

    # Valve housing (circle) and rotating disc inside it
    valve_center = (0.0, -0.95)
    ax.add_patch(patches.Circle(
        valve_center, 0.32, facecolor="#f0f0f0",
        edgecolor="#333", lw=1.5, zorder=2,
    ))
    # Butterfly-valve plate — a thin rectangle we rotate every frame.
    # Initial vertices describe a horizontal plate (alpha=0, fully closed).
    valve_disc = patches.Polygon(
        [(0, 0), (0, 0), (0, 0), (0, 0)],
        closed=True, facecolor="#2a3f5f", edgecolor="#0d1c33",
        lw=1.2, zorder=6,
    )
    ax.add_patch(valve_disc)
    # Central axle (so the plate visibly pivots around a fixed point).
    ax.add_patch(patches.Circle(
        valve_center, 0.04, facecolor="#0d1c33", edgecolor="none", zorder=7,
    ))
    ax.text(valve_center[0] + 0.55, valve_center[1], "valve", fontsize=9, color="#333")

    # Pump symbol
    ax.add_patch(patches.Rectangle(
        (-0.18, -1.45), 0.36, 0.25, facecolor="#dcdcdc", edgecolor="#333", lw=1.5,
    ))
    ax.add_patch(FancyBboxPatch(
        (-0.5, -2.05), 1.0, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#9bc1e0", edgecolor="#333", lw=1.5,
    ))
    ax.text(0, -1.78, "pump", ha="center", va="center", fontsize=10, color="#0d2c4e")
    ax.add_patch(FancyArrowPatch(
        (0, -2.05), (0, -2.32),
        arrowstyle="-|>", mutation_scale=14, color="#1f3b73", lw=1.5,
    ))

    # Live numeric readouts
    readout = ax.text(
        -1.45, -2.2, "", fontsize=10, family="monospace",
        color="#222", va="bottom",
    )

    return {
        "chamber": chamber,
        "needle": needle,
        "setpoint_marker": setpoint_marker,
        "gauge_center": gauge_center,
        "valve_center": valve_center,
        "valve_disc": valve_disc,
        "readout": readout,
    }


def _pressure_to_color(p, p_min=0.0, p_max=12.0):
    """Map pressure to a chamber fill color (blue=low, red=high)."""
    t = float(np.clip((p - p_min) / max(p_max - p_min, 1e-9), 0.0, 1.0))
    # interpolate from light blue (#cfe2ff) -> warm red (#f4a8a0)
    c0 = np.array([0.81, 0.886, 1.0])
    c1 = np.array([0.957, 0.659, 0.627])
    rgb = (1 - t) * c0 + t * c1
    return tuple(rgb)


class GasParticles:
    """Visual gas-particle cloud whose density scales with chamber pressure.

    Particles undergo Brownian-ish motion with a downward drift proportional
    to valve opening (a stand-in for pump suction). New particles spawn near
    the gas-inflow point at the chamber's top-left so the user sees fresh gas
    streaming in; the count is capped to `round(p * particles_per_torr)`.
    """

    def __init__(
        self,
        ax,
        *,
        bounds_x: tuple[float, float] = (-1.0, 1.0),
        bounds_y: tuple[float, float] = (-0.20, 1.40),
        inflow_xy: tuple[float, float] = (-0.95, 1.35),
        n_max: int = 320,
        particles_per_torr: float = 22.0,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.bx = bounds_x
        self.by = bounds_y
        self.inflow_xy = inflow_xy
        self.n_max = n_max
        self.scale = particles_per_torr

        # Pre-allocated particle pool. n_active <= n_max defines the visible subset.
        self.pos = np.column_stack([
            self.rng.uniform(bounds_x[0], bounds_x[1], n_max),
            self.rng.uniform(bounds_y[0], bounds_y[1], n_max),
        ])
        self.vel = self.rng.normal(0.0, 0.012, (n_max, 2))
        self.n_active = 0

        self.scatter = ax.scatter(
            np.empty(0), np.empty(0),
            s=9, c="#0d1c33", alpha=0.5,
            edgecolors="none", zorder=2,
        )

    def step(self, pressure: float, alpha_pct: float):
        target_n = int(np.clip(round(pressure * self.scale), 0, self.n_max))

        if target_n > self.n_active:
            # Spawn the new particles near the gas-inflow point.
            n_new = target_n - self.n_active
            idx = slice(self.n_active, target_n)
            cx, cy = self.inflow_xy
            self.pos[idx, 0] = cx + 0.10 * self.rng.standard_normal(n_new)
            self.pos[idx, 1] = cy - 0.05 * self.rng.random(n_new)
            self.vel[idx] = self.rng.normal(0.0, 0.015, (n_new, 2))
        self.n_active = target_n

        if target_n == 0:
            self.scatter.set_offsets(np.zeros((0, 2)))
            return

        act = slice(0, target_n)
        # Brownian kicks + light damping
        self.vel[act] += 0.005 * self.rng.standard_normal((target_n, 2))
        self.vel[act] *= 0.90
        # Pump suction: downward bias proportional to valve opening
        self.vel[act, 1] -= 0.0012 * (alpha_pct / 100.0)
        self.pos[act] += self.vel[act]

        # Reflect off chamber walls (separate x, y for clarity)
        for axis, (lo, hi) in enumerate((self.bx, self.by)):
            coords = self.pos[act, axis]
            below = coords < lo
            above = coords > hi
            coords = np.where(below, lo + (lo - coords), coords)
            coords = np.where(above, hi - (coords - hi), coords)
            self.pos[act, axis] = coords
            self.vel[act, axis] = np.where(
                below | above, -self.vel[act, axis] * 0.7, self.vel[act, axis]
            )

        self.scatter.set_offsets(self.pos[act])


def build_animation(history, *, fps: int = 25):
    """Compose a 2-column figure: schematic on the left, live traces on the right."""
    t = history["t"]
    n_frames = len(t)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.5], hspace=0.45, wspace=0.28)
    ax_schema = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_a = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_u = fig.add_subplot(gs[2, 1], sharex=ax_p)

    fig.suptitle("Pressure Chamber under PID Control", fontsize=14, weight="bold")

    artists = _draw_chamber_schematic(ax_schema)
    particles = GasParticles(ax_schema)

    # ---- Pressure plot ----
    p_max_plot = max(history["p"].max(), history["target"].max()) * 1.15
    p_min_plot = max(0.0, history["p"].min() - 0.5)
    ax_p.set_xlim(t[0], t[-1])
    ax_p.set_ylim(p_min_plot, p_max_plot)
    ax_p.set_ylabel("pressure (Torr)")
    ax_p.grid(True, alpha=0.3)
    target_line, = ax_p.plot([], [], color="#cc2233", lw=1.6, ls="--", label="setpoint")
    pressure_line, = ax_p.plot([], [], color="#1f3b73", lw=2.0, label="measured")
    pressure_dot, = ax_p.plot([], [], "o", color="#1f3b73", markersize=6)
    ax_p.legend(loc="upper right", fontsize=9)

    # ---- Valve angle plot ----
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_ylim(-2, 102)
    ax_a.set_ylabel("valve $\\alpha$ (%)")
    ax_a.grid(True, alpha=0.3)
    alpha_line, = ax_a.plot([], [], color="#2a7fbe", lw=1.8, label="actual")
    action_line, = ax_a.plot([], [], color="#888", lw=1.0, ls=":", label="commanded")
    ax_a.legend(loc="upper right", fontsize=9)

    # ---- PID component plot ----
    ax_u.set_xlim(t[0], t[-1])
    u_all = np.concatenate([history["u_p"], history["u_i"], history["u_d"], history["u"]])
    pad = 0.1 * (u_all.max() - u_all.min() + 1e-6)
    ax_u.set_ylim(u_all.min() - pad, u_all.max() + pad)
    ax_u.axhline(0, color="#888", lw=0.6)
    ax_u.set_ylabel("PID terms")
    ax_u.set_xlabel("time (s)")
    ax_u.grid(True, alpha=0.3)
    up_line, = ax_u.plot([], [], color="#2a7fbe", lw=1.4, label="$K_p\\,e$")
    ui_line, = ax_u.plot([], [], color="#33a02c", lw=1.4, label="$K_i\\,\\int e$")
    ud_line, = ax_u.plot([], [], color="#cc2233", lw=1.4, label="$K_d\\,\\dot e$")
    u_line,  = ax_u.plot([], [], color="#222", lw=1.6, ls="--", label="$u$ (sat)")
    ax_u.legend(loc="upper right", fontsize=8, ncol=4)

    # ---- valve disc geometry helper ----
    valve_center = artists["valve_center"]
    disc_half_len = 0.28      # along the plate's long axis
    disc_half_thick = 0.045   # plate thickness

    def _valve_disc_xy(alpha_pct: float):
        """Return 4 polygon vertices for the butterfly plate at angle alpha.

        alpha=0  -> plate horizontal (closed, blocks vertical flow).
        alpha=100 -> plate vertical  (fully open, parallel to flow).
        """
        theta = np.deg2rad(alpha_pct * 0.9)  # 0..90 deg
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = valve_center
        L, W = disc_half_len, disc_half_thick
        # Four corners of a rectangle (L long, W thick), rotated by theta.
        corners = np.array([
            [+L, -W],
            [+L, +W],
            [-L, +W],
            [-L, -W],
        ])
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = corners @ rot.T
        rotated[:, 0] += cx
        rotated[:, 1] += cy
        return rotated

    def _gauge_needle_xy(p_val: float, p_min=0.0, p_max=12.0):
        # gauge sweeps from 180deg (left, low) to 0deg (right, high) over the top
        frac = float(np.clip((p_val - p_min) / max(p_max - p_min, 1e-9), 0.0, 1.0))
        ang = np.pi - frac * np.pi
        cx, cy = artists["gauge_center"]
        return [cx, cx + 0.26 * np.cos(ang)], [cy, cy + 0.26 * np.sin(ang)]

    def _gauge_setpoint_xy(p_val: float, p_min=0.0, p_max=12.0):
        frac = float(np.clip((p_val - p_min) / max(p_max - p_min, 1e-9), 0.0, 1.0))
        ang = np.pi - frac * np.pi
        cx, cy = artists["gauge_center"]
        return [cx + 0.30 * np.cos(ang)], [cy + 0.30 * np.sin(ang)]

    p_max_for_gauge = max(history["target"].max(), history["p"].max()) * 1.05

    def init():
        for line in (target_line, pressure_line, pressure_dot, alpha_line, action_line,
                     up_line, ui_line, ud_line, u_line):
            line.set_data([], [])
        artists["needle"].set_data([], [])
        artists["setpoint_marker"].set_data([], [])
        artists["valve_disc"].set_xy(_valve_disc_xy(0.0))
        artists["readout"].set_text("")
        return [
            target_line, pressure_line, pressure_dot, alpha_line, action_line,
            up_line, ui_line, ud_line, u_line,
            artists["needle"], artists["setpoint_marker"],
            artists["valve_disc"], artists["readout"], artists["chamber"],
        ]

    def update(frame: int):
        i = frame + 1  # include current sample
        target_line.set_data(t[:i], history["target"][:i])
        pressure_line.set_data(t[:i], history["p"][:i])
        pressure_dot.set_data([t[i - 1]], [history["p"][i - 1]])
        alpha_line.set_data(t[:i], history["alpha"][:i])
        action_line.set_data(t[:i], history["action"][:i])
        up_line.set_data(t[:i], history["u_p"][:i])
        ui_line.set_data(t[:i], history["u_i"][:i])
        ud_line.set_data(t[:i], history["u_d"][:i])
        u_line.set_data(t[:i], history["u"][:i])

        p_now = history["p"][i - 1]
        a_now = history["alpha"][i - 1]
        tgt_now = history["target"][i - 1]
        e_now = p_now - tgt_now

        artists["chamber"].set_facecolor(_pressure_to_color(p_now, 0.0, p_max_for_gauge))
        artists["needle"].set_data(*_gauge_needle_xy(p_now, 0.0, p_max_for_gauge))
        artists["setpoint_marker"].set_data(*_gauge_setpoint_xy(tgt_now, 0.0, p_max_for_gauge))
        artists["valve_disc"].set_xy(_valve_disc_xy(a_now))
        particles.step(p_now, a_now)
        artists["readout"].set_text(
            f"t      = {t[i - 1]:6.2f} s\n"
            f"P      = {p_now:6.3f} Torr\n"
            f"target = {tgt_now:6.3f} Torr\n"
            f"error  = {e_now:+6.3f} Torr\n"
            f"alpha  = {a_now:6.2f} %"
        )

        return [
            target_line, pressure_line, pressure_dot, alpha_line, action_line,
            up_line, ui_line, ud_line, u_line,
            artists["needle"], artists["setpoint_marker"],
            artists["valve_disc"], artists["readout"], artists["chamber"],
            particles.scatter,
        ]

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=1000 / fps, blit=False, repeat=False,
    )
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display interactively")
    parser.add_argument("--gif", action="store_true", help="force GIF output")
    parser.add_argument("--out", type=str, default=None, help="output file path")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--kp", type=float, default=0.007)
    parser.add_argument("--ki", type=float, default=0.122)
    parser.add_argument("--kd", type=float, default=0.015)
    args = parser.parse_args()

    history = run_simulation(kp=args.kp, ki=args.ki, kd=args.kd)
    fig, anim = build_animation(history, fps=args.fps)

    if args.show:
        plt.show()
        return

    out = args.out
    use_gif = args.gif or (out is not None and out.lower().endswith(".gif"))
    if out is None:
        out = "pressure_pid_animation.gif" if use_gif else "pressure_pid_animation.mp4"

    if use_gif:
        anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    else:
        try:
            anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
        except (FileNotFoundError, RuntimeError):
            print("ffmpeg not available; falling back to GIF.")
            out = str(Path(out).with_suffix(".gif"))
            anim.save(out, writer=animation.PillowWriter(fps=args.fps))

    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()
