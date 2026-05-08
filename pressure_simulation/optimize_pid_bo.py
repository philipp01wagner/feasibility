"""Bayesian optimization of PID gains (kp, ki, kd) for the chamber simulator.

Cost = ITAE (integral of t*|error|): the standard PID criterion that rewards
both fast convergence and low residual error. Uses a Matern-2.5 GP on
log1p(cost) plus Expected Improvement.

Run with the weldopt env (Python 3.10+):
    "C:/Users/phw/Anaconda3/envs/weldopt/python.exe" optimize_pid_bo.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ChamberEnv import ChamberEnv
from pid import PID

# ---------------------------- config --------------------------------------
INITIAL_PRESSURE = 10.0   # Torr
TARGET_PRESSURE = 5.0     # Torr (step-down scenario)
SIM_DURATION = 5.0        # seconds simulated per evaluation
SAMPLE_RATE = 0.05        # control interval (s)
TIME_STEP = 0.005         # internal integration step (s)

# Useful PID gains for this plant span several orders of magnitude, so we
# search log-uniformly for kp and ki (linear for kd). Internally the BO
# operates on (log10(kp), log10(ki), kd); evaluate_pid takes the original
# linear gains.
KP_BOUNDS = (1e-4, 5.0)
KI_BOUNDS = (1e-4, 5.0)
KD_BOUNDS = (1e-4, 0.5)

# All three gains are searched log-uniformly. The internal optimizer sees
# (log10 kp, log10 ki, log10 kd); evaluate_pid takes the linear gains.
BOUNDS = np.array([
    [np.log10(KP_BOUNDS[0]), np.log10(KP_BOUNDS[1])],
    [np.log10(KI_BOUNDS[0]), np.log10(KI_BOUNDS[1])],
    [np.log10(KD_BOUNDS[0]), np.log10(KD_BOUNDS[1])],
])
DIM = 3


def to_pid_gains(x_internal):
    """Map an internal point (log10 kp, log10 ki, log10 kd) -> linear gains."""
    return 10.0 ** np.atleast_1d(x_internal).astype(float)


SEED = 42
N_INIT = 20
N_ITER = 40
N_CANDIDATES = 2000


# ---------------------------- objective -----------------------------------
def evaluate_pid(kp: float, ki: float, kd: float) -> tuple[float, dict]:
    """Run one closed-loop step response and return (ITAE, history dict)."""
    env_config = {
        "p_goal": TARGET_PRESSURE,
        "sample_rate": SAMPLE_RATE,
        "time_step": TIME_STEP,
        "max_steps": int(SIM_DURATION / SAMPLE_RATE) + 5,
    }
    env = ChamberEnv(env_config)
    env.reset(initial_pressure=INITIAL_PRESSURE, goal_pressure=TARGET_PRESSURE)
    pid = PID(
        dt=SAMPLE_RATE,
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
        u_bounds=(0.0, 1.0),
        use_antiwindup=True,
        clamping_antiwindup=True,
    )

    times = [0.0]
    pressures = [float(env.p)]
    valves = [float(env.alpha)]

    n_steps = int(np.ceil(SIM_DURATION / SAMPLE_RATE))
    for _ in range(n_steps):
        measurement = float(env.p)
        # Existing codebase convention: error = measurement - target
        # (matches simulate_matlab in ChamberEnv.py, where observation[2] is
        # self.p - self.p_goal). Kp>0 then pushes the valve more open when
        # pressure is above target -> pumps out -> pressure falls.
        error = measurement - TARGET_PRESSURE
        u, _ = pid.update(error, measurement, desired_value=TARGET_PRESSURE)
        action = pid.transform_control_variable(u)
        obs, _, _, info = env.step(action, duration=SAMPLE_RATE)
        times.append(float(info["time"]))
        pressures.append(float(obs[0]))
        valves.append(float(obs[1]))

    times = np.array(times)
    pressures = np.array(pressures)
    valves = np.array(valves)
    abs_err = np.abs(pressures - TARGET_PRESSURE)

    # ITAE = integral of t*|e(t)| dt — fast convergence + low residual error
    itae = float(np.trapz(times * abs_err, times))
    history = dict(times=times, pressures=pressures, valves=valves, itae=itae)
    return itae, history


# ---------------------------- BO bits -------------------------------------
def expected_improvement(mean, std, y_best, xi=0.01):
    imp = y_best - mean - xi
    Z = imp / np.where(std > 0, std, 1e-9)
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return np.where(std > 0, np.maximum(ei, 0.0), 0.0)


def make_kernel():
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )


def optimize_pid():
    rng = np.random.default_rng(SEED)

    # Sobol sequence for low-discrepancy initial coverage of the log-uniform
    # box; much more uniform than i.i.d. uniform in 3D for small N.
    from scipy.stats.qmc import Sobol
    sobol = Sobol(d=DIM, seed=SEED)
    unit = sobol.random(n=N_INIT)
    X = BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])
    y = np.array([evaluate_pid(*to_pid_gains(x))[0] for x in X])
    print(f"[init] best cost = {y.min():.3f}  "
          f"(across {N_INIT} Sobol samples)")

    history_best = [float(y.min())]
    t_start = time.time()

    for it in range(N_ITER):
        scaler = StandardScaler().fit(X)
        y_log = np.log1p(y)
        gp = GaussianProcessRegressor(
            kernel=make_kernel(),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=SEED,
        )
        gp.fit(scaler.transform(X), y_log)

        X_cand = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_CANDIDATES, DIM))
        mean, std = gp.predict(scaler.transform(X_cand), return_std=True)
        ei = expected_improvement(mean, std, y_log.min())
        x_next = X_cand[np.argmax(ei)]

        gains = to_pid_gains(x_next)
        cost, _ = evaluate_pid(*gains)
        X = np.vstack([X, x_next])
        y = np.append(y, cost)
        history_best.append(float(y.min()))

        print(
            f"[iter {it + 1:>2}/{N_ITER}] "
            f"kp={gains[0]:8.4f} ki={gains[1]:8.4f} kd={gains[2]:5.3f}  "
            f"cost={cost:8.3f}  best={y.min():8.3f}"
        )

    elapsed = time.time() - t_start
    best_idx = int(np.argmin(y))
    best_gains = to_pid_gains(X[best_idx])
    print(f"\nBO finished in {elapsed:.1f}s")
    print(f"Best cost (ITAE): {y[best_idx]:.3f}")
    print(f"Best PID: kp={best_gains[0]:.4f}, "
          f"ki={best_gains[1]:.4f}, kd={best_gains[2]:.4f}")
    return X, y, history_best, best_gains


# ---------------------------- main + plot ---------------------------------
def main():
    X, y, history_best, best = optimize_pid()

    # Re-run the best params for plotting
    cost_best, hist = evaluate_pid(*best)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(np.arange(len(history_best)), history_best, "o-", color="#2E86AB")
    axes[0].set_xlabel("BO iteration (after init)")
    axes[0].set_ylabel("Best ITAE so far")
    axes[0].set_title("BO convergence")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist["times"], hist["pressures"], color="#E63946", label="pressure")
    axes[1].axhline(
        TARGET_PRESSURE, color="black", linestyle="--",
        label=f"target = {TARGET_PRESSURE} Torr",
    )
    axes[1].axhline(
        INITIAL_PRESSURE, color="gray", linestyle=":", alpha=0.5,
        label=f"initial = {INITIAL_PRESSURE} Torr",
    )
    axes[1].set_ylabel("Pressure (Torr)")
    axes[1].set_title(
        f"Step response of best PID — "
        f"kp={best[0]:.3f}, ki={best[1]:.3f}, kd={best[2]:.3f}, "
        f"ITAE={cost_best:.2f}"
    )
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    axes[2].plot(hist["times"], hist["valves"], color="#06A77D")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Valve angle (%)")
    axes[2].set_title("Actuator trajectory")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = "pid_bo.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
