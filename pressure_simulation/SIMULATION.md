# Pressure Chamber Simulation

This document describes the physics, numerics, and code layout of the
chamber simulator used for PID tuning, Bayesian optimization, and transfer
learning experiments.

## 1. Overview

A vacuum chamber is modelled as a single lumped volume with:

- **Constant gas inflow** at mass-flow rate $q_m$ (g/s).
- **Variable pumping** controlled by a butterfly-style valve whose opening
  angle $\alpha \in [0, 100]\%$ governs the conductance.
- **Calibrated pump curves** $Q(\alpha)$ and $\sigma(\alpha)$ obtained from
  bench measurements stored in `data/TV_Pressure_Record.xlsx`.

The simulator exposes a Gym-like environment (`ChamberEnv`) so that any
controller (PID, BO acquisition, RL agent) can drive it through a single
`env.step(action)` call.

## 2. Governing equations

### 2.1 Pressure ODE

Conservation of mass for an ideal gas in a fixed-volume chamber gives

$$
\frac{dp}{dt} \;=\; \frac{1}{V}\left[\frac{R\,T}{M}\,q_m \;-\; Q(\alpha)\,\big(p \;-\; \sigma(\alpha)\sqrt{q_m}\big)\right]
$$

where

| symbol | meaning | default |
|---|---|---|
| $p$ | chamber pressure | Torr |
| $V$ | chamber volume | 34.0 L |
| $T$ | gas temperature | 293.15 K |
| $M$ | molecular weight | 28.0 g/mol (N$_2$) |
| $R$ | gas constant in chosen units | 62.355 Torr·L/(mol·K) |
| $q_m$ | mass inflow rate | 0.25 g/s |
| $\alpha$ | valve opening | $[0, 100]\%$ |
| $Q(\alpha)$ | pump volumetric conductance | calibrated spline |
| $\sigma(\alpha)$ | back-pressure offset coefficient | calibrated spline |

The first bracketed term, $R T q_m / M$, is the gas inflow expressed as a
pressure-volume rate (Torr·L/s); dividing by $V$ converts it to a pressure
rise rate (Torr/s).

The second term, $Q(\alpha)(p - \sigma(\alpha)\sqrt{q_m})$, is the
effective pump throughput. It vanishes at the equilibrium pressure
$\sigma(\alpha)\sqrt{q_m}$ (a valve-dependent floor that depends on the
inflow rate) and grows linearly with $p$ above that floor.

### 2.2 Steady state

Setting $dp/dt = 0$,

$$
p^\star(\alpha, q_m) \;=\; \sigma(\alpha)\sqrt{q_m} \;+\; \frac{R\,T\,q_m}{M\,Q(\alpha)}
$$

so for each $(\alpha, q_m)$ there is a unique steady-state pressure.
`ChamberEnv._estimate_alpha` uses this relation in reverse to find the
$\alpha$ whose $p^\star$ matches a target pressure.

### 2.3 Valve actuator (rate-limited)

The commanded valve angle $\alpha_{\rm cmd}$ is rate-limited at $v_{\rm
speed}$ (default 1000 %/s):

$$
\alpha(t+\Delta t) \;=\; \alpha(t) \;+\; \mathrm{sign}\!\left(\alpha_{\rm cmd}-\alpha(t)\right)\,\min\!\left(v_{\rm speed}\,\Delta t,\;|\alpha_{\rm cmd}-\alpha(t)|\right),
\quad \alpha\in[0,100]
$$

At the default slew rate the valve effectively reaches its setpoint within
one integration sub-step, so the dynamics are dominated by the chamber
ODE; lowering `v_speed` introduces a noticeable actuator lag.

## 3. Calibration: $Q(\alpha)$ and $\sigma(\alpha)$

The pump curves are constructed in `Alpha2Conductance.alpha_to_conductance`
from steady-state pressure measurements at multiple inflow rates
($q_m \in \{0.03125, \dots, 0.3125\}$ g/s, derived from the SLM grid in
`_Q_REFERENCE`).

For each calibration $\alpha$, a 2×2 linear system is solved using two
reference inflows $q_a, q_b$:

$$
\begin{pmatrix} p^\star(\alpha,q_a) & -\sqrt{q_a} \\ p^\star(\alpha,q_b) & -\sqrt{q_b} \end{pmatrix}
\begin{pmatrix} Q(\alpha) \\ Q(\alpha)\,\sigma(\alpha) \end{pmatrix}
\;=\;
\begin{pmatrix} \frac{RT}{M}\,q_a \\ \frac{RT}{M}\,q_b \end{pmatrix}
$$

producing $Q(\alpha)$ and $Q(\alpha)\sigma(\alpha)$ pointwise; their
ratio gives $\sigma(\alpha)$. Cubic splines through these point values
yield the smooth $Q(\alpha)$, $\sigma(\alpha)$ used at runtime.
A Gaussian filter (window 10) smooths the calibration curves before
spline fitting to suppress measurement noise.

The `FlowSplines` dataclass returned by `alpha_to_conductance` exposes:

| spline | meaning |
|---|---|
| `alpha_to_pressure` | $p^\star(\alpha)$ for the calibration $q_m$ |
| `alpha_to_flow` | $Q(\alpha)$ |
| `alpha_to_sigma` | $\sigma(\alpha)$ |
| `flow_to_alpha` | $\alpha(Q)$ inverse |
| `*_derivative` | first/second derivatives w.r.t. $\alpha$ |

## 4. Numerical integration

Within one **control interval** $\Delta t_{\rm ctrl}$ (also called
`sample_rate`, default 1 ms), the chamber ODE is advanced with forward
Euler at a smaller **integration step** $\Delta t_{\rm int}$
(`time_step`, default 0.1 ms):

$$
N_{\rm sub} \;=\; \left\lceil \frac{\Delta t_{\rm ctrl}}{\Delta t_{\rm int}} \right\rceil,\quad
\delta t \;=\; \frac{\Delta t_{\rm ctrl}}{N_{\rm sub}}
$$

For each sub-step:

1. Slew $\alpha$ toward the commanded value at rate $v_{\rm speed}$.
2. Evaluate $Q(\alpha)$, $\sigma(\alpha)$.
3. Update pressure: $p \leftarrow \max(p + \dot p \,\delta t, \;0)$.
4. Advance time: $t \leftarrow t + \delta t$.

The outer loop is the control interval; the inner loop is the integration
step. The `max(\cdot, 0)` clamps pressure away from negative values, which
can otherwise occur transiently near the back-pressure floor.

## 5. Environment API (`ChamberEnv`)

```python
env = ChamberEnv({
    "p_goal": 5.0,           # target pressure (Torr)
    "sample_rate": 0.05,     # control interval (s)
    "time_step": 0.005,      # integration step  (s)
    "max_steps": 200,        # episode horizon (in control steps)
    # optional plant overrides:
    "volume_chamb": 34.0,    # chamber volume (L)
    "qm": 0.25,              # mass inflow (g/s)
    "v_speed": 1000.0,       # valve slew rate (%/s)
})
obs = env.reset(initial_pressure=10.0, goal_pressure=5.0)
# obs = [p, alpha, p - p_goal]

obs, reward, done, info = env.step(action, duration=0.05)
# action: target valve angle in [0, 100]
# reward: exp(-0.5 * (p - p_goal)^2)
```

Reward and the `done` flag are present so the env can be plugged into RL
libraries; they are not used by the BO/PID experiments in this repo.

## 6. PID controller (`pid.py`)

The `PID` class computes

$$
u(t) \;=\; K_p\,e(t) \;+\; K_i \int_0^{t} e(\tau)\,d\tau \;+\; K_d\,\frac{d e_f(t)}{dt} \;+\; u_{\rm ff}(t)
$$

with a first-order low-pass filter on the derivative term to attenuate
measurement noise and a feed-forward term $u_{\rm ff}$ for setpoint changes:

$$
e_f(t) \;=\; \alpha_d\,e_f(t-\Delta t) \;+\; (1-\alpha_d)\,e(t),\quad
\alpha_d \;=\; \frac{\tau}{\tau+\Delta t},\quad
\tau \;=\; \mathrm{tau\_factor}\,\Delta t
$$

The output is clamped to a configured range $u\in[u_{\rm lo}, u_{\rm hi}]$
(default $[0,1]$) and `transform_control_variable` maps it to a valve
angle in $[0, 100]$. The integration is trapezoidal:
$\int e\,dt \approx \frac{1}{2}(e_n + e_{n-1})\,\Delta t$.

### 6.1 Anti-windup

Two modes are provided:

- **Clamping**: when the unsaturated output is outside the actuator range
  *and* the new error would push it further into saturation, the integral
  is held constant for that step.
- **Back-calculation**: subtract $k_{aw}(u_{\rm sat}-u_{\rm unsat})$ from
  the integral every step. With $k_{aw}=1/(K_i\Delta t)$ this is
  *dead-beat* anti-windup.

### 6.2 Sign convention (important)

The simulator/code passes `error = measurement - target` (note: not the
textbook setpoint − measurement). With $K_p>0$, this means: when pressure
is *above* target, $u>0$, opening the valve more, increasing pumping,
lowering pressure. Sign-consistent with the chamber physics.

## 7. Default system parameters

Loaded from `data/tv_monitor_demo.csm` via `global_settings.SimulationGlobals`:

| symbol | parameter | default | unit |
|---|---|---|---|
| $V$ | chamber volume | 34.0 | L |
| $T$ | chamber temperature | 293.15 | K |
| $M$ | molecular weight (N$_2$) | 28.0 | g/mol |
| $\rho_{\rm pump}$ | pump-side gas density | 1.25 | g/L |
| $q_m$ | mass inflow (derived: $\rho \cdot \mathrm{SLM}/60$) | 0.25 | g/s |
| $R$ | gas constant | 62.355 | Torr·L/(mol·K) |
| $p_0$ | initial pressure | 5.0 | Torr |
| $\alpha_0$ | initial valve angle | 0.0 | % |
| $v_{\rm speed}$ | valve slew rate | 1000 | %/s |
| $\Delta t_{\rm ctrl}$ | control interval (`sample_rate`) | 1e-3 | s |
| $\Delta t_{\rm int}$ | integration step (`time_step`) | 1e-4 | s |

## 8. File map

| file | purpose |
|---|---|
| `ChamberEnv.py` | Gym-like environment: `reset`, `step`, ODE integration |
| `Alpha2Conductance.py` | Calibration loader; builds `FlowSplines` |
| `global_settings.py` | `SimulationGlobals` dataclass; CSM/JSON loader |
| `pid.py` | `PID`, `DirectionalPID`, `FeedForwardController`, anti-windup |
| `pressure_controller.py` | High-level controller wrapper |
| `reference_curve.py` | Time-series setpoint schedules |
| `simulation.py`, `SystemSimulate.py` | Scripted scenario helpers |
| `StatusCalculate.py`, `StepResponse.py` | Settling-time / overshoot metrics |
| `gui.py` | Tkinter GUI for interactive runs |
| `data/TV_Pressure_Record.xlsx` | bench calibration of $p^\star(\alpha, q_m)$ |
| `data/tv_monitor_demo.csm` | default plant + control settings |
| `optimize_pid_bo.py` | single-target Bayesian optimization of PID gains |
| `tl_pid_bayopt.py` | 5-scheme TL-BO sweep across 10 plants |
| `pid_convergence_plot.py` | convergence plots into experiment folders |

## 9. Quick examples

### 9.1 Single open-loop step (constant valve)

```python
from ChamberEnv import ChamberEnv

env = ChamberEnv({"p_goal": 5.0, "sample_rate": 0.05, "time_step": 0.005})
env.reset(initial_pressure=10.0)
for _ in range(100):
    obs, reward, done, info = env.step(action=70.0, duration=0.05)
    print(obs[0])  # current pressure
```

### 9.2 Closed-loop step with PID

```python
from ChamberEnv import ChamberEnv
from pid import PID

env = ChamberEnv({"p_goal": 5.0, "sample_rate": 0.05, "time_step": 0.005})
env.reset(initial_pressure=10.0)
pid = PID(dt=0.05, kp=0.007, ki=0.122, kd=0.015,
          u_bounds=(0.0, 1.0),
          use_antiwindup=True, clamping_antiwindup=True)
for _ in range(100):
    error = env.p - 5.0           # codebase convention: measurement - target
    u, _ = pid.update(error, env.p, desired_value=5.0)
    action = pid.transform_control_variable(u)   # [0,1] -> [0,100]
    env.step(action, duration=0.05)
```

### 9.3 Vary the plant (different chamber, different inflow)

```python
# Twice the volume, half the mass inflow
env = ChamberEnv({
    "p_goal": 5.0,
    "sample_rate": 0.05, "time_step": 0.005,
    "volume_chamb": 68.0,
    "qm": 0.125,
})
```

This is exactly the lever the transfer-learning experiments
(`tl_pid_bayopt.py`) use to construct families of related plants.

## 10. Extending the model

Things the current model does *not* capture that you may want to add:

- **Temperature dynamics**: $T$ is treated as a constant. Adding an energy
  balance would let the simulator track adiabatic compression / expansion
  effects in the chamber.
- **Non-ideal gas**: the equation of state is ideal-gas (PV = nRT in the
  derivation of $RT/M\cdot q_m$). For high pressures or condensable
  vapours a real-gas correction is needed.
- **Pipe transport delay**: pressure changes propagate instantaneously
  from the inflow point. A finite transport delay would matter at very
  fast control rates.
- **Sensor noise**: the env returns the true pressure. A Gaussian-noise
  wrapper around `obs[0]` would let you study controller robustness.

All of these would slot into `ChamberEnv.step` without touching the rest
of the codebase.
