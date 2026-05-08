from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from Alpha2Conductance import alpha_to_conductance
from global_settings import get_simulation_globals
from pid import AbstractPID
from reference_curve import ReferenceCurve


class ChamberEnv:
    """
    A Gym-like environment for the vacuum chamber simulation.

    The environment allows an agent to control the valve opening to reach a
    target pressure.

    Action: A single float value representing the target valve opening (0-100%).
    State (Observation): A numpy array containing
        [current_pressure, current_valve_angle, error].
    Reward: Calculated based on the proximity to the target pressure.
    """

    def __init__(self, env_config=None):
        """
        Initializes the environment.

        Args:
            env_config (dict, optional): A dictionary to configure the environment.
                Keys can include 'p_goal', 'v_speed', 'dt', 'max_steps'.
        """
        # print("Initializing Chamber Environment...")
        # Load system characteristics from the calibration file
        self.system_params, self.flow_splines = alpha_to_conductance()
        self.a2p_spline = self.flow_splines.alpha_to_pressure
        self.a2q_spline = self.flow_splines.alpha_to_flow
        self.a2sigma_spline = self.flow_splines.alpha_to_sigma
        self.q2a_spline = self.flow_splines.flow_to_alpha
        # Aliases retained for backwards compatibility with legacy helpers
        self.a2c_spline = self.a2q_spline

        # --- Environment Configuration ---
        globals_cfg = get_simulation_globals()
        defaults = globals_cfg.as_env_defaults()
        config = {**defaults, **(env_config or {})}
        self.p_goal = float(config.get("p_goal", defaults["p_goal"]))
        # Match the valve slew rate used by the legacy simulator (1000 %/s by default).
        self.v_speed = float(config.get("v_speed", defaults["v_speed"]))
        self._default_initial_pressure = float(
            (env_config or {}).get("initial_pressure", globals_cfg.initial_pressure)
        )
        user_alpha = (env_config or {}).get("position_valve")
        self._alpha_from_config = user_alpha is not None
        base_alpha = (
            float(user_alpha)
            if user_alpha is not None
            else float(defaults["position_valve"])
        )
        self._default_alpha = float(np.clip(base_alpha, 0.0, 100.0))

        # Integration time step used for the internal plant model (s)
        time_step = float(
            config.get(
                "time_step",
                config.get("integration_dt", config.get("dt", 0.02)),
            )
        )
        if time_step <= 0:
            raise ValueError("time_step must be greater than zero")
        self.time_step = time_step

        # Sample interval mirrors the legacy sample rate idea (s)
        sample_rate = float(
            config.get("sample_rate", config.get("control_dt", self.time_step))
        )
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        self.sample_rate = sample_rate

        # Legacy attribute names for backwards compatibility with older code.
        self.control_dt = self.sample_rate
        self.integration_dt = self.time_step

        # Keep the legacy name for backwards compatibility
        self.dt = self.time_step
        self.max_steps = int(
            config.get("max_steps", 500)
        )  # Max control steps per episode

        # --- System Physics Parameters (loaded) ---
        self.R = self.system_params["R"]
        self.M = float(config.get("molweight_chamb", self.system_params["M"]))
        self.V = float(config.get("volume_chamb", self.system_params["V"]))
        self.T = float(config.get("temperature_chamb", self.system_params["T"]))

        if "rho_pump" in config:
            inflow_slm = self.system_params.get("inflow_slm", 12.0)
            self.qm = float(config["rho_pump"]) * inflow_slm / 60.0
        else:
            self.qm = float(config.get("qm", self.system_params["qm"]))
        self.sqrt_qm = float(np.sqrt(self.qm))
        self._rhs_term = self.R * self.T / self.M * self.qm
        self._inv_volume = 1.0 / self.V

        # --- Episode State ---
        self.current_step = 0
        self.time = 0.0
        self.p = 0.0
        self.alpha = 0.0
        # print("Environment initialized.")

    def reset(self, initial_pressure=None, goal_pressure=None, initial_alpha=None):
        """
        Resets the environment to an initial state.

        Args:
            initial_pressure (float | None): Starting pressure. When ``None`` the
                shared global settings provide the value.
            goal_pressure (float, optional): A new target pressure for the episode.

        Returns:
            np.ndarray: The initial observation [pressure, alpha].
        """
        globals_cfg = get_simulation_globals()
        default_pressure = getattr(
            self,
            "_default_initial_pressure",
            globals_cfg.initial_pressure,
        )
        pressure = default_pressure if initial_pressure is None else initial_pressure

        self.current_step = 0
        self.time = 0.0
        self.p = float(pressure)
        if initial_alpha is not None:
            self.alpha = float(np.clip(initial_alpha, 0.0, 100.0))
        else:
            if getattr(self, "_alpha_from_config", False):
                raw_alpha = getattr(self, "_default_alpha", globals_cfg.position_valve)
                self.alpha = float(np.clip(raw_alpha, 0.0, 100.0))
            else:
                self.alpha = self._estimate_alpha(self.p)

        if goal_pressure is not None:
            self.p_goal = goal_pressure

        # Return the initial state as the first observation
        return np.array([self.p, self.alpha, self.p - self.p_goal])

    def step(self, action, *, duration: float | None = None):
        """
        Executes one time step in the environment.

        Args:
            action (float): The target valve angle (0-100) set by the agent.
            duration (float, optional): Control interval to integrate over.
                Defaults to ``self.sample_rate``.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): [new_pressure, new_alpha, error].
                - reward (float): The reward for the current step.
                - done (bool): Whether the episode has ended.
                - info (dict): A dictionary for diagnostic information.
        """
        target_alpha = float(np.clip(action, 0.0, 100.0))
        step_duration = self.sample_rate if duration is None else float(duration)
        if step_duration <= 0.0:
            raise ValueError("duration must be greater than zero")

        steps = max(1, int(np.ceil(step_duration / self.time_step)))
        sub_dt = step_duration / steps

        current_q = float(self.a2q_spline(self.alpha))
        current_sigma = float(self.a2sigma_spline(self.alpha))

        for _ in range(steps):
            max_angle_change = self.v_speed * sub_dt
            alpha_error = target_alpha - self.alpha
            if abs(alpha_error) <= max_angle_change:
                self.alpha = target_alpha
            else:
                self.alpha += np.sign(alpha_error) * max_angle_change
            self.alpha = float(np.clip(self.alpha, 0.0, 100.0))

            current_q = float(self.a2q_spline(self.alpha))
            current_sigma = float(self.a2sigma_spline(self.alpha))
            p_prev = self.p
            dp_dt = (
                self._rhs_term - current_q * (p_prev - current_sigma * self.sqrt_qm)
            ) * self._inv_volume
            self.p = max(p_prev + dp_dt * sub_dt, 0.0)
            self.time += sub_dt

        self.current_step += 1
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps

        observation = np.array([self.p, self.alpha, self.p - self.p_goal])
        info = {
            "time": self.time,
            "sample_interval": step_duration,
            "time_step": sub_dt,
            "control_interval": step_duration,
            "integration_dt": sub_dt,
            "integration_steps": steps,
            "flow": current_q,
            "sigma": current_sigma,
        }

        return observation, reward, done, info

    def _calculate_reward(self):
        """
        Calculates the reward based on the distance to the goal pressure.
        The reward is normalized between 0 and 1.
        """
        # Using a squared error, normalized by the initial error squared
        # to keep the reward scale consistent. Let's use a simpler form for now.
        error = self.p - self.p_goal
        # Use an exponential to give high reward near the goal
        return np.exp(-0.5 * (error**2))

    def _estimate_alpha(self, pressure: float) -> float:
        alpha_grid = np.linspace(2.0, 100.0, 500)
        q = self.a2q_spline(alpha_grid)
        sigma = self.a2sigma_spline(alpha_grid)
        target = self._rhs_term
        values = q * (pressure - sigma * self.sqrt_qm)
        idx = int(np.argmin(np.abs(values - target)))
        return float(np.clip(alpha_grid[idx], 0.0, 100.0))


def simulate_matlab(
    pid: AbstractPID, reference_curve: ReferenceCurve, **kwargs: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Placeholder for the C# simulation function.
    """
    globals_cfg = get_simulation_globals()
    env_config = dict(kwargs.get("env_config", {}))
    #
    # if len(env_config) == 0:
    # print("Warning: No env_config provided. Using default environment parameters.")

    env_defaults = globals_cfg.as_env_defaults()
    for key, value in env_defaults.items():
        env_config.setdefault(key, value)

    initial_p = kwargs.get("initial_pressure", None)
    if initial_p is None:
        initial_p = globals_cfg.initial_pressure
    else:
        initial_p = float(initial_p)  # type: ignore

    sample_rate_guess = getattr(pid, "dt", reference_curve.dt)
    env_config["sample_rate"] = sample_rate_guess
    env = ChamberEnv(env_config)

    sample_rate = env.sample_rate
    ref_dt = reference_curve.dt
    if not np.isclose(sample_rate, ref_dt, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "ChamberEnv sample_rate must match ReferenceCurve.dt. "
            f"Got sample_rate={sample_rate}, reference_dt={ref_dt}."
        )

    if hasattr(pid, "dt"):
        try:
            setattr(pid, "dt", sample_rate)
        except Exception:
            pass

    times = reference_curve.get_time_steps()
    targets = reference_curve.get_values()
    pressures: list[float] = []

    previous_target: float | None = None
    initial_alpha = 0.0
    for n_steps, target in reference_curve.iterate_steps():
        # 2. Reset the environment to a starting state
        observation = env.reset(
            initial_pressure=initial_p,
            goal_pressure=target,
            initial_alpha=initial_alpha,
        )
        measurment = float(observation[0])
        error = float(observation[2])

        direction_setter = getattr(pid, "set_direction", None)
        if callable(direction_setter):
            delta = 0.0 if previous_target is None else float(target - previous_target)
            if abs(delta) <= 1.0e-6:
                direction = "flat"
            elif delta > 0:
                direction = "up"
            else:
                direction = "down"
            try:
                direction_setter(direction)
            except Exception:
                pass

        for _ in range(n_steps):
            pressures.append(measurment)
            u, *_ = pid.update(error, measurment, desired_value=float(target))
            u = pid.transform_control_variable(u)
            obs, *_ = env.step(action=u, duration=sample_rate)
            # unpack the observation
            measurment = float(obs[0])
            error = float(obs[2])
        previous_target = float(target)
        initial_p = measurment  # Update for the next step
        initial_alpha = float(obs[1])

    return times, targets, np.array(pressures, dtype=float)


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Configure and create the environment
    env_config = {
        "p_goal": 8.0,  # We want to reach 8.0 torr
        "v_speed": 1000.0,  # Use a faster valve
        "sample_rate": 0.05,
        "time_step": 0.005,
    }
    env = ChamberEnv(env_config)

    # 2. Reset the environment to a starting state
    initial_p = 10.0
    observation = env.reset(initial_pressure=initial_p)

    # Store history for plotting, and log the initial state (t=0)
    time = 0.0
    history = {
        "time": [time],
        "pressure": [observation[0]],
        "alpha": [observation[1]],
        "reward": [env._calculate_reward()],  # Calculate initial reward
        "action": [0.0],  # Initial action is 0
    }

    # A simple "agent" that sets a fixed target opening
    # In a real scenario, a learning algorithm would choose the action.
    ACTION = 89.3  # Agent decides to set valve to 70%

    done = False
    while not done:
        # 3. Take a step in the environment
        observation, reward, done, info = env.step(ACTION)

        # Log data for the new step
        time += float(info.get("sample_interval", env.sample_rate))
        history["time"].append(time)
        history["pressure"].append(observation[0])
        history["alpha"].append(observation[1])
        history["reward"].append(reward)
        history["action"].append(ACTION)

        if done:
            print(f"Episode finished after {env.current_step} steps.")

    # 4. Plot the results of the episode
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Environment Simulation (Goal: {env.p_goal:.1f} torr)", fontsize=16)

    axs[0].plot(history["time"], history["pressure"], "b-", label="Pressure")
    axs[0].axhline(
        y=env.p_goal, color="k", linestyle="--", label=f"Goal ({env.p_goal:.1f} torr)"
    )
    axs[0].set_ylabel("Pressure (torr)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history["time"], history["alpha"], "r-", label="Valve Angle")
    axs[1].axhline(
        y=ACTION, color="k", linestyle="--", label=f"Action ({ACTION:.1f} %)"
    )
    axs[1].set_ylabel("Alpha (%)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(history["time"], history["reward"], "g-", label="Reward")
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.show()
