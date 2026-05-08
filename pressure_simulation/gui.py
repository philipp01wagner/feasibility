from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass, replace
from tkinter import messagebox
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ChamberEnv import ChamberEnv
from global_settings import get_simulation_globals
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")


@dataclass
class PIDState:
    integral: float = 0.0
    prev_error: float = 0.0


@dataclass
class SimulationSettings:
    initial_pressure: float
    target_pressure: float
    sample_rate: float
    time_step: float
    valve_speed: float
    update_ms: int
    use_pid: bool
    kp: float
    ki: float
    kd: float
    manual_action: float


class GUISimulator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Chamber Simulator")
        self.geometry("1000x700")

        self.env: Optional[ChamberEnv] = None
        self.pid_state = PIDState()
        self.running = False
        self.after_id: Optional[str] = None

        self._build_variables()
        self._build_ui()
        self._build_plot()
        self._update_status()

    def _build_variables(self) -> None:
        globals_cfg = get_simulation_globals()

        self.initial_pressure_var = tk.DoubleVar(value=globals_cfg.initial_pressure)
        self.target_pressure_var = tk.DoubleVar(value=globals_cfg.initial_pressure)
        self.sample_rate_var = tk.DoubleVar(value=globals_cfg.sample_rate)
        self.time_step_var = tk.DoubleVar(value=globals_cfg.time_step)
        self.valve_speed_var = tk.DoubleVar(value=globals_cfg.speed_valve)
        self.action_var = tk.DoubleVar(value=50.0)
        self.update_ms_var = tk.IntVar(value=100)

        self.use_pid_var = tk.BooleanVar(value=False)
        self.kp_var = tk.DoubleVar(value=0.5)
        self.ki_var = tk.DoubleVar(value=0.1)
        self.kd_var = tk.DoubleVar(value=0.01)

        self.time = 0.0
        self.pressures: list[float] = []
        self.targets: list[float] = []
        self.valve_angles: list[float] = []
        self.times: list[float] = []

        self.status_pressure = tk.StringVar(value="---")
        self.status_alpha = tk.StringVar(value="---")
        self.status_time = tk.StringVar(value="0.0 s")
        self.status_message = tk.StringVar(value="Ready")

        self.pending_settings = SimulationSettings(
            initial_pressure=float(self.initial_pressure_var.get()),
            target_pressure=float(self.target_pressure_var.get()),
            sample_rate=float(self.sample_rate_var.get()),
            time_step=float(self.time_step_var.get()),
            valve_speed=float(self.valve_speed_var.get()),
            update_ms=int(self.update_ms_var.get()),
            use_pid=bool(self.use_pid_var.get()),
            kp=float(self.kp_var.get()),
            ki=float(self.ki_var.get()),
            kd=float(self.kd_var.get()),
            manual_action=float(np.clip(self.action_var.get(), 0.0, 100.0)),
        )
        self.active_settings: Optional[SimulationSettings] = None

    def _build_ui(self) -> None:
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        row0 = [
            ("Initial P (Torr)", self.initial_pressure_var, 8),
            ("Target P (Torr)", self.target_pressure_var, 8),
            ("Sample rate (s)", self.sample_rate_var, 8),
            ("Time step (s)", self.time_step_var, 8),
            ("Valve speed (%/s)", self.valve_speed_var, 8),
            ("Update (ms)", self.update_ms_var, 6),
        ]
        col = 0
        for label_text, variable, width in row0:
            tk.Label(control_frame, text=label_text).grid(
                row=0,
                column=col,
                padx=4,
                pady=2,
                sticky="w",
            )
            tk.Entry(control_frame, textvariable=variable, width=width).grid(
                row=0,
                column=col + 1,
                padx=4,
            )
            col += 2

        tk.Checkbutton(
            control_frame,
            text="Use PID",
            variable=self.use_pid_var,
        ).grid(row=1, column=0, padx=4, pady=4, sticky="w")

        col = 1
        for label_text, variable in (
            ("Kp", self.kp_var),
            ("Ki", self.ki_var),
            ("Kd", self.kd_var),
        ):
            tk.Label(control_frame, text=label_text).grid(
                row=1,
                column=col,
                padx=4,
                pady=2,
                sticky="w",
            )
            tk.Entry(control_frame, textvariable=variable, width=8).grid(
                row=1,
                column=col + 1,
                padx=4,
            )
            col += 2

        tk.Label(control_frame, text="Manual valve (%)").grid(
            row=1,
            column=col,
            padx=4,
            pady=2,
            sticky="w",
        )
        tk.Entry(control_frame, textvariable=self.action_var, width=8).grid(
            row=1,
            column=col + 1,
            padx=4,
        )

        buttons = [
            ("Apply", self.apply_settings),
            ("Start", self.start),
            ("Stop", self.stop),
            ("Reset", self.reset),
        ]
        for idx, (text, callback) in enumerate(buttons):
            tk.Button(control_frame, text=text, command=callback).grid(
                row=2,
                column=idx,
                padx=4,
                pady=6,
                sticky="ew",
            )

        status_frame = tk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        tk.Label(status_frame, text="Pressure:").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.status_pressure).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Label(status_frame, text="Valve:").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.status_alpha).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Label(status_frame, text="Time:").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.status_time).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Label(status_frame, textvariable=self.status_message).pack(side=tk.RIGHT)

    def _build_plot(self) -> None:
        self.fig, (self.ax_p, self.ax_v) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True
        )
        self.fig.subplots_adjust(hspace=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ax_p.set_ylabel("Pressure (Torr)")
        self.ax_v.set_ylabel("Valve (%)")
        self.ax_v.set_xlabel("Time (s)")
        (self.line_pressure,) = self.ax_p.plot([], [], label="measurement")
        (self.line_target,) = self.ax_p.plot([], [], linestyle="--", label="target")
        (self.line_valve,) = self.ax_v.plot([], [], label="alpha")
        self.ax_p.legend()
        self.ax_v.legend()

    def apply_settings(self) -> None:
        try:
            new_settings = self._gather_settings()
        except ValueError as exc:
            messagebox.showerror("Settings", str(exc))
            return

        previous_active = self.active_settings
        self.pending_settings = new_settings

        if self.env is not None:
            initial_changed = False
            if previous_active is not None:
                initial_changed = not math.isclose(
                    previous_active.initial_pressure,
                    new_settings.initial_pressure,
                )

            self._apply_settings_to_env(new_settings)
            self.pid_state = PIDState()
            if self.running and self.after_id is not None:
                try:
                    self.after_cancel(self.after_id)
                except Exception:
                    pass
                self.after_id = None
                self._schedule_step()
            if initial_changed:
                self._set_status_message(
                    "Settings applied; reset to use new initial pressure."
                )
            else:
                self._set_status_message("Settings applied to simulation.")
        else:
            self._set_status_message("Settings staged. Press Start to run.")

        self._update_status()

    def start(self) -> None:
        if self.running:
            return
        if self.env is None:
            if not self._initialize_environment():
                return
        elif self.active_settings is None:
            self._apply_settings_to_env(self.pending_settings)
        self.running = True
        self._set_status_message("Running")
        self._schedule_step()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self._set_status_message("Stopped")

    def reset(self) -> None:
        self.stop()
        self.time = 0.0
        self.pressures.clear()
        self.targets.clear()
        self.valve_angles.clear()
        self.times.clear()
        self.env = None
        self._initialize_environment()

    def _gather_settings(self) -> SimulationSettings:
        try:
            initial_pressure = float(self.initial_pressure_var.get())
            target_pressure = float(self.target_pressure_var.get())
            sample_rate = float(self.sample_rate_var.get())
            time_step = float(self.time_step_var.get())
            valve_speed = float(self.valve_speed_var.get())
            use_pid = bool(self.use_pid_var.get())
            kp = float(self.kp_var.get())
            ki = float(self.ki_var.get())
            kd = float(self.kd_var.get())
            manual_action = float(self.action_var.get())
            update_ms = int(self.update_ms_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid input: {exc}")

        if sample_rate <= 0 or time_step <= 0:
            raise ValueError("Sample interval and time step must be positive")
        if time_step > sample_rate:
            time_step = sample_rate
            self.time_step_var.set(time_step)
        manual_action = float(np.clip(manual_action, 0.0, 100.0))
        self.action_var.set(manual_action)
        if update_ms <= 0:
            update_ms = 20
            self.update_ms_var.set(update_ms)
        self.use_pid_var.set(use_pid)
        self.kp_var.set(kp)
        self.ki_var.set(ki)
        self.kd_var.set(kd)

        return SimulationSettings(
            initial_pressure=initial_pressure,
            target_pressure=target_pressure,
            sample_rate=sample_rate,
            time_step=time_step,
            valve_speed=valve_speed,
            update_ms=update_ms,
            use_pid=use_pid,
            kp=kp,
            ki=ki,
            kd=kd,
            manual_action=manual_action,
        )

    def _initialize_environment(
        self, settings: Optional[SimulationSettings] = None
    ) -> bool:
        try:
            target_settings = settings or self.pending_settings
            env_config = {
                "p_goal": target_settings.target_pressure,
                "sample_rate": target_settings.sample_rate,
                "time_step": target_settings.time_step,
                "v_speed": target_settings.valve_speed,
                "max_steps": 100000,
            }
            self.env = ChamberEnv(env_config)
            self.env.reset(
                initial_pressure=target_settings.initial_pressure,
                goal_pressure=target_settings.target_pressure,
            )
            self._apply_settings_to_env(target_settings)
            self.pid_state = PIDState()
            self.time = 0.0
            self.pressures = [float(self.env.p)]
            self.targets = [float(self.env.p_goal)]
            self.valve_angles = [float(self.env.alpha)]
            self.times = [0.0]
            self._update_plot()
            self._update_status()
            self._set_status_message("Environment ready.")
            return True
        except Exception as exc:
            messagebox.showerror(
                "Environment",
                f"Failed to build environment: {exc}",
            )
            return False

    def _schedule_step(self) -> None:
        if not self.running:
            return
        settings = self._current_settings()
        delay = max(20, int(settings.update_ms))
        self.after_id = self.after(delay, self._run_step)

    def _run_step(self) -> None:
        if not self.env or not self.running:
            return

        settings = self._current_settings()
        action = float(settings.manual_action)
        target = float(settings.target_pressure)
        self.env.p_goal = target

        sample_interval = float(settings.sample_rate)

        if settings.use_pid:
            kp = float(settings.kp)
            ki = float(settings.ki)
            kd = float(settings.kd)
            measurement = float(self.env.p)
            error = measurement - target
            self.pid_state.integral += error * sample_interval
            derivative = 0.0
            if sample_interval > 0:
                derivative = (error - self.pid_state.prev_error) / sample_interval
            action = kp * error + ki * self.pid_state.integral + kd * derivative
            action = float(np.clip(action, 0.0, 100.0))
            self.pid_state.prev_error = error

        observation, _, done, info = self.env.step(action, duration=sample_interval)
        measurement = float(observation[0])
        alpha = float(observation[1])
        self.time = float(info.get("time", self.time + sample_interval))

        self.times.append(self.time)
        self.pressures.append(measurement)
        self.targets.append(target)
        self.valve_angles.append(alpha)

        max_len = 20000
        if len(self.times) > max_len:
            del self.times[:1000]
            del self.pressures[:1000]
            del self.targets[:1000]
            del self.valve_angles[:1000]

        self._update_plot()
        self._update_status()

        if done:
            messagebox.showinfo(
                "Simulation",
                "Episode finished (max steps reached).",
            )
            self.stop()
        else:
            self._schedule_step()

    def _update_plot(self) -> None:
        if not self.times:
            self.line_pressure.set_data([], [])
            self.line_target.set_data([], [])
            self.line_valve.set_data([], [])
        else:
            self.line_pressure.set_data(self.times, self.pressures)
            self.line_target.set_data(self.times, self.targets)
            self.line_valve.set_data(self.times, self.valve_angles)
            self.ax_p.relim()
            self.ax_p.autoscale_view()
            self.ax_v.relim()
            self.ax_v.autoscale_view()
        self.canvas.draw_idle()

    def _update_status(self) -> None:
        pressure = self.pressures[-1] if self.pressures else float("nan")
        alpha = self.valve_angles[-1] if self.valve_angles else float("nan")
        self.status_pressure.set(
            f"{pressure:.3f} Torr" if not math.isnan(pressure) else "---"
        )
        self.status_alpha.set(f"{alpha:.2f} %" if not math.isnan(alpha) else "---")
        self.status_time.set(f"{self.time:.3f} s")

    def _set_status_message(self, message: str) -> None:
        self.status_message.set(message)

    def _clone_settings(self, settings: SimulationSettings) -> SimulationSettings:
        return replace(settings)

    def _current_settings(self) -> SimulationSettings:
        return self.active_settings or self.pending_settings

    def _apply_settings_to_env(self, settings: SimulationSettings) -> None:
        self.active_settings = self._clone_settings(settings)
        if not self.env:
            return
        self.env.sample_rate = settings.sample_rate
        self.env.control_dt = settings.sample_rate
        self.env.time_step = settings.time_step
        self.env.integration_dt = settings.time_step
        self.env.dt = settings.time_step
        self.env.v_speed = settings.valve_speed
        self.env.max_steps = 100000
        self.env.p_goal = settings.target_pressure

    def _reset_pid_state(self) -> None:
        self.pid_state = PIDState()

    def on_close(self) -> None:
        self.stop()
        self.destroy()


if __name__ == "__main__":
    app = GUISimulator()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
