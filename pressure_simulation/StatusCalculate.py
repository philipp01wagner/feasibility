from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from xingqi.simulation.Alpha2Conductance import FlowSplines
from xingqi.simulation.pressure_controller import QLimits, pressure_controller


def status_calculate(
    t: np.ndarray,
    p0: float,
    pd: float,
    v: float,
    *,
    flow_splines: FlowSplines,
    system_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the chamber dynamics using the updated first-order model."""

    st = float(system_params.get("st", 1.0e-3))
    dt = float(t[1] - t[0]) if t.size > 1 else st
    steps_per_sample = max(1, int(round(st / dt)))
    total_samples = int(np.floor((t[-1] - t[0]) / st)) + 1

    alpha = np.zeros(total_samples)
    q_values = np.zeros(total_samples)
    sigma_values = np.zeros(total_samples)
    pressures = np.zeros(total_samples)
    dpdt = np.zeros(total_samples)
    pval_record = np.zeros(total_samples)

    qm = float(system_params["qm"])
    sqrt_qm = np.sqrt(qm)
    R = float(system_params["R"])
    T = float(system_params["T"])
    M = float(system_params["M"])
    V = float(system_params["V"])
    rhs_term = R * T / M * qm

    alpha[0] = _initial_alpha(p0, flow_splines, qm, rhs_term)
    q_values[0] = float(flow_splines.alpha_to_flow(alpha[0]))
    sigma_values[0] = float(flow_splines.alpha_to_sigma(alpha[0]))
    pressures[0] = p0
    dpdt[0] = 0.0
    pval_record[0] = p0 - sigma_values[0] * sqrt_qm

    alpha_grid = np.linspace(0.0, 100.0, 1024)
    q_samples = flow_splines.alpha_to_flow(alpha_grid)
    q_limits = QLimits(float(np.min(q_samples)), float(np.max(q_samples)))

    alpha_prev = alpha[0]
    q_prev = q_values[0]
    sigma_prev = sigma_values[0]
    p_prev = pressures[0]

    sample_index = 1
    for step in range(1, t.size):
        ahat = pressure_controller(
            alpha_prev,
            p_prev,
            pd,
            flow_splines=flow_splines,
            system_params=system_params,
            limits=q_limits,
        )

        if p_prev - pd > 0:
            alpha_now = min(alpha_prev + v * dt, ahat)
        else:
            alpha_now = max(alpha_prev - v * dt, ahat)

        q_now = float(flow_splines.alpha_to_flow(alpha_now))
        sigma_now = float(flow_splines.alpha_to_sigma(alpha_now))

        dp_now = (rhs_term - q_prev * (p_prev - sigma_prev * sqrt_qm)) / V
        p_now = p_prev + dp_now * dt

        dpdt_now = (rhs_term - q_now * (p_now - sigma_now * sqrt_qm)) / V
        pval_now = p_prev - sigma_prev * sqrt_qm

        if step % steps_per_sample == 0 and sample_index < total_samples:
            alpha[sample_index] = alpha_now
            q_values[sample_index] = q_now
            sigma_values[sample_index] = sigma_now
            pressures[sample_index] = p_now
            dpdt[sample_index] = dpdt_now
            pval_record[sample_index] = pval_now
            sample_index += 1

        alpha_prev = alpha_now
        q_prev = q_now
        sigma_prev = sigma_now
        p_prev = p_now

    return (
        alpha[:sample_index],
        q_values[:sample_index],
        sigma_values[:sample_index],
        pressures[:sample_index],
        dpdt[:sample_index],
        pval_record[:sample_index],
    )


def _initial_alpha(
    pressure: float,
    flow_splines: FlowSplines,
    qm: float,
    rhs_term: float,
) -> float:
    alpha_candidates = np.arange(0.0, 100.0 + 0.5, 0.5)
    q_values = flow_splines.alpha_to_flow(alpha_candidates)
    sigma_values = flow_splines.alpha_to_sigma(alpha_candidates)
    sqrt_qm = np.sqrt(qm)
    l_values = q_values * sigma_values * sqrt_qm - q_values * pressure + rhs_term

    sort_idx = np.argsort(l_values)
    l_sorted = l_values[sort_idx]
    alpha_sorted = alpha_candidates[sort_idx]

    unique_l, unique_indices = np.unique(l_sorted, return_index=True)
    alpha_unique = alpha_sorted[unique_indices]

    if unique_l.size >= 2 and np.all(np.isfinite(unique_l)):
        spline = CubicSpline(unique_l, alpha_unique, extrapolate=True)
        alpha0 = float(spline(0.0))
    else:  # pragma: no cover - degenerate fallback mirrors MATLAB behaviour
        values = q_values * (pressure - sigma_values * sqrt_qm)
        idx = int(np.argmin(np.abs(values - rhs_term)))
        alpha0 = float(alpha_candidates[idx])

    return float(np.clip(alpha0, 0.0, 100.0))
