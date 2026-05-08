"""Lyapunov-based pressure controller translated from the MATLAB model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Alpha2Conductance import FlowSplines


@dataclass(frozen=True)
class QLimits:
    """Bounds for the outflow coefficient Q."""

    qmin: float
    qmax: float


def pressure_controller(
    alpha: float,
    pressure: float,
    pressure_target: float,
    *,
    flow_splines: FlowSplines,
    system_params: dict[str, float],
    limits: QLimits,
    gain: float = 6.5,
) -> float:
    """Compute the next valve position following the updated MATLAB logic."""

    sigma = float(flow_splines.alpha_to_sigma(alpha))
    q_ref = float(flow_splines.alpha_to_flow(alpha))
    dq_da = float(flow_splines.alpha_to_flow_derivative(alpha))

    R = system_params["R"]
    T = system_params["T"]
    M = system_params["M"]
    V = system_params["V"]
    qm = system_params["qm"]

    denom = pressure - sigma * np.sqrt(qm)
    if abs(denom) < 1e-9:
        denom = np.copysign(1e-9, denom if denom != 0 else 1.0)

    q_obj = (gain * V * (pressure - pressure_target) + R * T / M * qm) / denom
    q_obj = float(np.clip(q_obj, limits.qmin, limits.qmax))

    product = q_ref * q_obj
    if abs(product) < 1e-9:
        product = np.copysign(1e-9, product if product != 0 else 1.0)

    ahat = alpha + dq_da * (q_obj - q_ref) / product
    return float(np.clip(ahat, 0.0, 100.0))
