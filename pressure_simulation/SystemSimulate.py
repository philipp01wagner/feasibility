import matplotlib.pyplot as plt
import numpy as np

from xingqi.simulation.Alpha2Conductance import alpha_to_conductance
from xingqi.simulation.StatusCalculate import status_calculate


def system_simulate():
    """Replicate the MATLAB demo using the first-order valve model."""

    plt.close("all")

    system_params, flow_splines = alpha_to_conductance()

    t = np.arange(0.0, 3.000001, 1.0e-6)
    v = 1000.0
    p0 = 3.0
    pd = 10.0

    alpha, q, sigma, pressure, dpdt, _ = status_calculate(
        t,
        p0,
        pd,
        v,
        flow_splines=flow_splines,
        system_params=system_params,
    )

    sample_time = float(system_params.get("st", 1.0e-3))
    time_axis = t[0] + np.arange(alpha.size) * sample_time
    pd_plot = np.full_like(time_axis, pd)

    alpha_range = np.linspace(0.0, 100.0, 400)

    fig1 = plt.figure(1)
    fig1.suptitle("Calibration Splines", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.plot(alpha_range, flow_splines.alpha_to_pressure(alpha_range), "b")
    plt.xlabel("alpha (%)")
    plt.ylabel("Pressure (torr)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(
        alpha_range,
        flow_splines.alpha_to_pressure_derivative(alpha_range),
        "b",
    )
    plt.xlabel("alpha (%)")
    plt.ylabel("dP/dα")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(
        alpha_range,
        flow_splines.alpha_to_pressure_second_derivative(alpha_range),
        "b",
    )
    plt.xlabel("alpha (%)")
    plt.ylabel("d²P/dα²")
    plt.grid(True)

    fig1.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig2 = plt.figure(2)
    fig2.suptitle("Flow Characteristics", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(alpha_range, flow_splines.alpha_to_flow(alpha_range), "b")
    plt.xlabel("alpha (%)")
    plt.ylabel("Q")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        alpha_range,
        flow_splines.alpha_to_flow_derivative(alpha_range),
        "b",
    )
    plt.xlabel("alpha (%)")
    plt.ylabel("dQ/dα")
    plt.grid(True)

    fig2.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig3 = plt.figure(3)
    fig3.suptitle("Sigma Characteristics", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(alpha_range, flow_splines.alpha_to_sigma(alpha_range), "b")
    plt.xlabel("alpha (%)")
    plt.ylabel("σ")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        alpha_range,
        flow_splines.alpha_to_sigma_derivative(alpha_range),
        "b",
    )
    plt.xlabel("alpha (%)")
    plt.ylabel("dσ/dα")
    plt.grid(True)

    fig3.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig4 = plt.figure("Control Analysis")
    fig4.suptitle("Closed-loop Response", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(time_axis, alpha, "b", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("alpha (%)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time_axis, q, "b", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("Q")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time_axis, pressure, "b", linewidth=1)
    plt.plot(time_axis, pd_plot, "r", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("Pressure (torr)")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time_axis, dpdt, "b", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("dp/dt (torr/s)")
    plt.grid(True)

    fig4.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    system_simulate()
