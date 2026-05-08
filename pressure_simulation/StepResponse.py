import matplotlib.pyplot as plt
import numpy as np

from xingqi.simulation.ChamberEnv import ChamberEnv


def main():
    """
    Main function to run the step response simulation using ChamberEnv.
    """
    plt.close("all")

    # --- 1. Simulation Configuration ---
    P0 = 20.0  # Initial pressure (torr)
    V_SPEED = 2000.0  # Valve plate speed (%/s)
    DURATION = 10.0  # Simulation duration (s)
    TARGET_OPENING = 0.0  # The target opening for the valve step (%)
    TIME_STEP = 0.002  # Time step for simulation consistency

    # --- 2. Initialization ---
    print("Initializing Chamber Environment...")
    # Configure the environment with the desired parameters
    env_config = {
        "v_speed": V_SPEED,
        "dt": TIME_STEP,
        "max_steps": int(DURATION / TIME_STEP),
    }
    env = ChamberEnv(env_config)

    # Reset the environment to the specified initial pressure
    observation = env.reset(initial_pressure=P0)
    print(
        f"Running step response simulation with P0={P0} torr to {TARGET_OPENING}% opening..."  # noqa: E501
    )

    # --- 3. Simulation Loop ---
    history = {"time": [], "pressure": [], "alpha": []}

    done = False
    time = 0.0
    while not done:
        # Log the current state before taking a step
        history["time"].append(time)
        history["pressure"].append(observation[0])
        history["alpha"].append(observation[1])

        # The action is always to drive the valve to the target opening
        action = TARGET_OPENING

        # Take a step in the environment
        observation, _, done, _ = env.step(action)
        time += env.dt

    print("Simulation finished.")

    # --- 4. Post-processing ---
    # Calculate the conductance history from the alpha history for plotting
    alpha_history = np.array(history["alpha"])
    conductance_history = env.a2q_spline(alpha_history)

    # --- 5. Plotting Results ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"System Step Response to {TARGET_OPENING}% Opening (P0 = {P0} torr)",
        fontsize=16,
    )

    # Plot 1: Valve Opening (alpha) vs. Time
    axs[0].plot(history["time"], history["alpha"], "r-", label="Valve Opening")
    axs[0].set_ylabel("Alpha (%)")
    axs[0].set_title("Valve Opening vs. Time")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Pressure (P) vs. Time
    axs[1].plot(history["time"], history["pressure"], "b-", label="Pressure")
    axs[1].set_ylabel("Pressure (torr)")
    axs[1].set_title("Pressure vs. Time")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Conductance (C) vs. Time
    axs[2].plot(history["time"], conductance_history, "g-", label="Conductance")
    axs[2].set_ylabel("Q")
    axs[2].set_title("Outflow Coefficient vs. Time")
    axs[2].grid(True)
    axs[2].legend()

    # Common X-axis label
    axs[2].set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
