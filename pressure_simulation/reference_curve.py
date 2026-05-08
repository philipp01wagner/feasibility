from collections.abc import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np


class ReferenceCurve:
    def __init__(
        self,
        dt: float,
        steps: list[tuple[int, float]],
    ) -> None:
        """
        Create a reference curve from a list of (n timesteps, desired value) tuples.

        Note:
        - The first tuple defines the starting point of the curve.
        - Therefore it does not define a step, but the initial value.

        Parameters
        ----------
        dt : float
            sampling time of the reference curve [in seconds]
        steps : list[tuple[int, float]]
            list of (n timesteps, desired value) tuples
        """
        self.dt = dt
        self.steps = steps

        self.time_steps = np.arange(
            0, self.dt * sum(s[0] for s in self.steps) - self.dt / 100, self.dt
        )
        discrete_values = []
        for n, v in self.steps:
            discrete_values.extend([v] * n)
        self.values = np.array(discrete_values)

    def __len__(self) -> int:
        return len(self.time_steps)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        name = f"ReferenceCurve(dt={self.dt}s, n_changes={self.n_changes}, "
        name += f"total_time={self.dt * len(self)}s, n_steps={len(self)})"
        return name

    def get_values(self) -> np.ndarray:
        """
        Get the reference curve values.

        Returns
        -------
        np.ndarray
            reference curve values
        """
        return self.values

    def get_time_steps(self) -> np.ndarray:
        """
        Get the time steps of the reference curve.

        Returns
        -------
        np.ndarray
            time steps of the reference curve
        """
        return self.time_steps

    @property
    def n_changes(self) -> int:
        """
        Get the number of changes in the reference curve.

        Returns
        -------
        int
            number of changes in the reference curve
        """
        return len(self.steps) - 1

    def plot(self, path: str | None = None, name: str = "reference_curve") -> None:
        """
        Plot the reference curve.

        Parameters
        ----------
        path : str | None, optional
            by default None
            - if None, the plot will be shown
            - if str, the plot will be saved to the given path
        name : str, optional
            name for the plot, by default "reference_curve"
        """
        plt.figure(figsize=(10, 5))
        plt.step(self.time_steps, self.values, where="post")
        plt.xlabel("Time [s]")
        plt.ylabel("Reference value")
        plt.title("Reference Curve")
        plt.grid()
        if path is None:
            plt.show()
        else:
            plt.savefig(f"{path}/{name}.png")
        plt.close()

    def iterate_steps(self) -> Iterator[tuple[int, float]]:
        """
        Iterate over the steps of the reference curve.

        Returns
        ------
        Iterator[tuple[int, float]]
            Iterator over (n timesteps, desired value) tuples.
        """
        for n_steps, target in self.steps:
            yield n_steps, target


def build_reference_curve(
    dt: float, schedule: Sequence[tuple[float, float] | tuple[float, float, float]]
) -> ReferenceCurve:
    """
    Create a :class:`ReferenceCurve` from duration/value pairs.

    Parameters
    ----------
    dt : float
        Sampling time in seconds for the generated curve.
        schedule : Sequence[tuple[float, float] | tuple[float, float, float]]
                Sequence of schedule entries.
                - ``(duration_seconds, target_value)`` keeps the value constant for the
                    whole duration.
                - ``(duration_seconds, target_value, ramp_seconds)`` linearly ramps from
                    the previous value to ``target_value`` during ``ramp_seconds``
                    and then keeps the target value for the rest of the duration.

                Constraints:
                - each duration must be positive,
                - ``0 <= ramp_seconds <= duration_seconds``,
                - the first entry must not contain a ramp (its ramp must be 0).

    Returns
    -------
    ReferenceCurve
        Generated reference curve.

    Examples
    --------
    >>> curve = build_reference_curve(
    ...     dt=0.01,
    ...     schedule=[
    ...         (1.0, 10.0, 0.0),  # 0 - 1 s: 10 Torr
    ...         (1.0, 25.0, 0.5),  # 1 - 1.5 s: ramp to 25 Torr, then hold
    ...         (1.5, 10.0, 0.0),  # 2 - 3.5 s: back to 10 Torr
    ...     ],
    ... )
    >>> len(curve)
    350
    """

    steps: list[tuple[int, float]] = []
    previous_value: float | None = None

    for i, entry in enumerate(schedule):
        if len(entry) == 2:
            duration, value = entry
            ramp_duration = 0.0
        elif len(entry) == 3:
            duration, value, ramp_duration = entry
        else:
            raise ValueError(
                "Each schedule entry must be (duration, value) or "
                "(duration, value, ramp_duration)"
            )

        if duration <= 0:
            raise ValueError("Durations must be positive")

        if ramp_duration < 0:
            raise ValueError("Ramp duration must be non-negative")
        if ramp_duration > duration:
            raise ValueError("Ramp duration must be <= segment duration")
        if i == 0 and ramp_duration > 0:
            raise ValueError("The first schedule entry must not define a ramp")

        n_steps = max(1, int(round(duration / dt)))
        n_ramp_steps = int(round(ramp_duration / dt))

        if n_ramp_steps > 0 and previous_value is not None:
            for j in range(n_ramp_steps):
                frac = (j + 1) / n_ramp_steps
                ramp_value = previous_value + (value - previous_value) * frac
                steps.append((1, float(ramp_value)))
            n_hold_steps = n_steps - n_ramp_steps
            if n_hold_steps > 0:
                steps.append((n_hold_steps, value))
        else:
            steps.append((n_steps, value))

        previous_value = value

    return ReferenceCurve(dt=dt, steps=steps)


if __name__ == "__main__":
    rc = ReferenceCurve(
        dt=0.1,
        steps=[
            (5, 0.0),
            (10, 1.0),
        ],
    )
    print(rc)
    print(len(rc))
    rc.plot()  # show plot
