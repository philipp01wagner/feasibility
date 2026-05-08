import warnings
from typing import Any

import pandas as pd

from xingqi.config.config import XingqiConfig
from xingqi.controller.pid import AbstractPID, DummyPID
from xingqi.controller.reference_curve import ReferenceCurve
from xingqi.evaluation.evalute_controller import EvaluteController
from xingqi.simulation.ChamberEnv import simulate_matlab


def simulate(
    config: XingqiConfig,
    use_matlab_simulation: bool,
    pid: AbstractPID,
    reference_curve: ReferenceCurve,
    **kwargs: dict[str, Any],
) -> EvaluteController:
    """
    Run the simulation with the configuration, PID controller, and reference curve.

    kwargs : dict[str, Any]
        Additional keyword arguments to be passed to the simulation function.

    Parameters
    ----------
    config : XingqiConfig
        Configuration for the evaluation.
    use_matlab_simulation : bool
        If True, use the version of the simulation based on MATLAB.
        Else, fall back to the unified ChamberEnv dynamics.
    pid : AbstractPID
        PID controller to be used in the simulation.
    reference_curve : ReferenceCurve
        Reference curve for the simulation.

    Returns
    -------
    EvaluteController
        The result of the simulation evaluation.
        The object contains the time series data as pd.DataFrame.
        To calculate the performance metrics, call `.evaluate()` method.
    """
    if not use_matlab_simulation:
        warnings.warn(
            "The legacy C# simulation backend has been removed. "
            "Falling back to the ChamberEnv dynamics.",
            RuntimeWarning,
            stacklevel=2,
        )

    _, targets, pressures = simulate_matlab(pid, reference_curve, **kwargs)

    # Build the DataFrame for evaluation
    data = pd.DataFrame(
        {
            "setpoint": targets,
            "measurement": pressures,
        }
    )
    evaluator = EvaluteController(config, data, allow_multi_setpoint=True)
    return evaluator


def simulate_step_response(
    config: XingqiConfig,
    use_matlab_simulation: bool,
    percent_opening: float,
    reference_curve: ReferenceCurve,
    **kwargs: dict[str, Any],
) -> EvaluteController:
    """
    Run the step response simulation with the configuration.

    kwargs : dict[str, Any]
        Additional keyword arguments to be passed to the simulation function.

    Parameters
    ----------
    config : XingqiConfig
        Configuration for the evaluation.
    use_matlab_simulation : bool
        If True, use the version of the simulation based on MATLAB.
        Else, fall back to the unified ChamberEnv dynamics.
    percent_opening : float
        The percentage of valve opening for the step response (0 to 100).
    reference_curve : ReferenceCurve
        Reference curve for the simulation.
        Only used for determining the time steps / duration of the simulation.

    Returns
    -------
    EvaluteController
        The result of the simulation evaluation.
        The object contains the time series data as pd.DataFrame.
        To calculate the performance metrics, call `.evaluate()` method.
    """
    if percent_opening < 0.0 or percent_opening > 100.0:
        raise ValueError("percent_opening must be between 0 and 100")

    pid = DummyPID(dt=reference_curve.dt, despired_signal=percent_opening)
    eval = simulate(config, use_matlab_simulation, pid, reference_curve, **kwargs)

    data = eval.data["measurement"].to_numpy()
    return EvaluteController.from_step_response(
        config=config, dt=reference_curve.dt, measurement=data, u_zero=percent_opening
    )
