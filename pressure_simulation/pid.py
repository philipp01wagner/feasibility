import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np


class SharedPIDState:
    def __init__(self) -> None:
        """
        Shared state for PID controllers to allow sharing of data
        between PID controllers in an ensemble.
        Thoose values are the prev errors, integral and derivative terms.

        Note:
        - The integral and derivative term refer to the controllers output.
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = 0.0
        self.direction: Literal["up", "down", "flat"] = "flat"
        self.feed_forward_value: float = 0.0

    def update(
        self,
        prev_error: float,
        integral: float,
        derivative: float,
        prev_measurement: float,
        feed_forward_value: float,
    ) -> None:
        """
        Update the shared state.

        Parameters
        ----------
        prev_error : float
            new previous error
        integral : float
            new integral value
        derivative : float
            new derivative value
        prev_measurement : float
            new previous measurement
        """
        self.prev_error = prev_error
        self.integral = integral
        self.derivative = derivative
        self.prev_measurement = prev_measurement
        self.feed_forward_value = feed_forward_value

    def reset(self) -> None:
        """
        Reset the shared states to zero.
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = 0.0
        self.feed_forward_value = 0.0
        self.direction = "flat"

    def serialize(self) -> dict[str, float]:
        """
        Serialize the shared state to a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary with the shared state.
        """
        return {
            "prev_error": self.prev_error,
            "integral": self.integral,
            "derivative": self.derivative,
            "prev_measurement": self.prev_measurement,
            "feed_forward_value": self.feed_forward_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "SharedPIDState":
        """
        Create a SharedPIDState from a dictionary.

        Parameters
        ----------
        data : dict[str, float]
            Dictionary with the shared state.

        Returns
        -------
        SharedPIDState
            SharedPIDState object.
        """
        state = cls()
        state.prev_error = data.get("prev_error", 0.0)
        state.integral = data.get("integral", 0.0)
        state.derivative = data.get("derivative", 0.0)
        state.prev_measurement = data.get("prev_measurement", 0.0)
        state.feed_forward_value = data.get("feed_forward_value", 0.0)
        state.direction = data.get("direction", "flat")
        return state

    def set_direction(
        self,
        direction: Literal["up", "down", "flat"],
        use_up_ff: bool,
        use_down_ff: bool,
    ) -> None:
        """
        Set the direction of the PID controller.

        Note:
        - Adjust the integral term when changing to a direction with different ff

        Parameters
        ----------
        direction : Literal["up", "down", "flat"]
            Direction to be set.
        use_up_ff : bool
            Whether the controller uses feed-forward for up direction.
        use_down_ff: bool
            Whether the controller uses feed-forward for down direction.
        """
        _prev_direction = self.direction
        self.direction = direction

        # same direction -> no change
        if _prev_direction == direction:
            return

        # same ff settings -> no change
        if (use_up_ff and use_down_ff) or (not use_up_ff and not use_down_ff):
            return

        # from flat to up or down -> no change
        if _prev_direction == "flat" or direction == "flat":
            return

        # changing direction with different ff settings
        # differentiate between:
        # - from ff to no ff -> increase the integral by ff value -> no jump
        # - from no ff to ff -> decrease the integral by ff value -> no jump
        prev_ff = self.feed_forward_value > 0.0
        if direction == "up":
            new_ff = use_up_ff
        else:  # direction == "down"
            new_ff = use_down_ff

        # case 1: from ff to no ff
        if prev_ff and not new_ff:
            self.integral += self.feed_forward_value
        # case 2: from no ff to ff
        elif not prev_ff and new_ff:
            self.integral = 0

        self.integral = max(0.0, min(self.integral, 1.0))


class AbstractPID(ABC):
    def __init__(self) -> None:
        """
        Abstract base class for PID controllers.
        """
        super().__init__()

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the PID controller state (integral, derivative, previous error).
        """
        pass

    @abstractmethod
    def update(
        self,
        error: float,
        measurement: float | None = None,
        *,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Update the PID controller with the given error and return the control signal.

        Parameters
        ----------
        error : float
            Current error (setpoint - measurement)
        measurement : float | None, optional
            Current measurement value, by default None

        Returns
        -------
        float
            Control signal
        dict[str, float]
            dditional debug information
        """
        pass

    def set_parameters(self, params: dict[str, float]) -> None:
        """
        Set the PID parameters.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary with the parameters to be set.
            Possible keys are 'kp', 'ki', 'kd', 'kaw', 'lower_bound', 'upper_bound'
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PID parameters to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with the parameters.
        """
        _dict = deepcopy(self.__dict__)
        if "shared_state" in _dict:
            _dict["shared_state"] = None  # do not serialize state
        if "feed_forward_controller" in _dict:
            _dict["feed_forward_controller"] = _dict[
                "feed_forward_controller"
            ].to_dict()
        return _dict

    @classmethod
    def from_dict(
        cls, params: dict[str, Any], state: SharedPIDState | None = None
    ) -> "AbstractPID":
        """
        Set the PID parameters from a dictionary.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary with the parameters to be set.
        state : SharedPIDState | None, optional
            Shared PID state, by default None

        Returns
        -------
        AbstractPID
            PID controller with the given parameters.
        """
        if "shared_state" in params:
            params["shared_state"] = state
        return cls(**params)

    def to_json(self, file_path: str) -> None:
        """
        Save the PID parameters to a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(
        cls, file_path: str, state: SharedPIDState | None = None
    ) -> "AbstractPID":
        """
        Load the PID parameters from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file.
        state : SharedPIDState | None, optional
            Shared PID state, by default None

        Returns
        -------
        AbstractPID
            PID controller with the loaded parameters.
        """
        with open(file_path, "r") as f:
            params = json.load(f)
        return cls.from_dict(params, state=state)

    @abstractmethod
    def transform_control_variable(self, u: float) -> float:
        """
        Transform the control variable to the actual actuator signal.

        Parameters
        ----------
        u : float
            Control variable to be transformed.

        Returns
        -------
        float
            Transformed control variable.
        """

    def __call__(
        self,
        error: float,
        measurement: float | None = None,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        return self.update(error, measurement, desired_value=desired_value)


class DummyPID(AbstractPID):
    def __init__(self, dt: float, despired_signal: float) -> None:
        """
        Dummy PID controller that always returns the desired signal.

        Parameters
        ----------
        dt : float
            Sampling time [in seconds]
        despired_signal : float
            Desired control signal to be returned by the controller
        """
        super().__init__()
        self.despired_signal = despired_signal
        self.dt = dt

    def reset(self) -> None:
        """
        Reset the Dummy PID controller state (no state to reset).
        """
        pass

    def update(
        self,
        error: float,
        measurment: float | None = None,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Update the Dummy PID controller and return the desired signal.

        Parameters
        ----------
        error : float
            Current error (setpoint - measurement), not used
        measurment : float | None, optional
            current measurement (not used), by default None

        Returns
        -------
        float
            Desired control signal
        dict[str, float]
            Empty dictionary (no additional information)
        """
        return self.despired_signal, {}

    def transform_control_variable(self, u: float) -> float:
        """
        Transform the control variable to the actual actuator signal.

        Parameters
        ----------
        u : float
            Control variable to be transformed.

        Returns
        -------
        float
            Transformed control variable.
        """
        factor = 100.0 if self.despired_signal <= 1.0 else 1.0
        return u * factor


class AbstractFeedForwardController(AbstractPID):
    def __init__(self, adjust_integral: bool = True) -> None:
        """
        Abstract base class for feed-forward controllers.

        Parameters
        ----------
        adjust_integral : bool, optional
            Whether to adjust the integral term when switching ff on/off,
            by default True
        """
        super().__init__()
        self.adjust_integral = adjust_integral

    def get_additional_info_keys(self) -> list[str]:
        return ["u_ff"]

    def reset(self) -> None:
        return super().reset()

    def transform_control_variable(self, u: float) -> float:
        return u

    def to_dict(self) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return "NO-FF"

    @abstractmethod
    def in_use(self) -> bool:
        """
        Check if the feed-forward controller is in use.

        Returns
        -------
        bool
            True if the feed-forward controller is in use, False otherwise.
        """
        pass


class DummyFeedForwardController(AbstractFeedForwardController):
    def update(
        self,
        error: float,
        measurement: float | None = None,
        *,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        return 0.0, {"u_ff": 0.0}

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DummyFeedForwardController)

    def in_use(self) -> bool:
        return False


class FeedForwardController(AbstractFeedForwardController):
    def __init__(
        self,
        control_signals: list[float],
        desired_values: list[float],
        adjust_integral: bool = True,
    ) -> None:
        self.control_signals = np.array(control_signals)
        self.desired_values = np.array(desired_values)
        self.u_min = float(np.min(self.control_signals))
        self.u_max = float(np.max(self.control_signals))
        self._check()
        super().__init__(adjust_integral=adjust_integral)

    def in_use(self) -> bool:
        return True

    def _check(self) -> None:
        if len(self.control_signals) != len(self.desired_values):
            raise ValueError(
                "Control signals and desired values must have the same length"
            )
        if len(self.control_signals) < 2:
            raise ValueError(
                "At least two control signals and desired values are required for interpolation"  # noqa: E501
            )

        # check if the desired values are sorted
        if not np.all(np.diff(self.desired_values) >= 0):
            raise ValueError("Desired values must be sorted in ascending order")

    def update(
        self,
        error: float,
        measurement: float | None = None,
        desired_value: float | None = 0,
    ) -> tuple[float, dict[str, float]]:
        if desired_value is None:
            return 0.0, {"u_ff": 0.0}

        u_ff = np.interp(
            desired_value,
            self.desired_values,
            self.control_signals,
            left=self.u_max,
            right=self.u_min,
        )
        u_ff = float(u_ff)
        # clip to 0, 1
        u_ff = max(0.0, min(1.0, u_ff))
        return u_ff, {"u_ff": u_ff}

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, FeedForwardController):
            return False

        return np.array_equal(
            self.control_signals, value.control_signals
        ) and np.array_equal(self.desired_values, value.desired_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_signals": self.control_signals.tolist(),
            "desired_values": self.desired_values.tolist(),
        }

    def __str__(self) -> str:
        return "FF"


class PID(AbstractPID):
    def __init__(
        self,
        dt: float,
        kp: float,
        ki: float,
        kd: float = 0.0,
        tau_factor: float = 10.0,
        use_antiwindup: bool = False,
        clamping_antiwindup: bool = False,
        kaw: float | None = None,
        u_bounds: tuple[float, float] = (0.0, 1.0),
        state: SharedPIDState | None = None,
        operating_window: tuple[float, float] = (float("-inf"), float("inf")),
        feed_forward_controller: AbstractFeedForwardController | None = None,
    ):
        """
        Implementation of a PID controller with anti-windup and operating window.

        Parameters
        ----------
        dt : float
            Sampling time [in seconds]
        kp : float
            proportional gain
        ki : float
            integral gain
            - Approximation of integral term is done using trapezoidal integration.
            - Units of ki are 1/s.
            - ki will be multiplied with dt internally.
        kd : float, optional
            derivative gain, by default 0.0
            - Derivative term is filtered using a first-order low-pass filter
            - see tau_factor
        tau_factor : float, optional
            factor for the low-pass filter of the derivate part, by default 10.0
            - a low pass filter of first order is used
            - tau factor is multiplied with dt to get the actual tau
            - tau = tau_factor * dt
        use_antiwindup : bool, optional
            wheter to use anti-windup or not, by default False
            - Type of anti-windup is defined by clamping_antiwindup and kaw
        clamping_antiwindup : bool, optional
            wether to use clamping or back-calculation as aw, by default False
            - if True, clamping is used, else back-calculation is used
            - Clamping: clip the integral term to the actuator bounds
            - Back-calculation: reduce the integral based on u_sat - u_unsat
        kaw : float | None, optional
            anti-windup gain for back calculation, by default None
            - if None and ki != 0, 1/(ki*dt) is used (Dead-Beat Anti-Windup)
            - Factor for back-calculation anti-windup:
            - kaw * ki * dt
            - If dead-beat -> factor is 1
        u_bounds : tuple[float, float], optional
            bounds of the actuator, by default (0.0, 1.0)
        state : SharedPIDState | None, optional
            shared pid state, by default None
            if None, a new state will be created
        operating_window : tuple[float, float], optional
            operating window of that PID, by default (float("-inf"), float("inf"))

        Raises
        ------
        ValueError
            If kaw is None and ki is zero.
        ValueError
            If operating window is invalid (lower bound greater than upper bound).
        """
        super().__init__()
        # Sampling time
        self.dt = dt

        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Derivative filter (first-order low-pass)
        self.tau = tau_factor * self.dt
        self.alpha = self.tau / (self.tau + self.dt)

        # Anti-windup
        self.use_antiwindup = use_antiwindup
        self.clamping_antiwindup = clamping_antiwindup
        self.lower_bound, self.upper_bound = u_bounds
        if self.use_antiwindup and not self.clamping_antiwindup:
            if kaw is None and ki == 0:
                raise ValueError("kaw must be provided if ki is zero")
            self.kaw = kaw or 1 / (self.ki * self.dt)  # Dead-Beat Anti-Windup
        else:
            self.kaw = 0.0

        # operating window
        self.op_lower, self.op_upper = operating_window
        if self.op_lower > self.op_upper:
            raise ValueError(
                "Invalid operating window: lower bound is greater than upper bound"
            )

        # Feed-forward controller
        self.feed_forward_controller: AbstractFeedForwardController = (
            feed_forward_controller or DummyFeedForwardController()
        )

        # saved values
        self.shared_state = state or SharedPIDState()

    def clamp(self, value: float) -> float:
        """
        Clamp the value to the actuator bounds.

        Parameters
        ----------
        value : float
            value to be clamped

        Returns
        -------
        float
            clamped value
        """
        return max(self.lower_bound, min(self.upper_bound, value))

    def update(
        self,
        error: float,
        measurment: float | None = None,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Update the PID controller with the given error and return the control signal.

        Note:
        - This also updates the shared state.

        Parameters
        ----------
        error : float
            current error (setpoint - measurement)
        measurment : float | None, optional
            current measurement, by default None
            if not None, derivative on measurement is used instead of error.
            Advantage: Derivative kick on setpoint change is avoided.

        Returns
        -------
        float
            control signal
        dict[str, float]
            dictionary with detailed information about the control signal
            - u_p: proportional part of the control signal
            - u_i: integral part of the control signal
            - u_d: derivative part of the control signal
            - u_unsat: unsaturated control signal (before clamping to actuator bounds)
            - u: saturated control signal (after clamping to actuator bounds)
            - i_aw: change in integral term due to anti-windup (0 if no aw is used)
        """
        # get the current state
        prev_error = self.shared_state.prev_error
        integral = self.shared_state.integral
        derivative = self.shared_state.derivative
        prev_measurement = self.shared_state.prev_measurement

        # Feed-forward part
        u_ff, u_ff_info = self.feed_forward_controller.update(
            error, measurment, desired_value=desired_value
        )

        # proportional
        u_p = self.kp * error

        # integral (trapezoidal integration)
        i_inc = self.ki * 0.5 * (error + prev_error) * self.dt
        integral += i_inc

        # derivative (filtered)
        if measurment is not None:
            diff = measurment - prev_measurement
        else:
            diff = error - prev_error
        derivative = (
            self.alpha * derivative + (self.kd * (1 - self.alpha) / self.dt) * diff
        )

        # control
        u_i = integral
        u_d = derivative
        u_unsat = u_p + u_i + u_d + u_ff
        u_sat = self.clamp(u_unsat)

        # anti-windup -> reduce the integral sum
        i_aw = 0
        if self.use_antiwindup and self.ki != 0:
            if self.clamping_antiwindup:
                # Conditional Integration (Clamping)
                saturated_high = u_unsat > self.upper_bound
                saturated_low = u_unsat < self.lower_bound

                i_aw = 0
                # If saturated and integrating further into saturation, stop integration
                # -> remove the last increment
                if (saturated_high and i_inc > 0) or (saturated_low and i_inc < 0):
                    integral -= i_inc
                    i_aw -= i_inc

                i_aw += self.clamp(integral) - integral
                integral = self.clamp(integral)
            else:
                # back-calculation anti-windup
                i_aw = self.kaw * (u_sat - u_unsat) * self.ki * self.dt
                integral += i_aw

        # update state
        self.shared_state.update(error, integral, derivative, measurment or 0.0, u_ff)

        return u_sat, {
            "u_p": u_p,
            "u_i": u_i,
            "u_d": u_d,
            "u_unsat": u_unsat,
            "u": u_sat,
            "i_aw": i_aw,
            **u_ff_info,
        }

    def get_additional_info_keys(self) -> list[str]:
        """
        Get the keys of the additional information returned by the PID controller.

        Returns
        -------
        list[str]
            list of keys of the additional information
        """
        return [
            "u_p",
            "u_i",
            "u_d",
            "u_unsat",
            "u",
            "i_aw",
        ] + self.feed_forward_controller.get_additional_info_keys()

    def reset(self) -> None:
        """
        Reset the PID controller state (integral, derivative, previous error).
        """
        self.shared_state.reset()

    def in_operation_window(self, operating_point: float) -> bool:
        """
        Check if the given operating point is within the operating window of the PID.

        Parameters
        ----------
        operating_point : float
            operating point to be checked

        Returns
        -------
        bool
            True if the operating point is within the operating window, False otherwise
        """
        return self.op_lower <= operating_point <= self.op_upper

    def contains_operating_window(self, other: "PID") -> bool:
        """
        Check if the operating window of the other PID is completely contained
        within the operating window of this PID.

        Parameters
        ----------
        other : PID
            other PID controller

        Returns
        -------
        bool
            True if the other PID's operating window is completely contained
            within this PID's operating window, False otherwise
        """
        return self.op_lower <= other.op_lower and self.op_upper >= other.op_upper

    def same_antiwindup_settings(self, other: "PID") -> bool:
        """
        Check if the aw settings of this PID are the same as those of another PID.

        Parameters
        ----------
        other : PID
            other PID controller

        Returns
        -------
        bool
            True if the anti-windup settings are the same, False otherwise
        """
        return (
            self.use_antiwindup == other.use_antiwindup
            and self.clamping_antiwindup == other.clamping_antiwindup
            and self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    def check_same(self, other: "PID", attr: str) -> bool:
        """
        Check if the given attribute is the same for this PID and another PID.

        Parameters
        ----------
        other : PID
            other PID controller
        attr : str
            attribute to be checked

        Returns
        -------
        bool
            True if the attribute is the same for both PIDs, False otherwise

        Raises
        ------
        AttributeError
            If the attribute is not found in one of the PIDs.
        """
        if hasattr(self, attr) and hasattr(other, attr):
            return getattr(self, attr) == getattr(other, attr)

        raise AttributeError(
            f"Attribute {attr} not found in one of the PID controllers"
        )

    def interpolate_parameter(self, p1: float, p2: float, factor: float) -> float:
        """
        Linearly interpolate between two parameters.

        Note:
        - if factor=0 -> return p1
        - if factor=1 -> return p2
        - else -> return linear interpolation between p1 and p2

        Parameters
        ----------
        p1 : float
            first parameter
        p2 : float
            second parameter
        factor : float
            interpolation factor between 0 and 1

        Returns
        -------
        float
            interpolated parameter
        """
        return p1 + (p2 - p1) * factor

    def interpolate_pid(self, other: "PID", operating_point: float) -> "PID":
        """
        Interpolate between this PID and another PID based on the given operating point.

        Note:
        - Performance checks to ensure compatibility of the two PIDs
        - The operating point must be within the operating windows of both PIDs
        - The operating windows of the two PIDs must overlap

        Parameters
        ----------
        other : PID
            other PID controller to interpolate with this one
        operating_point : float
            operating point for interpolation

        Returns
        -------
        PID
            new PID controller with interpolated parameters
        """
        # check operating window
        if not self.in_operation_window(operating_point):
            raise ValueError("Operating point out of range for this PID controller")

        if not other.in_operation_window(operating_point):
            raise ValueError(
                "Operating point out of range for the other PID controller"
            )

        if self.contains_operating_window(other):
            raise ValueError(
                "Other PID's operating window is contained within this PID's operating window."  # noqa: E501
            )

        if other.contains_operating_window(self):
            raise ValueError(
                "This PID's operating window is contained within the other PID's operating window."  # noqa: E501
            )
        # check compatibility
        if not self.same_antiwindup_settings(other):
            raise ValueError("PID controllers have different anti-windup settings")

        if not self.check_same(other, "tau"):
            raise ValueError("PID controllers have different tau values")

        if not self.check_same(other, "dt"):
            raise ValueError("PID controllers have different dt values")

        # now we know the operating point is within both controllers' operating windows
        # and the controllers are compatible
        # get the overlapping range
        window_lower = max(self.op_lower, other.op_lower)
        window_upper = min(self.op_upper, other.op_upper)
        if window_lower >= window_upper:
            raise ValueError("PID controllers' operating windows do not overlap")

        # factor=0: Parameter von PID mit op_lower==lower
        # factor=1: Parameter von PID mit op_upper==upper
        denom = window_upper - window_lower
        factor = min(max((operating_point - window_lower) / denom, 0.0), 1.0)

        # determine which PID is "upper" and which is "lower"
        if self.op_lower == window_lower and other.op_upper == window_upper:
            lower = other
            upper = self
        elif other.op_lower == window_lower and self.op_upper == window_upper:
            lower = self
            upper = other
        else:
            raise ValueError("Error during defining upper and lower for interpolation")

        new_kp = self.interpolate_parameter(lower.kp, upper.kp, factor)
        new_ki = self.interpolate_parameter(lower.ki, upper.ki, factor)
        new_kd = self.interpolate_parameter(lower.kd, upper.kd, factor)
        new_kaw = self.interpolate_parameter(lower.kaw, upper.kaw, factor)

        return PID(
            dt=self.dt,
            kp=new_kp,
            ki=new_ki,
            kd=new_kd,
            tau_factor=self.tau / self.dt,
            use_antiwindup=self.use_antiwindup,
            clamping_antiwindup=self.clamping_antiwindup,
            kaw=new_kaw,
            u_bounds=(self.lower_bound, self.upper_bound),
            state=self.shared_state,
            operating_window=(operating_point, operating_point),
            feed_forward_controller=self.feed_forward_controller,
        )

    def __str__(self) -> str:
        # PID parameters
        name = f"PID(kp={self.kp:.3f}, ki={self.ki:.3f}, kd={self.kd:.3f}, tau={self.tau:.3f}, dt={self.dt}s)"  # noqa: E501

        # add the operating window
        name += f"[{self.op_lower:.3f}, {self.op_upper:.3f}]"

        # add anti-windup info
        if self.use_antiwindup:
            name += ", AW"
            if self.clamping_antiwindup:
                name += "(clamping)"
            else:
                name += f"(kaw={self.kaw:.3f})"
        else:
            name += ", no AW"

        # Add Feed-Forward info
        name += f", {self.feed_forward_controller}"

        name += f"[{self.lower_bound:.3f}, {self.upper_bound:.3f}]"

        return name

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "PID") -> bool:
        if not isinstance(other, PID):
            return False

        # check all values
        for key in self.__dict__.keys():
            if not self.check_same(other, key):
                print(key, getattr(self, key), getattr(other, key))
                return False

        return True

    def transform_control_variable(self, u: float) -> float:
        """
        Transform the control variable to the actual actuator signal.

        Parameters
        ----------
        u : float
            Control variable to be transformed.

        Returns
        -------
        float
            Transformed control variable.

        Raises
        ------
        NotImplementedError
            if the bounds are not (0,1) or (0,100)
        """
        if (self.lower_bound, self.upper_bound) == (0.0, 1.0):
            return u * 100
        elif (self.lower_bound, self.upper_bound) == (0.0, 100.0):
            return u
        else:
            raise NotImplementedError(
                "transform_control_variable is only implemented for bounds (0,1) and (0,100)"  # noqa: E501
            )


class DirectionalPID(AbstractPID):
    def __init__(
        self,
        up_pid: PID,
        down_pid: PID,
        *,
        default_direction: str = "flat",
    ) -> None:
        """PID wrapper that switches gains based on reference direction."""

        super().__init__()
        self.up_pid = up_pid
        self.down_pid = down_pid
        self._validate_alignment()

        self.dt = self.up_pid.dt
        self.lower_bound = self.up_pid.lower_bound
        self.upper_bound = self.up_pid.upper_bound

        self._direction = default_direction
        self._last_non_flat_direction = "up"

    def _validate_alignment(self) -> None:
        if self.up_pid.dt != self.down_pid.dt:
            raise ValueError("DirectionalPID members must share the same dt")
        bounds_up = (self.up_pid.lower_bound, self.up_pid.upper_bound)
        bounds_down = (self.down_pid.lower_bound, self.down_pid.upper_bound)
        if bounds_up != bounds_down:
            raise ValueError("DirectionalPID members must share the same bounds")
        if id(self.up_pid.shared_state) != id(self.down_pid.shared_state):
            raise ValueError(
                "DirectionalPID members must reuse the same SharedPIDState"
            )
        if not self.up_pid.same_antiwindup_settings(self.down_pid):
            raise ValueError("DirectionalPID members must share anti-windup settings")

    def set_direction(self, direction: str, adjust_shared_state: bool = False) -> None:
        direction_lower = direction.lower()
        if direction_lower not in {"up", "down", "flat"}:
            raise ValueError(f"Unsupported direction '{direction}' for DirectionalPID")
        if direction_lower in {"up", "down"}:
            self._last_non_flat_direction = direction_lower
        self._direction = direction_lower

        # Update the shared state direction
        if not adjust_shared_state:
            return

        use_up_ff = self.up_pid.feed_forward_controller.in_use()
        use_down_ff = self.down_pid.feed_forward_controller.in_use()

        if use_down_ff and not self.down_pid.feed_forward_controller.adjust_integral:
            return
        if use_up_ff and not self.up_pid.feed_forward_controller.adjust_integral:
            return

        # Update in one direction is sufficient, both share the same state
        _dir = self._direction
        if _dir == "flat":
            _dir = self._last_non_flat_direction

        self.up_pid.shared_state.set_direction(
            direction=_dir,  # type: ignore[arg-type]
            use_up_ff=use_up_ff,
            use_down_ff=use_down_ff,
        )

    def _active_pid(self) -> PID:
        if self._direction == "down":
            return self.down_pid
        if self._direction == "flat":
            if self._last_non_flat_direction == "down":
                return self.down_pid
            return self.up_pid
        return self.up_pid

    def update(
        self,
        error: float,
        measurement: float | None = None,
        desired_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        controller = self._active_pid()
        control, info = controller.update(
            error, measurement, desired_value=desired_value
        )
        info = dict(info)
        direction_encoding = {"up": 1.0, "down": -1.0, "flat": 0.0}
        info["direction_flag"] = direction_encoding[self._direction]
        return control, info

    def reset(self) -> None:
        self.up_pid.reset()
        self.down_pid.reset()
        self._direction = "flat"
        self._last_non_flat_direction = "up"

    def transform_control_variable(self, u: float) -> float:
        return self._active_pid().transform_control_variable(u)

    def get_additional_info_keys(self) -> list[str]:
        keys = set(self.up_pid.get_additional_info_keys())
        keys.update(self.down_pid.get_additional_info_keys())
        keys.add("direction_flag")
        return sorted(keys)

    def __str__(self) -> str:
        return f"DirectionalPID(up={self.up_pid}, down={self.down_pid})"

    def interpolate_pid(self, other: "DirectionalPID", operating_point: float) -> PID:
        this_active_pid = self._active_pid()
        other_active_pid = other._active_pid()
        return this_active_pid.interpolate_pid(other_active_pid, operating_point)


if __name__ == "__main__":
    pid = PID(1.0, kp=0.5, ki=0.1)
    print(pid.update(1.0))
    print(pid)
