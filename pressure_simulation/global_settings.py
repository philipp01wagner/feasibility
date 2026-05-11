from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator

GainBounds = tuple[tuple[float, float, float], tuple[float, float, float]]


ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = ROOT / "data" / "tv_monitor_demo.csm"


@dataclass(frozen=True)
class SimulationGlobals:
    """Container for simulation-wide settings loaded from the CSM file."""

    mode_pump: int = 0
    quant_pump: float = 9.0
    temperature_pump: float = 293.15
    molweight_pump: float = 28.0
    rho_pump: float = 1.25
    volume_chamb: float = 34.0
    temperature_chamb: float = 293.15
    initial_pressure_chamb: float = 5.0
    molweight_chamb: float = 28.0
    rho_chamb: float = 1.25
    setposition_control: float = 0.0
    initial_position_valve: float = 0.0
    speed_valve: float = 1000.0
    resolution_valve: float = 0.02
    sample_rate: float = 1.0e-3
    time_step: float = 1.0e-4
    max_steps: int = 500_000
    pid_gain_bounds: GainBounds | None = None
    pid_use_antiwindup: bool = False
    pid_clamping_antiwindup: bool = False
    pid_kaw: float | None = None
    pid_tau_factor: float = 10.0
    pid_ensemble: bool = False
    chamber_pressure_max: float | None = None
    chamber_pressure_min: float | None = None
    allowed_rise_time: float | None = None
    rise_time_penalty_coefficient: float = 10.0
    rise_time_penalty_offset: float = 100.0
    reference_jitter_enable: bool = False
    reference_jitter_duration_jitter: float = 0.0
    reference_jitter_pressure_jitter: float = 0.0
    reference_jitter_min_step_delta: float = 0.0
    early_stopping_patience: int | None = None
    grid_step_size: float = 5.0
    grid_task_duration_seconds: float | None = None
    reference_curve_schedule: tuple[tuple[float, float], ...] = (
        (1.0, 10.0),
        (1.0, 20.0),
        (1.0, 15.0),
        (1.0, 30.0),
        (1.0, 5.0),
        (1.0, 10.0),
    )
    objective_weights: tuple[tuple[str, float], ...] = (
        ("final_error", 1.0),
        ("settling_time", 0.2),
        ("overshoot", 0.4),
    )
    acquisition_function: str = "lcb"
    acquisition_beta: float = 0.1
    # Transfer learning / meta-learning configuration
    source_molweights: tuple[float, ...] = ()
    target_molweight: float | None = None
    source_dataset_iterations: int | None = None
    malibo_classifier_config: dict[str, Any] = field(default_factory=dict)
    malibo_train_config: dict[str, Any] = field(default_factory=dict)
    rgpe_mc_samples: int | None = None
    rgpe_mc_samples_ranking: int | None = None
    # =========================================================================
    # Optimization Mode Configuration (Direct vs CTM)
    # =========================================================================
    # optimization_mode: "direct" optimizes (kp, ki, kd) directly
    #                    "ctm" (or "indirect") uses CTM factors (f_p, f_i, f_d)
    optimization_mode: str = "direct"
    # PI controller bounds (used when ctm_use_derivative=False)
    ctm_pi_f_p_bounds: tuple[float, float] = (0.4, 1.2)
    ctm_pi_f_i_bounds: tuple[float, float] = (0.4, 4.0)
    ctm_pi_f_d_bounds: tuple[float, float] = (0.0, 0.0)
    # PID controller bounds (used when ctm_use_derivative=True)
    ctm_pid_f_p_bounds: tuple[float, float] = (0.2, 1.2)
    ctm_pid_f_i_bounds: tuple[float, float] = (0.5, 4.0)
    ctm_pid_f_d_bounds: tuple[float, float] = (0.05, 1.0)
    # Use PID (True) or PI (False) controller in CTM mode
    ctm_use_derivative: bool = False
    # Opening percentage for step response in CTM system identification
    ctm_opening_percentage: float = 0.25

    # -------------------------------------------------------------------------
    # Dynamic CTM bounds properties (select PI or PID based on ctm_use_derivative)
    # -------------------------------------------------------------------------
    @property
    def ctm_f_p_bounds(self) -> tuple[float, float]:
        """Return f_p bounds based on ctm_use_derivative setting."""
        if self.ctm_use_derivative:
            return self.ctm_pid_f_p_bounds
        return self.ctm_pi_f_p_bounds

    @property
    def ctm_f_i_bounds(self) -> tuple[float, float]:
        """Return f_i bounds based on ctm_use_derivative setting."""
        if self.ctm_use_derivative:
            return self.ctm_pid_f_i_bounds
        return self.ctm_pi_f_i_bounds

    @property
    def ctm_f_d_bounds(self) -> tuple[float, float]:
        """Return f_d bounds based on ctm_use_derivative setting."""
        if self.ctm_use_derivative:
            return self.ctm_pid_f_d_bounds
        return self.ctm_pi_f_d_bounds

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "SimulationGlobals":
        values: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__.keys():  # type: ignore[attr-defined]
            if key == "reference_curve_schedule" and key in raw:
                values[key] = _normalize_schedule(raw[key])
            elif key == "objective_weights" and key in raw:
                values[key] = _normalize_objective_weights(raw[key])
            elif key == "pid_gain_bounds" and key in raw:
                values[key] = _normalize_gain_bounds(raw[key])
            elif (
                key in {"malibo_classifier_config", "malibo_train_config"}
                and key in raw
            ):
                values[key] = _normalize_dict(raw[key])
            elif key == "source_molweights" and key in raw:
                values[key] = _normalize_float_sequence(raw[key])
            elif (
                key
                in {
                    "ctm_pi_f_p_bounds",
                    "ctm_pi_f_i_bounds",
                    "ctm_pi_f_d_bounds",
                    "ctm_pid_f_p_bounds",
                    "ctm_pid_f_i_bounds",
                    "ctm_pid_f_d_bounds",
                }
                and key in raw
            ):
                values[key] = _normalize_two_float_tuple(raw[key])
            elif key == "initial_position_valve":
                if "initial_position_valve" in raw:
                    values[key] = raw["initial_position_valve"]
                elif "position_valve" in raw:
                    values[key] = raw["position_valve"]
                else:
                    values[key] = getattr(cls, key)
            else:
                values[key] = raw.get(key, getattr(cls, key))
        return cls(**values)  # type: ignore[arg-type]

    def as_env_defaults(self) -> Dict[str, Any]:
        """Return default environment configuration values."""
        return {
            "p_goal": float(self.initial_pressure_chamb),
            "v_speed": float(self.speed_valve),
            "sample_rate": float(self.sample_rate),
            "time_step": float(self.time_step),
            "max_steps": int(self.max_steps),
            "position_valve": float(self.position_valve),
            "volume_chamb": float(self.volume_chamb),
            "molweight_chamb": float(self.molweight_chamb),
            "temperature_chamb": float(self.temperature_chamb),
            "rho_pump": float(self.rho_pump),
        }

    @property
    def initial_pressure(self) -> float:
        return float(self.initial_pressure_chamb)

    @property
    def reference_schedule(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(duration), float(target))
            for duration, target in self.reference_curve_schedule
        )

    @property
    def objective_weight_map(self) -> dict[str, float]:
        return {str(name): float(weight) for name, weight in self.objective_weights}

    @property
    def position_valve(self) -> float:
        return float(self.initial_position_valve)


def _parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    if "\n" not in value:
        for comment_marker in ("//", "#"):
            if comment_marker in value:
                value = value.split(comment_marker, 1)[0].strip()
        if not value:
            return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _normalize_key(key: str) -> str:
    return key.strip().replace(" ", "_").replace("-", "_").lower()


def _strip_json_comments(payload: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    idx = 0
    length = len(payload)

    while idx < length:
        char = payload[idx]

        if escape:
            result.append(char)
            escape = False
            idx += 1
            continue

        if char == "\\":
            if in_string:
                escape = True
            result.append(char)
            idx += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            idx += 1
            continue

        if (
            not in_string
            and char == "/"
            and idx + 1 < length
            and payload[idx + 1] == "/"
        ):
            idx += 2
            while idx < length and payload[idx] not in {"\n", "\r"}:
                idx += 1
            continue

        result.append(char)
        idx += 1

    return "".join(result)


def _normalize_schedule(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid schedule JSON: {value}") from exc
    else:
        parsed = value

    if parsed is None:
        return tuple()

    result: list[tuple[float, float]] = []
    for entry in parsed:
        if isinstance(entry, dict):
            duration = entry.get("duration")
            target = entry.get("target")
        else:
            duration, target = entry
        if duration is None or target is None:
            raise ValueError(f"Malformed schedule entry: {entry}")
        result.append((float(duration), float(target)))
    return tuple(result)


def _normalize_gain_bounds(value: Any) -> GainBounds | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid PID bound JSON: {value}") from exc
    else:
        parsed = value

    if parsed is None:
        return None

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"PID bounds must be length-2 sequence, got {parsed!r}")

    rows: list[tuple[float, float, float]] = []
    for row in parsed:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"PID bound row must have 3 entries, got {row!r}")
        rows.append((float(row[0]), float(row[1]), float(row[2])))

    return (rows[0], rows[1])


def _normalize_objective_weights(value: Any) -> tuple[tuple[str, float], ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid objective-weight JSON: {value}") from exc
    else:
        parsed = value

    if parsed is None:
        return tuple()

    if isinstance(parsed, dict):
        items = parsed.items()
    else:
        items = parsed

    result: list[tuple[str, float]] = []
    for entry in items:
        if isinstance(entry, dict):
            name = entry.get("name")
            weight = entry.get("weight")
        else:
            name, weight = entry
        if name is None or weight is None:
            raise ValueError(f"Malformed objective-weight entry: {entry}")
        result.append((str(name), float(weight)))
    return tuple(result)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON dictionary: {value}") from exc
    elif isinstance(value, dict):
        parsed = value
    else:
        raise ValueError(f"Expected dict-like value, got {value!r}")
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Normalized value is not a dict: {parsed!r}")
    return parsed


def _normalize_float_sequence(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid float sequence JSON: {value}") from exc
    else:
        parsed = value

    if parsed is None:
        return tuple()

    try:
        return tuple(float(v) for v in parsed)
    except Exception as exc:  # noqa: BLE001 - broad for robust parsing
        raise ValueError(f"Malformed float sequence: {parsed!r}") from exc


def _normalize_two_float_tuple(value: Any) -> tuple[float, float]:
    """Normalize a two-element float tuple (e.g., CTM bounds)."""
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_comments(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid two-float tuple JSON: {value}") from exc
    else:
        parsed = value

    if parsed is None:
        return (0.0, 0.0)

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"Expected 2-element sequence, got {parsed!r}")

    return (float(parsed[0]), float(parsed[1]))


def _needs_multiline(value: str) -> bool:
    if not value:
        return False
    if value in {"[", "{"}:
        return True
    bracket_balance = value.count("[") - value.count("]")
    brace_balance = value.count("{") - value.count("}")
    return bracket_balance > 0 or brace_balance > 0


def _collect_multiline_value(initial: str, lines_iter: Iterator[str]) -> str:
    combined = [initial]
    bracket_balance = initial.count("[") - initial.count("]")
    brace_balance = initial.count("{") - initial.count("}")

    while bracket_balance > 0 or brace_balance > 0:
        try:
            next_line = next(lines_iter)
        except StopIteration as exc:
            raise ValueError("Unterminated multi-line configuration value") from exc
        stripped_next = next_line.strip()
        if not stripped_next:
            continue
        combined.append(stripped_next)
        bracket_balance += stripped_next.count("[") - stripped_next.count("]")
        brace_balance += stripped_next.count("{") - stripped_next.count("}")

    return "\n".join(combined)


def _load_raw_settings(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Global settings file not found: {path}")

    parsed: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        lines_iter = iter(handle)
        for line in lines_iter:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            if ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", maxsplit=1)
            normalized_key = _normalize_key(key)
            value = raw_value.strip()
            if _needs_multiline(value):
                value = _collect_multiline_value(value, lines_iter)
            parsed[normalized_key] = _parse_value(value)

    alias_map = {
        "pressure_chamb": "initial_pressure_chamb",
        "position_valve": "initial_position_valve",
    }
    for alias, canonical in alias_map.items():
        if alias in parsed and canonical not in parsed:
            parsed[canonical] = parsed[alias]
    return parsed


@lru_cache(maxsize=1)
def get_simulation_globals(path: Path | None = None) -> SimulationGlobals:
    """Load and cache the simulation-wide global settings."""

    settings_path = path if path is not None else DEFAULT_SETTINGS_PATH
    raw = _load_raw_settings(settings_path)
    return SimulationGlobals.from_mapping(raw)


def reload_simulation_globals(path: Path | None = None) -> SimulationGlobals:
    """Clear the cache and reload the simulation globals from disk."""

    get_simulation_globals.cache_clear()  # type: ignore[attr-defined]
    return get_simulation_globals(path)
