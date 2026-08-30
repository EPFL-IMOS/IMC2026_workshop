"""Fast run-to-failure simulation of a closed-loop degrading CSTR.

Fast mass/energy balances are coupled to two slow latent health states:
catalyst activity and heat-transfer fouling resistance.  PICID receives only
measured process/controller signals and known operating context.  Product
concentration, conversion, both health states, and the virtual reference test
remain privileged simulator diagnostics.

The mechanism structure is literature-inspired, not calibrated to a plant.
Exact stochastic laws, exponents, distributions, schedules, and the virtual
failure test are synthetic tutorial choices; see CSTR_DEGRADATION_LITERATURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd


CSTR_FEATURE_COLUMNS = (
    "reactor_temperature",
    "coolant_temperature",
    "feed_flow",
    "feed_concentration",
    "feed_temperature",
    "temperature_setpoint",
)


@dataclass(frozen=True)
class CSTRParameters:
    """Nominal physical, degradation, controller, and sampling parameters."""

    dt_minutes: float = 0.25
    volume: float = 1.0
    feed_concentration: float = 1.0
    feed_temperature: float = 330.0
    nominal_feed_flow: float = 0.10

    reference_temperature: float = 350.0
    reaction_rate_at_reference: float = 0.32
    reaction_activation_over_r: float = 2_500.0
    heat_release: float = 55.0
    heat_transfer_rate: float = 0.040

    temperature_setpoint: float = 350.0
    coolant_bias: float = 300.0
    coolant_min: float = 280.0
    coolant_max: float = 345.0
    controller_kp: float = 2.0
    controller_ki: float = 0.05
    controller_integral_limit: float = 300.0

    deactivation_rate_at_reference: float = 0.0065
    deactivation_activation_over_r: float = 4_200.0
    deactivation_activity_order: float = 1.10
    deactivation_flow_exponent: float = 0.55
    deactivation_concentration_exponent: float = 0.45
    deactivation_process_noise: float = 0.06

    fouling_rate_at_reference: float = 0.0090
    fouling_activation_over_r: float = 3_200.0
    fouling_flow_exponent: float = 0.35
    fouling_reaction_exponent: float = 0.45
    fouling_process_noise: float = 0.08
    max_fouling_resistance: float = 2.5

    reference_min_conversion: float = 0.60
    reference_required_capacity: float = 0.105
    reference_flow_min: float = 0.030
    reference_flow_max: float = 0.180
    reference_flow_grid_size: int = 301
    failure_hold_steps: int = 5
    failure_check_start: int = 60
    equilibration_steps: int = 240
    max_steps: int = 720
    regime_min_steps: int = 20
    regime_max_steps: int = 44

    temperature_measurement_noise: float = 0.15
    coolant_measurement_noise: float = 0.10
    flow_measurement_noise: float = 0.0005
    feed_concentration_measurement_noise: float = 0.002
    feed_temperature_measurement_noise: float = 0.08

    def __post_init__(self) -> None:
        if self.dt_minutes <= 0:
            raise ValueError("dt_minutes must be positive")
        if self.volume <= 0 or self.nominal_feed_flow <= 0:
            raise ValueError("volume and nominal_feed_flow must be positive")
        if not 0 < self.reference_min_conversion < 1:
            raise ValueError("reference_min_conversion must lie between 0 and 1")
        if not self.reference_flow_min < self.reference_required_capacity:
            raise ValueError("reference_required_capacity must exceed flow_min")
        if not self.reference_required_capacity < self.reference_flow_max:
            raise ValueError("reference_required_capacity must be below flow_max")
        if self.reference_flow_grid_size < 3:
            raise ValueError("reference_flow_grid_size must be at least three")
        if self.max_fouling_resistance <= 0:
            raise ValueError("max_fouling_resistance must be positive")
        if self.deactivation_rate_at_reference <= 0:
            raise ValueError("deactivation_rate_at_reference must be positive")
        if self.fouling_rate_at_reference <= 0:
            raise ValueError("fouling_rate_at_reference must be positive")
        if self.failure_hold_steps < 1 or self.failure_check_start < 0:
            raise ValueError("failure timing parameters are invalid")
        if self.regime_min_steps < 1:
            raise ValueError("regime_min_steps must be positive")
        if self.regime_max_steps < self.regime_min_steps:
            raise ValueError("regime_max_steps must be >= regime_min_steps")


@dataclass(frozen=True)
class CSTRUnitParameters:
    """Unit-to-unit variability applied to the shared reactor parameters."""

    deactivation_multiplier: float = 1.0
    fouling_multiplier: float = 1.0
    reaction_multiplier: float = 1.0
    heat_transfer_multiplier: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.deactivation_multiplier,
            self.fouling_multiplier,
            self.reaction_multiplier,
            self.heat_transfer_multiplier,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all unit multipliers must be positive")


@dataclass(frozen=True)
class OperatingSchedule:
    """Piecewise operating inputs used during one simulated lifetime."""

    feed_flow: np.ndarray
    feed_concentration: np.ndarray
    feed_temperature: np.ndarray
    temperature_setpoint: np.ndarray
    regime_id: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.feed_flow),
            len(self.feed_concentration),
            len(self.feed_temperature),
            len(self.temperature_setpoint),
            len(self.regime_id),
        }
        if len(lengths) != 1:
            raise ValueError("operating schedule arrays must have equal length")


@dataclass
class CSTRUnitResult:
    """One complete run-to-failure trajectory and its simulation metadata."""

    unit_name: str
    trajectory: pd.DataFrame
    parameters: CSTRParameters
    unit_parameters: CSTRUnitParameters
    seed: int

    @property
    def failure_step(self) -> int:
        return len(self.trajectory) - 1

    @property
    def failure_time_minutes(self) -> float:
        return float(self.trajectory["time_minutes"].iloc[-1])

    def model_frame(self) -> pd.DataFrame:
        """Return only observable features and the RUL target for PICID."""

        return self.trajectory.loc[
            :, [*CSTR_FEATURE_COLUMNS, "rul"]
        ].astype(np.float32)


@dataclass
class CSTRFleet:
    """A named collection of independently simulated reactor units."""

    units: dict[str, CSTRUnitResult] = field(default_factory=dict)

    def model_frames(self) -> dict[str, pd.DataFrame]:
        """Return PICID-ready frames without latent diagnostic variables."""

        return {name: result.model_frame() for name, result in self.units.items()}

    def export_model_csvs(self, output_dir: str | Path) -> dict[str, Path]:
        """Persist one PICID-facing CSV per unit and return their paths.

        Only the six observable/context channels and the RUL target are written.
        A manifest records file names and shapes without exposing latent simulator
        states or product-quality diagnostics.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_paths: dict[str, Path] = {}
        manifest_rows = []
        for unit_name, frame in self.model_frames().items():
            csv_path = output_dir / f"{unit_name}.csv"
            frame.to_csv(csv_path, index=False, float_format="%.9g")
            csv_paths[unit_name] = csv_path
            manifest_rows.append(
                {
                    "unit": unit_name,
                    "csv_file": csv_path.name,
                    "time_steps": len(frame),
                    "feature_columns": len(CSTR_FEATURE_COLUMNS),
                    "target_column": "rul",
                }
            )

        pd.DataFrame(manifest_rows).to_csv(
            output_dir / "manifest.csv",
            index=False,
        )
        return csv_paths

    def summary(self) -> pd.DataFrame:
        """Return one compact row per simulated run-to-failure unit."""

        rows = []
        for name, result in self.units.items():
            trajectory = result.trajectory
            rows.append(
                {
                    "unit": name,
                    "time_steps": len(trajectory),
                    "failure_time_minutes": result.failure_time_minutes,
                    "deactivation_multiplier": (
                        result.unit_parameters.deactivation_multiplier
                    ),
                    "fouling_multiplier": result.unit_parameters.fouling_multiplier,
                    "final_catalyst_activity": float(
                        trajectory["catalyst_activity"].iloc[-1]
                    ),
                    "final_fouling_resistance": float(
                        trajectory["fouling_resistance"].iloc[-1]
                    ),
                    "final_reference_capacity": float(
                        trajectory["reference_capacity"].iloc[-1]
                    ),
                    "feature_columns": len(CSTR_FEATURE_COLUMNS),
                }
            )
        return pd.DataFrame(rows)


def _temperature_rate(
    rate_at_reference: float,
    activation_over_r: float,
    temperature: float,
    reference_temperature: float,
) -> float:
    """Evaluate an Arrhenius rate relative to a numerically stable reference."""

    exponent = -activation_over_r * (
        1.0 / temperature - 1.0 / reference_temperature
    )
    return float(rate_at_reference * np.exp(np.clip(exponent, -20.0, 20.0)))


def make_operating_schedule(
    *,
    parameters: CSTRParameters,
    rng: np.random.Generator,
) -> OperatingSchedule:
    """Create seeded multivariable regimes plus smooth measured disturbances."""

    flow_levels = np.array([0.86, 1.13, 0.95, 1.06, 0.90, 1.16])
    concentration_levels = np.array([1.08, 0.94, 0.90, 1.10, 1.01, 0.97])
    temperature_offsets = np.array([2.0, -3.0, 1.0, 0.0, -1.5, 3.0])
    setpoint_offsets = np.array([-2.0, 2.5, 1.0, -1.5, 3.0, 0.0])
    feed_flow = np.empty(parameters.max_steps, dtype=float)
    feed_concentration = np.empty(parameters.max_steps, dtype=float)
    feed_temperature = np.empty(parameters.max_steps, dtype=float)
    temperature_setpoint = np.empty(parameters.max_steps, dtype=float)
    regime_id = np.empty(parameters.max_steps, dtype=int)

    cursor = 0
    previous_regime = -1
    while cursor < parameters.max_steps:
        choices = [index for index in range(len(flow_levels)) if index != previous_regime]
        regime = int(rng.choice(choices))
        duration = int(
            rng.integers(
                parameters.regime_min_steps,
                parameters.regime_max_steps + 1,
            )
        )
        stop = min(cursor + duration, parameters.max_steps)
        feed_flow[cursor:stop] = (
            parameters.nominal_feed_flow * flow_levels[regime]
        )
        feed_concentration[cursor:stop] = (
            parameters.feed_concentration * concentration_levels[regime]
        )
        feed_temperature[cursor:stop] = (
            parameters.feed_temperature + temperature_offsets[regime]
        )
        temperature_setpoint[cursor:stop] = (
            parameters.temperature_setpoint + setpoint_offsets[regime]
        )
        regime_id[cursor:stop] = regime
        previous_regime = regime
        cursor = stop

    # Smooth, measured disturbances prevent every transition from being a step
    # while retaining all operating causes in the exported context channels.
    n_disturbances = max(3, parameters.max_steps // 100)
    for _ in range(n_disturbances):
        width = int(rng.integers(8, 18))
        start = int(rng.integers(0, parameters.max_steps - width))
        stop = start + width
        pulse = np.sin(np.linspace(0.0, np.pi, width))
        feed_flow[start:stop] += rng.uniform(-0.006, 0.006) * pulse
        feed_concentration[start:stop] += rng.uniform(-0.035, 0.035) * pulse
        feed_temperature[start:stop] += rng.uniform(-2.0, 2.0) * pulse

    return OperatingSchedule(
        feed_flow=feed_flow,
        feed_concentration=feed_concentration,
        feed_temperature=feed_temperature,
        temperature_setpoint=temperature_setpoint,
        regime_id=regime_id,
    )


def _controller_coolant_temperature(
    state: np.ndarray,
    parameters: CSTRParameters,
    temperature_setpoint: float,
) -> float:
    reactor_temperature = float(state[1])
    integral_error = float(state[4])
    error = reactor_temperature - temperature_setpoint
    unconstrained = (
        parameters.coolant_bias
        - parameters.controller_kp * error
        - parameters.controller_ki * integral_error
    )
    return float(
        np.clip(
            unconstrained,
            parameters.coolant_min,
            parameters.coolant_max,
        )
    )


def _state_derivative(
    state: np.ndarray,
    *,
    feed_flow: float,
    feed_concentration: float,
    feed_temperature: float,
    temperature_setpoint: float,
    parameters: CSTRParameters,
    unit_parameters: CSTRUnitParameters,
    deactivation_noise_multiplier: float,
    fouling_noise_multiplier: float,
    enable_degradation: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    concentration, temperature, activity, fouling_resistance, _ = state
    coolant_temperature = _controller_coolant_temperature(
        state,
        parameters,
        temperature_setpoint,
    )

    reaction_rate_constant = _temperature_rate(
        parameters.reaction_rate_at_reference,
        parameters.reaction_activation_over_r,
        temperature,
        parameters.reference_temperature,
    )
    reaction_rate = (
        activity
        * unit_parameters.reaction_multiplier
        * reaction_rate_constant
        * concentration
    )

    dilution_rate = feed_flow / parameters.volume
    concentration_derivative = (
        dilution_rate * (feed_concentration - concentration)
        - reaction_rate
    )
    effective_heat_transfer_rate = (
        parameters.heat_transfer_rate
        * unit_parameters.heat_transfer_multiplier
        / (1.0 + max(fouling_resistance, 0.0))
    )
    temperature_derivative = (
        dilution_rate * (feed_temperature - temperature)
        + parameters.heat_release * reaction_rate
        + effective_heat_transfer_rate * (coolant_temperature - temperature)
    )

    if enable_degradation:
        deactivation_rate_constant = _temperature_rate(
            parameters.deactivation_rate_at_reference,
            parameters.deactivation_activation_over_r,
            temperature,
            parameters.reference_temperature,
        )
        flow_stress = (
            feed_flow / parameters.nominal_feed_flow
        ) ** parameters.deactivation_flow_exponent
        concentration_stress = (
            feed_concentration / parameters.feed_concentration
        ) ** parameters.deactivation_concentration_exponent
        activity_derivative = -(
            unit_parameters.deactivation_multiplier
            * deactivation_noise_multiplier
            * deactivation_rate_constant
            * flow_stress
            * concentration_stress
            * max(activity, 0.0) ** parameters.deactivation_activity_order
        )

        fouling_rate_constant = _temperature_rate(
            parameters.fouling_rate_at_reference,
            parameters.fouling_activation_over_r,
            temperature,
            parameters.reference_temperature,
        )
        reaction_stress = max(
            reaction_rate
            / (parameters.reaction_rate_at_reference * parameters.feed_concentration),
            0.05,
        ) ** parameters.fouling_reaction_exponent
        fouling_flow_stress = (
            feed_flow / parameters.nominal_feed_flow
        ) ** parameters.fouling_flow_exponent
        fouling_derivative = (
            unit_parameters.fouling_multiplier
            * fouling_noise_multiplier
            * fouling_rate_constant
            * fouling_flow_stress
            * reaction_stress
        )
    else:
        activity_derivative = 0.0
        fouling_derivative = 0.0

    integral_derivative = temperature - temperature_setpoint
    derivatives = np.array(
        [
            concentration_derivative,
            temperature_derivative,
            activity_derivative,
            fouling_derivative,
            integral_derivative,
        ],
        dtype=float,
    )
    diagnostics = {
        "coolant_temperature": coolant_temperature,
        "reaction_rate": float(reaction_rate),
        "deactivation_rate": float(-activity_derivative),
        "fouling_rate": float(fouling_derivative),
        "effective_heat_transfer_rate": float(effective_heat_transfer_rate),
        "controller_saturated": bool(
            np.isclose(coolant_temperature, parameters.coolant_min)
            or np.isclose(coolant_temperature, parameters.coolant_max)
        ),
    }
    return derivatives, diagnostics


def _rk4_step(
    state: np.ndarray,
    *,
    feed_flow: float,
    feed_concentration: float,
    feed_temperature: float,
    temperature_setpoint: float,
    parameters: CSTRParameters,
    unit_parameters: CSTRUnitParameters,
    deactivation_noise_multiplier: float,
    fouling_noise_multiplier: float,
    enable_degradation: bool,
) -> np.ndarray:
    kwargs = {
        "feed_flow": feed_flow,
        "feed_concentration": feed_concentration,
        "feed_temperature": feed_temperature,
        "temperature_setpoint": temperature_setpoint,
        "parameters": parameters,
        "unit_parameters": unit_parameters,
        "deactivation_noise_multiplier": deactivation_noise_multiplier,
        "fouling_noise_multiplier": fouling_noise_multiplier,
        "enable_degradation": enable_degradation,
    }
    dt = parameters.dt_minutes
    k1, _ = _state_derivative(state, **kwargs)
    k2, _ = _state_derivative(state + 0.5 * dt * k1, **kwargs)
    k3, _ = _state_derivative(state + 0.5 * dt * k2, **kwargs)
    k4, _ = _state_derivative(state + dt * k3, **kwargs)
    next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    next_state[0] = np.clip(next_state[0], 0.0, 1.5 * feed_concentration)
    next_state[1] = np.clip(next_state[1], 270.0, 450.0)
    next_state[2] = np.clip(next_state[2], 0.0, 1.0)
    next_state[3] = np.clip(next_state[3], 0.0, parameters.max_fouling_resistance)
    next_state[4] = np.clip(
        next_state[4],
        -parameters.controller_integral_limit,
        parameters.controller_integral_limit,
    )
    return next_state


def _equilibrate_clean_reactor(
    *,
    parameters: CSTRParameters,
    unit_parameters: CSTRUnitParameters,
) -> np.ndarray:
    state = np.array(
        [0.25, parameters.temperature_setpoint, 1.0, 0.0, 0.0],
        dtype=float,
    )
    for _ in range(parameters.equilibration_steps):
        state = _rk4_step(
            state,
            feed_flow=parameters.nominal_feed_flow,
            feed_concentration=parameters.feed_concentration,
            feed_temperature=parameters.feed_temperature,
            temperature_setpoint=parameters.temperature_setpoint,
            parameters=parameters,
            unit_parameters=unit_parameters,
            deactivation_noise_multiplier=1.0,
            fouling_noise_multiplier=1.0,
            enable_degradation=False,
        )
    state[2] = 1.0
    state[3] = 0.0
    return state


def evaluate_reference_capacity(
    *,
    catalyst_activity: float,
    fouling_resistance: float,
    parameters: CSTRParameters,
    unit_parameters: CSTRUnitParameters,
) -> dict[str, float | bool]:
    """Evaluate hidden production capacity under common reference conditions.

    The test assumes steady operation at the reference temperature and scans
    feed flow.  A candidate flow is feasible only when it meets the reference
    conversion requirement and the corresponding steady coolant command lies
    inside the actuator limits.  Nothing returned here is passed to PICID.
    """

    flows = np.linspace(
        parameters.reference_flow_min,
        parameters.reference_flow_max,
        parameters.reference_flow_grid_size,
    )
    dilution_rates = flows / parameters.volume
    reaction_rate_constant = _temperature_rate(
        parameters.reaction_rate_at_reference,
        parameters.reaction_activation_over_r,
        parameters.reference_temperature,
        parameters.reference_temperature,
    )
    effective_rate = (
        max(catalyst_activity, 0.0)
        * unit_parameters.reaction_multiplier
        * reaction_rate_constant
    )
    steady_concentration = (
        dilution_rates
        * parameters.feed_concentration
        / (dilution_rates + effective_rate)
    )
    conversion = 1.0 - steady_concentration / parameters.feed_concentration
    reaction_rate = effective_rate * steady_concentration
    effective_heat_transfer_rate = (
        parameters.heat_transfer_rate
        * unit_parameters.heat_transfer_multiplier
        / (1.0 + max(fouling_resistance, 0.0))
    )
    coolant_required = parameters.reference_temperature - (
        dilution_rates
        * (parameters.feed_temperature - parameters.reference_temperature)
        + parameters.heat_release * reaction_rate
    ) / effective_heat_transfer_rate
    feasible = (
        (conversion >= parameters.reference_min_conversion)
        & (coolant_required >= parameters.coolant_min)
        & (coolant_required <= parameters.coolant_max)
    )
    capacity = float(flows[feasible].max()) if np.any(feasible) else 0.0

    required_index = int(
        np.argmin(np.abs(flows - parameters.reference_required_capacity))
    )
    return {
        "capacity": capacity,
        "capacity_margin": capacity - parameters.reference_required_capacity,
        "conversion_at_required_flow": float(conversion[required_index]),
        "coolant_at_required_flow": float(coolant_required[required_index]),
        "passes": bool(capacity >= parameters.reference_required_capacity),
    }


def simulate_cstr_unit(
    unit_name: str,
    *,
    seed: int,
    parameters: CSTRParameters | None = None,
    unit_parameters: CSTRUnitParameters | None = None,
) -> CSTRUnitResult:
    """Simulate one unit until the hidden reference-capacity test fails."""

    parameters = parameters or CSTRParameters()
    unit_parameters = unit_parameters or CSTRUnitParameters()
    rng = np.random.default_rng(seed)
    schedule = make_operating_schedule(parameters=parameters, rng=rng)
    state = _equilibrate_clean_reactor(
        parameters=parameters,
        unit_parameters=unit_parameters,
    )

    rows: list[dict[str, float | int | bool]] = []
    consecutive_failures = 0
    for step in range(parameters.max_steps):
        feed_flow = float(schedule.feed_flow[step])
        feed_concentration = float(schedule.feed_concentration[step])
        feed_temperature = float(schedule.feed_temperature[step])
        temperature_setpoint = float(schedule.temperature_setpoint[step])
        deactivation_noise = float(
            np.exp(
                parameters.deactivation_process_noise * rng.normal()
                - 0.5 * parameters.deactivation_process_noise**2
            )
        )
        fouling_noise = float(
            np.exp(
                parameters.fouling_process_noise * rng.normal()
                - 0.5 * parameters.fouling_process_noise**2
            )
        )
        _, diagnostics = _state_derivative(
            state,
            feed_flow=feed_flow,
            feed_concentration=feed_concentration,
            feed_temperature=feed_temperature,
            temperature_setpoint=temperature_setpoint,
            parameters=parameters,
            unit_parameters=unit_parameters,
            deactivation_noise_multiplier=deactivation_noise,
            fouling_noise_multiplier=fouling_noise,
            enable_degradation=True,
        )
        concentration, temperature, activity, fouling_resistance, _ = state
        conversion = 1.0 - concentration / feed_concentration
        reference_test = evaluate_reference_capacity(
            catalyst_activity=float(activity),
            fouling_resistance=float(fouling_resistance),
            parameters=parameters,
            unit_parameters=unit_parameters,
        )

        if step >= parameters.failure_check_start:
            if not reference_test["passes"]:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
        is_failure = consecutive_failures >= parameters.failure_hold_steps

        rows.append(
            {
                "time_minutes": step * parameters.dt_minutes,
                "reactor_temperature": temperature
                + rng.normal(0.0, parameters.temperature_measurement_noise),
                "coolant_temperature": diagnostics["coolant_temperature"]
                + rng.normal(0.0, parameters.coolant_measurement_noise),
                "feed_flow": feed_flow
                + rng.normal(0.0, parameters.flow_measurement_noise),
                "feed_concentration": feed_concentration
                + rng.normal(
                    0.0,
                    parameters.feed_concentration_measurement_noise,
                ),
                "feed_temperature": feed_temperature
                + rng.normal(0.0, parameters.feed_temperature_measurement_noise),
                "temperature_setpoint": temperature_setpoint,
                "outlet_concentration": concentration,
                "catalyst_activity": activity,
                "fouling_resistance": fouling_resistance,
                "conversion": conversion,
                "reaction_rate": diagnostics["reaction_rate"],
                "deactivation_rate": diagnostics["deactivation_rate"],
                "fouling_rate": diagnostics["fouling_rate"],
                "effective_heat_transfer_rate": diagnostics[
                    "effective_heat_transfer_rate"
                ],
                "controller_saturated": diagnostics["controller_saturated"],
                "reference_capacity": reference_test["capacity"],
                "reference_capacity_margin": reference_test["capacity_margin"],
                "reference_conversion": reference_test[
                    "conversion_at_required_flow"
                ],
                "reference_coolant_temperature": reference_test[
                    "coolant_at_required_flow"
                ],
                "regime_id": int(schedule.regime_id[step]),
                "is_failure": is_failure,
            }
        )
        if is_failure:
            break

        state = _rk4_step(
            state,
            feed_flow=feed_flow,
            feed_concentration=feed_concentration,
            feed_temperature=feed_temperature,
            temperature_setpoint=temperature_setpoint,
            parameters=parameters,
            unit_parameters=unit_parameters,
            deactivation_noise_multiplier=deactivation_noise,
            fouling_noise_multiplier=fouling_noise,
            enable_degradation=True,
        )
    else:
        raise RuntimeError(
            f"{unit_name} did not fail the reference-capacity test within "
            f"{parameters.max_steps} steps"
        )

    trajectory = pd.DataFrame(rows)
    failure_time = float(trajectory["time_minutes"].iloc[-1])
    trajectory["rul"] = failure_time - trajectory["time_minutes"]
    return CSTRUnitResult(
        unit_name=unit_name,
        trajectory=trajectory,
        parameters=parameters,
        unit_parameters=unit_parameters,
        seed=seed,
    )


def simulate_cstr_fleet(
    *,
    n_units: int = 3,
    seed: int = 7,
    parameters: CSTRParameters | None = None,
) -> CSTRFleet:
    """Simulate a reproducible ragged fleet through one high-level call."""

    if n_units < 1:
        raise ValueError("n_units must be positive")
    parameters = parameters or CSTRParameters()
    rng = np.random.default_rng(seed)
    nominal_deactivation = np.linspace(1.18, 0.82, n_units)
    nominal_fouling = np.linspace(0.82, 1.18, n_units)
    units: dict[str, CSTRUnitResult] = {}
    for unit_index, (deactivation_multiplier, fouling_multiplier) in enumerate(
        zip(nominal_deactivation, nominal_fouling, strict=True)
    ):
        unit_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        unit_parameters = CSTRUnitParameters(
            deactivation_multiplier=float(
                deactivation_multiplier * rng.normal(1.0, 0.025)
            ),
            fouling_multiplier=float(
                fouling_multiplier * rng.normal(1.0, 0.025)
            ),
            reaction_multiplier=float(rng.normal(1.0, 0.025)),
            heat_transfer_multiplier=float(rng.normal(1.0, 0.020)),
        )
        unit_name = f"reactor_{unit_index + 1:02d}"
        units[unit_name] = simulate_cstr_unit(
            unit_name,
            seed=unit_seed,
            parameters=replace(parameters),
            unit_parameters=unit_parameters,
        )
    return CSTRFleet(units=units)
