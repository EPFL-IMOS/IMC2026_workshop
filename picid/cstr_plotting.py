"""Plotting and schema helpers for the degrading-CSTR simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

from cstr_simulator import CSTR_FEATURE_COLUMNS, CSTRFleet, CSTRUnitResult


RED = "#D13A3A"
DARK = "#212121"
BLUE = "#3F72AF"
GRAY = "#687078"
LIGHT = "#F5F6F7"
GRID = "#D8DADD"
REGIME_COLORS = {
    0: "#D9E4F2",
    1: "#E5E5E5",
    2: "#FAD6D6",
    3: "#E1EED8",
    4: "#F8E5C2",
    5: "#E7DDF2",
}


def _prepare_output(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_reactor_schema() -> plt.Figure:
    """Draw the reactor, controller, hidden health states, and data boundary."""

    figure, axis = plt.subplots(figsize=(12.0, 5.7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")

    # Vessel and coolant jacket.
    jacket = FancyBboxPatch(
        (4.3, 1.25),
        3.4,
        4.35,
        boxstyle="round,pad=0.15,rounding_size=0.65",
        linewidth=2.0,
        edgecolor=BLUE,
        facecolor="#EAF2FA",
    )
    vessel = FancyBboxPatch(
        (4.65, 1.55),
        2.7,
        3.75,
        boxstyle="round,pad=0.10,rounding_size=0.55",
        linewidth=2.2,
        edgecolor=DARK,
        facecolor="white",
    )
    liquid = Rectangle(
        (4.75, 1.68),
        2.5,
        2.75,
        facecolor="#F7DADA",
        edgecolor="none",
        alpha=0.85,
    )
    surface = Ellipse(
        (6.0, 4.43),
        2.5,
        0.32,
        facecolor="#F7DADA",
        edgecolor=RED,
        linewidth=1.0,
    )
    axis.add_patch(jacket)
    axis.add_patch(vessel)
    axis.add_patch(liquid)
    axis.add_patch(surface)

    # Agitator and catalyst sites.
    axis.plot([6.0, 6.0], [4.55, 5.95], color=DARK, linewidth=2.0)
    axis.plot([5.35, 6.65], [2.55, 2.55], color=DARK, linewidth=2.3)
    axis.plot([5.55, 6.0], [2.25, 2.55], color=DARK, linewidth=1.5)
    axis.plot([6.45, 6.0], [2.25, 2.55], color=DARK, linewidth=1.5)
    motor = FancyBboxPatch(
        (5.55, 5.75),
        0.9,
        0.55,
        boxstyle="round,pad=0.05",
        facecolor=LIGHT,
        edgecolor=DARK,
        linewidth=1.3,
    )
    axis.add_patch(motor)
    axis.text(6.0, 6.03, "motor", ha="center", va="center", fontsize=9)
    for x, y, alpha in [
        (5.25, 3.05, 1.0),
        (5.75, 3.65, 0.85),
        (6.35, 3.15, 0.70),
        (6.75, 3.75, 0.55),
        (5.15, 3.90, 0.40),
    ]:
        axis.add_patch(
            Circle((x, y), 0.09, facecolor=RED, edgecolor="none", alpha=alpha)
        )

    # Material and coolant flows.
    arrow = dict(arrowstyle="-|>", linewidth=2.0, color=DARK, mutation_scale=14)
    axis.annotate("", xy=(4.65, 4.65), xytext=(1.1, 4.65), arrowprops=arrow)
    axis.text(1.15, 4.95, "feed", color=DARK, fontsize=11, fontweight="bold")
    axis.text(1.15, 4.43, r"$C_{A,f},\ T_f,\ q(t)$", color=GRAY, fontsize=10)

    axis.annotate("", xy=(10.9, 2.15), xytext=(7.35, 2.15), arrowprops=arrow)
    axis.text(9.1, 2.46, "product", color=DARK, fontsize=11, fontweight="bold")
    axis.text(9.1, 1.87, r"$C_A(t),\ T(t)$", color=GRAY, fontsize=10)

    coolant_arrow = dict(
        arrowstyle="-|>", linewidth=2.0, color=BLUE, mutation_scale=14
    )
    axis.annotate("", xy=(4.25, 1.85), xytext=(2.2, 1.85), arrowprops=coolant_arrow)
    axis.text(2.2, 2.15, "coolant in", color=BLUE, fontsize=10, fontweight="bold")
    axis.text(2.2, 1.55, r"$T_c(t)$", color=BLUE, fontsize=10)
    axis.annotate("", xy=(9.7, 5.05), xytext=(7.75, 5.05), arrowprops=coolant_arrow)
    axis.text(8.25, 5.35, "coolant out", color=BLUE, fontsize=10)

    # Measurement and feedback-control path.
    sensor = Circle((7.95, 3.85), 0.32, facecolor="white", edgecolor=RED, linewidth=1.7)
    axis.add_patch(sensor)
    axis.text(7.95, 3.85, "T", color=RED, ha="center", va="center", fontweight="bold")
    axis.plot([7.35, 7.63], [3.85, 3.85], color=RED, linewidth=1.5)
    controller = FancyBboxPatch(
        (9.0, 3.40),
        1.55,
        0.90,
        boxstyle="round,pad=0.08",
        facecolor=LIGHT,
        edgecolor=DARK,
        linewidth=1.4,
    )
    axis.add_patch(controller)
    axis.text(9.78, 3.92, "PI controller", ha="center", va="center", fontsize=10)
    axis.text(9.78, 3.60, r"varying $T_{set}(t)$", ha="center", va="center", fontsize=8, color=GRAY)
    axis.annotate(
        "",
        xy=(9.0, 3.85),
        xytext=(8.28, 3.85),
        arrowprops=dict(arrowstyle="->", linewidth=1.4, color=RED),
    )
    axis.annotate(
        "",
        xy=(3.25, 1.85),
        xytext=(9.78, 3.40),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.4,
            color=BLUE,
            connectionstyle="angle3,angleA=-90,angleB=0",
        ),
    )

    axis.text(6.0, 3.65, r"reaction $r=a\,k(T)C_A$", ha="center", fontsize=10)
    axis.text(
        6.0,
        0.78,
        r"latent health: catalyst activity $a\downarrow$  +  fouling resistance $R_f\uparrow$",
        ha="center",
        fontsize=11,
        color=RED,
        fontweight="bold",
    )
    axis.text(
        6.0,
        0.35,
        r"$r=a\,k(T)C_A$;  $U_{eff}=U_{clean}/(1+R_f)$; product quality and health stay hidden",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    axis.set_title(
        "Closed-loop CSTR with two slow degradation mechanisms",
        fontsize=15,
        fontweight="bold",
        pad=8,
    )
    figure.tight_layout()
    return figure


def save_reactor_schema(output_path: Path) -> Path:
    """Save the vector-style reactor schema as a high-resolution PNG."""

    output_path = _prepare_output(output_path)
    figure = plot_reactor_schema()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _shade_regimes(axis: plt.Axes, result: CSTRUnitResult) -> None:
    trajectory = result.trajectory
    regimes = trajectory["regime_id"].to_numpy(dtype=int)
    time = trajectory["time_minutes"].to_numpy()
    start = 0
    for index in range(1, len(regimes) + 1):
        if index == len(regimes) or regimes[index] != regimes[start]:
            axis.axvspan(
                time[start],
                time[index - 1],
                color=REGIME_COLORS[regimes[start]],
                alpha=0.35,
                linewidth=0,
            )
            start = index


def plot_degradation_trace(result: CSTRUnitResult) -> plt.Figure:
    """Explain hidden health, capacity failure, control, and operating context."""

    frame = result.trajectory
    time = frame["time_minutes"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True)

    heat_transfer_fraction = 1.0 / (1.0 + frame["fouling_resistance"])
    axes[0, 0].plot(
        time,
        frame["catalyst_activity"],
        color=RED,
        linewidth=1.8,
        label="catalyst activity",
    )
    axes[0, 0].plot(
        time,
        heat_transfer_fraction,
        color=BLUE,
        linewidth=1.8,
        label="heat-transfer fraction",
    )
    axes[0, 0].set(title="Two latent health states", ylabel="fraction of clean state")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(time, frame["reference_capacity"], color=DARK, linewidth=1.6)
    axes[0, 1].axhline(
        result.parameters.reference_required_capacity,
        color=RED,
        linestyle="--",
        label="required capacity",
    )
    axes[0, 1].set(title="Hidden virtual reference test", ylabel="maximum feasible feed flow")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        time,
        frame["reactor_temperature"],
        color=DARK,
        label="reactor T",
    )
    axes[1, 0].plot(
        time,
        frame["coolant_temperature"],
        color=BLUE,
        label="controller output T_c",
    )
    axes[1, 0].plot(
        time,
        frame["temperature_setpoint"],
        color=RED,
        linestyle=":",
        linewidth=1.0,
        label="setpoint",
    )
    axes[1, 0].set(title="Feedback redistributes the signature", ylabel="kelvin")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8)

    _shade_regimes(axes[1, 1], result)
    operating_columns = [
        "feed_flow",
        "feed_concentration",
        "feed_temperature",
        "temperature_setpoint",
    ]
    operating = frame.loc[:, operating_columns]
    operating_range = (operating.max() - operating.min()).replace(0.0, 1.0)
    operating = (operating - operating.min()) / operating_range
    for column, color in zip(
        operating_columns,
        [DARK, RED, BLUE, GRAY],
        strict=True,
    ):
        axes[1, 1].plot(
            time,
            operating[column],
            color=color,
            linewidth=1.0,
            label=column.replace("_", " "),
        )
    axes[1, 1].set(title="Known operating context", ylabel="display-scaled")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)

    for axis in axes.flat:
        _style_axis(axis)
    for axis in axes[1, :]:
        axis.set_xlabel("time [min]")
    figure.suptitle(
        f"Degradation-to-failure trace — {result.unit_name}",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def save_degradation_trace(result: CSTRUnitResult, output_path: Path) -> Path:
    """Save a degradation-mechanism figure for one representative unit."""

    output_path = _prepare_output(output_path)
    figure = plot_degradation_trace(result)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_fleet_overview(fleet: CSTRFleet) -> plt.Figure:
    """Compare two latent mechanisms, capacity, and emergent lifetimes."""

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    axes = axes.ravel()
    colors = [RED, DARK, BLUE, GRAY]
    for index, (name, result) in enumerate(fleet.units.items()):
        frame = result.trajectory
        time = frame["time_minutes"]
        color = colors[index % len(colors)]
        axes[0].plot(time, frame["catalyst_activity"], color=color, label=name)
        axes[1].plot(time, frame["fouling_resistance"], color=color, label=name)
        axes[2].plot(time, frame["reference_capacity"], color=color, label=name)
    axes[0].set(title="Latent catalyst activity", xlabel="time [min]", ylabel="a(t)")
    axes[1].set(title="Latent thermal fouling", xlabel="time [min]", ylabel=r"$R_f(t)$")
    axes[2].axhline(
        next(iter(fleet.units.values())).parameters.reference_required_capacity,
        color=RED,
        linestyle="--",
        linewidth=1.0,
        label="required capacity",
    )
    axes[2].set(
        title="Hidden reference capacity",
        xlabel="time [min]",
        ylabel="maximum feasible feed flow",
    )

    summary = fleet.summary()
    bars = axes[3].bar(
        summary["unit"],
        summary["failure_time_minutes"],
        color=colors[: len(summary)],
    )
    axes[3].bar_label(bars, fmt="%.1f min", padding=3, fontsize=8)
    axes[3].set(title="Emergent ragged lifetimes", ylabel="failure time [min]")
    axes[3].tick_params(axis="x", rotation=25)

    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        _style_axis(axis)
    figure.suptitle(
        "Seeded unit variability and operating regimes produce unequal lifetimes",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def save_fleet_overview(fleet: CSTRFleet, output_path: Path) -> Path:
    """Save the fleet-level degradation and lifetime comparison."""

    output_path = _prepare_output(output_path)
    figure = plot_fleet_overview(fleet)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    sensitivity = np.mean(predictions[positive] == 1)
    specificity = np.mean(predictions[negative] == 0)
    return float(0.5 * (sensitivity + specificity))


def _fit_best_threshold(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    best_score = -np.inf
    best_threshold = float(candidates[0])
    best_direction = 1.0
    for threshold in candidates:
        for direction in (1.0, -1.0):
            predictions = (
                direction * values >= direction * threshold
            ).astype(int)
            score = _balanced_accuracy(labels, predictions)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_direction = direction
    return best_threshold, best_direction


def _spectral_frames(
    fleet: CSTRFleet,
    *,
    late_life_fraction: float,
    window: int,
    stride: int,
) -> dict[str, pd.DataFrame]:
    spectral_frames: dict[str, pd.DataFrame] = {}
    signal_columns = [
        "reactor_temperature",
        "coolant_temperature",
        "temperature_error",
    ]
    for unit_name, result in fleet.units.items():
        frame = result.model_frame().copy()
        frame["temperature_error"] = (
            frame["reactor_temperature"] - frame["temperature_setpoint"]
        )
        rows: list[dict[str, float | int]] = []
        for start in range(0, len(frame) - window + 1, stride):
            stop = start + window
            endpoint = stop - 1
            row: dict[str, float | int] = {
                "late_life": int(
                    frame["rul"].iloc[endpoint]
                    <= late_life_fraction * frame["rul"].iloc[0]
                )
            }
            for column in signal_columns:
                signal = frame[column].iloc[start:stop].to_numpy(dtype=float)
                signal = signal - np.mean(signal)
                power = np.abs(np.fft.rfft(signal)) ** 2 / window
                low_stop = max(2, len(power) // 4)
                total_power = np.sum(power[1:])
                frequencies = np.fft.rfftfreq(window)
                row[f"{column}: low-band power"] = float(
                    np.log1p(np.sum(power[1:low_stop]))
                )
                row[f"{column}: high-band power"] = float(
                    np.log1p(np.sum(power[low_stop:]))
                )
                row[f"{column}: spectral centroid"] = float(
                    np.sum(frequencies[1:] * power[1:])
                    / max(total_power, np.finfo(float).eps)
                )
            rows.append(row)
        spectral_frames[unit_name] = pd.DataFrame(rows)
    return spectral_frames


def threshold_detection_audit(
    fleet: CSTRFleet,
    *,
    late_life_fraction: float = 0.25,
    time_window: int = 64,
    spectral_window: int = 64,
    spectral_stride: int = 8,
) -> pd.DataFrame:
    """Cross-unit audit of fixed time- and frequency-domain thresholds.

    A threshold and its direction are selected on all but one unit and then
    evaluated on the held-out unit.  Labels mark the final fraction of each
    lifetime.  The audit intentionally tests simple scalar cutoffs, not every
    possible detector.
    """

    if not 0.0 < late_life_fraction < 0.5:
        raise ValueError("late_life_fraction must lie between zero and 0.5")
    if len(fleet.units) < 2:
        raise ValueError("threshold audit requires at least two units")
    if time_window < 2 or spectral_window < 4 or spectral_stride < 1:
        raise ValueError("threshold-audit window/stride parameters are invalid")

    time_frames: dict[str, pd.DataFrame] = {}
    for unit_name, result in fleet.units.items():
        frame = result.model_frame().copy()
        frame["temperature_error"] = (
            frame["reactor_temperature"] - frame["temperature_setpoint"]
        )
        for column in (
            "reactor_temperature",
            "coolant_temperature",
            "temperature_error",
        ):
            rolling = frame[column].rolling(time_window, min_periods=time_window)
            frame[f"{column}: rolling mean"] = rolling.mean()
            frame[f"{column}: rolling std"] = rolling.std()
            frame[f"{column}: rolling slope"] = (
                frame[column].diff(time_window - 1) / (time_window - 1)
            )
        frame["late_life"] = (
            frame["rul"] <= late_life_fraction * frame["rul"].iloc[0]
        ).astype(int)
        time_frames[unit_name] = frame

    frequency_frames = _spectral_frames(
        fleet,
        late_life_fraction=late_life_fraction,
        window=spectral_window,
        stride=spectral_stride,
    )
    domains = {
        "time": (
            time_frames,
            [
                column
                for column in next(iter(time_frames.values())).columns
                if column not in {"rul", "late_life"}
            ],
        ),
        "frequency": (
            frequency_frames,
            [
                column
                for column in next(iter(frequency_frames.values())).columns
                if column != "late_life"
            ],
        ),
    }

    rows: list[dict[str, float | str]] = []
    for domain, (frames, features) in domains.items():
        for feature in features:
            fold_scores: list[float] = []
            for held_out_name, held_out in frames.items():
                train = pd.concat(
                    [frame for name, frame in frames.items() if name != held_out_name],
                    ignore_index=True,
                )
                train_feature = train.loc[
                    train[feature].notna(),
                    [feature, "late_life"],
                ]
                held_out_feature = held_out.loc[
                    held_out[feature].notna(),
                    [feature, "late_life"],
                ]
                threshold, direction = _fit_best_threshold(
                    train_feature[feature].to_numpy(dtype=float),
                    train_feature["late_life"].to_numpy(dtype=int),
                )
                predictions = (
                    direction * held_out_feature[feature].to_numpy(dtype=float)
                    >= direction * threshold
                ).astype(int)
                fold_scores.append(
                    _balanced_accuracy(
                        held_out_feature["late_life"].to_numpy(dtype=int),
                        predictions,
                    )
                )
            rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "mean_balanced_accuracy": float(np.mean(fold_scores)),
                    "min_balanced_accuracy": float(np.min(fold_scores)),
                    "max_balanced_accuracy": float(np.max(fold_scores)),
                }
            )
    return pd.DataFrame(rows)


def plot_threshold_detection_audit(audit: pd.DataFrame) -> plt.Figure:
    """Plot the held-out performance of scalar time/frequency thresholds."""

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True)
    for axis, domain in zip(axes, ("time", "frequency"), strict=True):
        domain_audit = (
            audit.loc[audit["domain"] == domain]
            .nlargest(8, "mean_balanced_accuracy")
            .sort_values("mean_balanced_accuracy")
        )
        labels = [
            label.replace("temperature_", "T ")
            .replace("reactor_", "reactor ")
            .replace("coolant_", "coolant ")
            .replace("feed_", "feed ")
            .replace("_", " ")
            for label in domain_audit["feature"]
        ]
        scores = domain_audit["mean_balanced_accuracy"].to_numpy()
        lower = scores - domain_audit["min_balanced_accuracy"].to_numpy()
        upper = domain_audit["max_balanced_accuracy"].to_numpy() - scores
        axis.barh(
            labels,
            scores,
            xerr=np.vstack([lower, upper]),
            color=BLUE if domain == "frequency" else DARK,
            alpha=0.88,
            error_kw={"ecolor": GRAY, "capsize": 2.0, "linewidth": 0.8},
        )
        axis.axvline(0.5, color=RED, linestyle="--", linewidth=1.2, label="chance")
        axis.axvline(
            0.70,
            color=GRAY,
            linestyle=":",
            linewidth=1.2,
            label="mean-score guardrail",
        )
        axis.set(
            title=f"{domain.capitalize()}-domain scalar cutoffs",
            xlabel="held-out balanced accuracy",
            xlim=(0.0, 1.0),
        )
        axis.legend(frameon=False, fontsize=8)
        _style_axis(axis)
    figure.suptitle(
        "No fixed univariate threshold transfers reliably across reactor units",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def save_threshold_detection_audit(
    audit: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save the tutorial threshold-audit figure."""

    output_path = _prepare_output(output_path)
    figure = plot_threshold_detection_audit(audit)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path
