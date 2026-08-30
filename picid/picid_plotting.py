"""Plotting helpers for PICID workshop input, training, and evaluation traces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from picid_training import CheckpointUnroll, PreparedExperiment, TrainingRun


FIT_COLOR = "#D13A3A"
BAYESIAN_COLOR = "#E88924"
FIT_POINT_COLOR = "#E3A0A0"
BAYESIAN_POINT_COLOR = "#F5BA73"
TREE_COLOR = "#168B76"
DEEP_COLOR = "#212121"
CUSTOM_COLOR = "#3F72AF"
VALIDATION_COLOR = "#687078"
GRID_COLOR = "#D8DADD"
SPLIT_COLORS = {
    "train": DEEP_COLOR,
    "val": FIT_COLOR,
    "test": VALIDATION_COLOR,
}


def _prepare_output(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _style_trace_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _epoch_metric(metrics: pd.DataFrame, column: str) -> pd.Series:
    """Return the last finite value logged for each epoch."""

    if column not in metrics:
        return pd.Series(dtype=float)
    values = metrics.loc[metrics[column].notna(), ["epoch", column]].copy()
    values["epoch"] = pd.to_numeric(values["epoch"], errors="coerce")
    values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna()
    return values.groupby("epoch", sort=True)[column].last()


def _finite_values(metrics: pd.DataFrame, column: str) -> pd.Series:
    """Return the finite values of a sparsely logged metric column."""

    if column not in metrics:
        return pd.Series(dtype=float)
    return pd.to_numeric(metrics[column], errors="coerce").dropna()


def _training_epoch_count(metrics: pd.DataFrame) -> int:
    """Return the number of epochs with an aggregated training loss."""

    return len(_epoch_metric(metrics, "train/loss_epoch"))


def plot_unit_data(
    unit_frames: Mapping[str, pd.DataFrame],
    unit_names_by_split: Mapping[str, list[str]],
) -> plt.Figure:
    """Plot every raw trajectory and show its whole-unit split assignment."""

    unit_names = list(unit_frames)
    n_units = len(unit_names)
    figure, axes = plt.subplots(
        n_units,
        2,
        figsize=(12, 1.85 * n_units),
        sharex=False,
        squeeze=False,
    )
    signal_columns = [
        column for column in unit_frames[unit_names[0]].columns if column != "rul"
    ]
    split_by_unit = {
        unit_name: split
        for split, split_unit_names in unit_names_by_split.items()
        for unit_name in split_unit_names
    }
    if set(split_by_unit) != set(unit_names):
        raise ValueError("Every plotted unit must be assigned to exactly one split.")

    for row_index, unit_name in enumerate(unit_names):
        frame = unit_frames[unit_name]
        time_step = np.arange(len(frame))
        split = split_by_unit[unit_name]
        split_color = SPLIT_COLORS[split]
        display_features = frame.loc[:, signal_columns]
        display_range = (display_features.max() - display_features.min()).replace(
            0.0,
            1.0,
        )
        display_features = (
            display_features - display_features.min()
        ) / display_range

        signal_axis, rul_axis = axes[row_index]
        for column in signal_columns:
            signal_axis.plot(
                time_step,
                display_features[column],
                linewidth=1.1,
                label=column,
            )
        signal_axis.set_ylabel(f"{unit_name}\n[{split}]")
        signal_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)

        rul_axis.plot(time_step, frame["rul"], color=FIT_COLOR, linewidth=1.6)
        for axis in (signal_axis, rul_axis):
            axis.set_facecolor((*to_rgb(split_color), 0.055))
            axis.spines[["top", "right"]].set_visible(False)

        if row_index == 0:
            signal_axis.set_title("Observed features (display-scaled per channel)")
            signal_axis.legend(ncol=2, frameon=False, fontsize=7)
            rul_axis.set_title("RUL target; the complete row stays in one split")
        if row_index == n_units - 1:
            signal_axis.set_xlabel("time step")
            rul_axis.set_xlabel("time step")

    figure.suptitle(
        "Simulated CSTR run-to-failure fleet — whole-unit train/validation/test split",
        fontweight="bold",
        y=0.997,
    )
    figure.legend(
        handles=[
            Patch(facecolor=color, alpha=0.20, label=split)
            for split, color in SPLIT_COLORS.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    return figure


def save_unit_data_trace(
    unit_frames: Mapping[str, pd.DataFrame],
    unit_names_by_split: Mapping[str, list[str]],
    output_path: Path,
) -> Path:
    """Persist the raw unit trajectories for notebook and report display."""

    output_path = _prepare_output(output_path)
    figure = plot_unit_data(unit_frames, unit_names_by_split)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_input_trace(
    prepared: PreparedExperiment,
    output_path: Path,
) -> Path:
    """Plot ragged split lengths alongside PICID's payload schema status."""

    output_path = _prepare_output(output_path)
    report = prepared.split_alignment_report
    splits = [split for split in ("train", "val", "test") if split in report["splits"]]
    report_keys = [row["key"] for row in report["rows"]]
    status_to_int = {"empty": 0, "homogeneous": 1, "heterogeneous": -1}

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(report_keys) + 2.5))
    x = np.arange(len(prepared.unit_names))
    split_by_unit = {
        unit_name: split
        for split, split_unit_names in prepared.unit_names_by_split.items()
        for unit_name in split_unit_names
    }
    lengths = [
        prepared.unit_lengths_by_split[split_by_unit[name]][name]
        for name in prepared.unit_names
    ]
    colors = [SPLIT_COLORS[split_by_unit[name]] for name in prepared.unit_names]
    axes[0].bar(x, lengths, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prepared.unit_names, rotation=25, ha="right")
    axes[0].set_ylabel("time steps")
    axes[0].set_title("Whole-unit split preserves ragged lifetimes")
    axes[0].legend(
        handles=[Patch(facecolor=SPLIT_COLORS[split], label=split) for split in splits],
        frameon=False,
    )

    status_matrix = np.array(
        [
            [status_to_int[row["schema_status"][split]] for split in splits]
            for row in report["rows"]
        ]
    )
    axes[1].imshow(status_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(len(splits)))
    axes[1].set_xticklabels(splits)
    axes[1].set_yticks(range(len(report_keys)))
    axes[1].set_yticklabels(report_keys)
    for row_index, row in enumerate(report["rows"]):
        for split_index, split in enumerate(splits):
            axes[1].text(
                split_index,
                row_index,
                row["schema_status"][split],
                ha="center",
                va="center",
                fontsize=8,
            )
    axes[1].set_title("Schema status")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_optimization_trace(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    seed: int,
) -> Path:
    """Plot loss, validation metrics, and learning rate by epoch."""

    output_path = _prepare_output(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    train_loss = _epoch_metric(metrics, "train/loss_epoch")
    val_loss = _epoch_metric(metrics, "val/loss")
    axes[0].plot(train_loss.index + 1, train_loss, color=DEEP_COLOR, label="train")
    axes[0].plot(val_loss.index + 1, val_loss, color=FIT_COLOR, label="validation")
    axes[0].set(title="Objective", xlabel="epoch", ylabel="normalized loss")
    axes[0].legend(frameon=False)

    for metric_name, label, color in [
        ("val/mae_normalized", "MAE", FIT_COLOR),
        ("val/rmse_normalized", "RMSE", DEEP_COLOR),
    ]:
        values = _epoch_metric(metrics, metric_name)
        axes[1].plot(values.index + 1, values, color=color, label=label)
    axes[1].set(title="Validation task metrics", xlabel="epoch", ylabel="error")
    axes[1].legend(frameon=False)

    learning_rate = _finite_values(metrics, "lr-AdamW")
    axes[2].plot(
        np.arange(1, len(learning_rate) + 1),
        learning_rate.to_numpy(),
        color=FIT_COLOR,
        marker="o",
        markersize=2.5,
    )
    axes[2].set(title="Optimizer schedule", xlabel="epoch", ylabel="learning rate")
    for axis in axes:
        _style_trace_axis(axis)
    fig.suptitle(
        (
            "MLP optimization trace — "
            f"{_training_epoch_count(metrics)} training epochs, seed {seed}"
        ),
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_system_trace(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Plot epoch duration, throughput, and mean phase batch latency."""

    output_path = _prepare_output(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    epoch_duration = _epoch_metric(metrics, "time/train_epoch_duration_sec")
    axes[0].plot(
        epoch_duration.index + 1,
        epoch_duration,
        color=DEEP_COLOR,
        marker="o",
        markersize=2.5,
    )
    axes[0].set(title="Training duration", xlabel="epoch", ylabel="seconds")

    throughput = _finite_values(metrics, "efficiency/train_throughput")
    axes[1].plot(
        np.arange(1, len(throughput) + 1),
        throughput.to_numpy(),
        color=FIT_COLOR,
        linewidth=1.1,
    )
    axes[1].axhline(
        throughput.median(),
        color=VALIDATION_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"median {throughput.median():.0f}",
    )
    axes[1].set(
        title="Training throughput",
        xlabel="logged batch",
        ylabel="samples / s",
    )
    axes[1].legend(frameon=False)

    batch_time_candidates = [
        ("train", "AvgTime/train_batch_mean_ms", 1.0),
        ("validation", "AvgTime/val_batch_mean", 1000.0),
        ("test", "efficiency/inference_latency_mean_ms", 1.0),
    ]
    batch_labels = []
    batch_times = []
    for label, column, scale in batch_time_candidates:
        values = _finite_values(metrics, column)
        if not values.empty:
            batch_labels.append(label)
            batch_times.append(float(values.iloc[-1]) * scale)
    axes[2].bar(
        batch_labels,
        batch_times,
        color=[DEEP_COLOR, VALIDATION_COLOR, FIT_COLOR][: len(batch_labels)],
    )
    axes[2].set(title="Mean batch time", ylabel="milliseconds")
    for axis in axes:
        _style_trace_axis(axis)
    fig.suptitle("MLP system trace — CPU run with ResourceTracker", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_custom_unet_trace(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    seed: int,
    max_epochs: int,
) -> Path:
    """Plot the custom U-Net objective, validation errors, and epoch timing."""

    output_path = _prepare_output(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    train_loss = _epoch_metric(metrics, "train/loss_epoch")
    val_loss = _epoch_metric(metrics, "val/loss")
    axes[0].plot(
        train_loss.index + 1,
        train_loss,
        color=CUSTOM_COLOR,
        label="train",
    )
    axes[0].plot(val_loss.index + 1, val_loss, color=FIT_COLOR, label="validation")
    axes[0].set(title="Objective", xlabel="epoch", ylabel="normalized loss")
    axes[0].legend(frameon=False)

    for metric_name, label, color in [
        ("val/mae_normalized", "MAE", FIT_COLOR),
        ("val/rmse_normalized", "RMSE", CUSTOM_COLOR),
    ]:
        values = _epoch_metric(metrics, metric_name)
        axes[1].plot(values.index + 1, values, color=color, label=label)
    axes[1].set(title="Validation task metrics", xlabel="epoch", ylabel="error")
    axes[1].legend(frameon=False)

    epoch_duration = _epoch_metric(metrics, "time/train_epoch_duration_sec")
    axes[2].plot(
        epoch_duration.index + 1,
        epoch_duration,
        color=CUSTOM_COLOR,
        marker="o",
        markersize=2.5,
    )
    if not epoch_duration.empty:
        axes[2].axhline(
            epoch_duration.median(),
            color=VALIDATION_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"median {epoch_duration.median():.2f} s",
        )
        axes[2].legend(frameon=False)
    axes[2].set(title="Training duration", xlabel="epoch", ylabel="seconds")

    for axis in axes:
        _style_trace_axis(axis)
    training_epochs = _training_epoch_count(metrics)
    stop_label = " (early stop)" if training_epochs < max_epochs else ""
    fig.suptitle(
        f"Custom 1D U-Net trace — {training_epochs} epochs{stop_label}, seed {seed}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_evaluation_trace(result_frame: pd.DataFrame, output_path: Path) -> Path:
    """Compare aggregate normalized test errors returned by all seven branches."""

    output_path = _prepare_output(output_path)
    metric_columns = [
        ("test/mae_normalized", "MAE"),
        ("test/rmse_normalized", "RMSE"),
        ("test/mse_normalized", "MSE"),
    ]
    metric_labels = [label for _, label in metric_columns]
    positions = np.arange(len(metric_columns))
    model_styles = [
        ("XGBoost", FIT_POINT_COLOR),
        ("Bayesian ridge", BAYESIAN_POINT_COLOR),
        ("XGBoost (24-step)", FIT_COLOR),
        ("Bayesian ridge (24-step)", BAYESIAN_COLOR),
        ("Decision tree (24-step)", TREE_COLOR),
        ("MLP", DEEP_COLOR),
        ("custom U-Net", CUSTOM_COLOR),
    ]
    bar_width = 0.11
    fig, axis = plt.subplots(figsize=(12.0, 4.5))
    for model_index, (family, color) in enumerate(model_styles):
        row = result_frame.loc[result_frame["family"] == family].iloc[0]
        values = [float(row[column]) for column, _ in metric_columns]
        offset = (model_index - (len(model_styles) - 1) / 2) * bar_width
        bars = axis.bar(
            positions + offset,
            values,
            bar_width,
            label=family,
            color=color,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
    axis.set_xticks(positions, metric_labels)
    axis.set_ylabel("normalized error (lower is better)")
    axis.set_title(
        "Returned test metrics from the same simulated CSTR protocol",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, fontsize=8)
    _style_trace_axis(axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_checkpoint_unroll_trace(
    checkpoint_unroll: CheckpointUnroll,
    output_path: Path,
    *,
    split: str,
) -> Path:
    """Plot all persisted-model predictions across every unit in one split."""

    output_path = _prepare_output(output_path)
    split_windows = checkpoint_unroll.windows.loc[
        checkpoint_unroll.windows["split"] == split
    ]
    unit_names = split_windows["unit"].drop_duplicates().tolist()
    if not unit_names:
        raise ValueError(f"Checkpoint unroll contains no units for split={split!r}.")

    n_columns = 2 if len(unit_names) > 3 else 1
    n_rows = int(np.ceil(len(unit_names) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3.0 * n_rows),
        squeeze=False,
    )
    for plot_index, unit_name in enumerate(unit_names):
        row_index, column_index = divmod(plot_index, n_columns)
        axis = axes[row_index, column_index]
        unit_windows = split_windows.loc[split_windows["unit"] == unit_name]
        axis.plot(
            unit_windows["time_index"],
            unit_windows["target"],
            color="#000000",
            linewidth=2.4,
            label="target",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["constant_prediction"],
            color=VALIDATION_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="training-mean constant",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["xgboost_prediction"],
            color=FIT_POINT_COLOR,
            linestyle=":",
            linewidth=1.0,
            label="XGBoost pointwise",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["bayesian_prediction"],
            color=BAYESIAN_POINT_COLOR,
            linestyle=":",
            linewidth=1.0,
            label="Bayesian pointwise",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["xgboost_24_step_prediction"],
            color=FIT_COLOR,
            linewidth=1.25,
            label="XGBoost 24-step",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["bayesian_24_step_prediction"],
            color=BAYESIAN_COLOR,
            linewidth=1.25,
            label="Bayesian 24-step",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["decision_tree_24_step_prediction"],
            color=TREE_COLOR,
            linewidth=1.15,
            label="Decision tree 24-step",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["mlp_prediction"],
            color=DEEP_COLOR,
            linewidth=1.4,
            label="MLP checkpoint",
        )
        axis.plot(
            unit_windows["time_index"],
            unit_windows["custom_unet_prediction"],
            color=CUSTOM_COLOR,
            linewidth=1.5,
            label="U-Net checkpoint",
        )

        unit_metrics = checkpoint_unroll.unit_summary.loc[
            (checkpoint_unroll.unit_summary["split"] == split)
            & (checkpoint_unroll.unit_summary["unit"] == unit_name)
        ].set_index("model")
        best_model = unit_metrics["skill_vs_constant"].idxmax()
        best_skill = unit_metrics.loc[best_model, "skill_vs_constant"]
        axis.set_title(
            f"{unit_name}: best {best_model}, skill {best_skill:+.1%}",
            loc="left",
        )
        axis.set_ylabel("normalized RUL")
        plotted_columns = [
            "target",
            "constant_prediction",
            "xgboost_prediction",
            "bayesian_prediction",
            "xgboost_24_step_prediction",
            "bayesian_24_step_prediction",
            "decision_tree_24_step_prediction",
            "mlp_prediction",
            "custom_unet_prediction",
        ]
        axis.set_ylim(
            min(-0.08, float(unit_windows[plotted_columns].min().min()) - 0.04),
            max(1.08, float(unit_windows[plotted_columns].max().max()) + 0.04),
        )
        _style_trace_axis(axis)
        if row_index == n_rows - 1:
            axis.set_xlabel("time step at end of input window")

    for axis in axes.ravel()[len(unit_names) :]:
        axis.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        f"Persisted-model lifetime unroll — {split} units",
        fontweight="bold",
        y=1.005,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_run_traces(
    prepared: PreparedExperiment,
    run: TrainingRun,
    checkpoint_unroll: CheckpointUnroll,
    *,
    asset_dir: Path,
    artifact_dir: Path,
    seed: int,
) -> dict[str, Path]:
    """Persist run tables and generate the workshop trace figures."""

    asset_dir = Path(asset_dir)
    artifact_dir = Path(artifact_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(run.deep_metrics_path)
    custom_unet_metrics = pd.read_csv(run.custom_unet_metrics_path)
    metrics.to_csv(artifact_dir / "deep_learning_metrics.csv", index=False)
    custom_unet_metrics.to_csv(
        artifact_dir / "custom_unet_metrics.csv",
        index=False,
    )
    run.result_frame.to_csv(artifact_dir / "returned_test_metrics.csv", index=False)
    checkpoint_unroll.windows.to_csv(
        artifact_dir / "checkpoint_unroll_windows.csv",
        index=False,
    )
    checkpoint_unroll.unit_summary.to_csv(
        artifact_dir / "checkpoint_unroll_by_unit.csv",
        index=False,
    )
    checkpoint_unroll.split_summary.to_csv(
        artifact_dir / "checkpoint_unroll_by_split.csv",
        index=False,
    )

    trace_paths = {
        "input": save_input_trace(prepared, asset_dir / "picid_input_trace.png"),
        "optimization": save_optimization_trace(
            metrics,
            asset_dir / "picid_optimization_trace.png",
            seed=seed,
        ),
        "system": save_system_trace(metrics, asset_dir / "picid_system_trace.png"),
        "custom U-Net": save_custom_unet_trace(
            custom_unet_metrics,
            asset_dir / "picid_custom_unet_trace.png",
            seed=seed,
            max_epochs=run.deep_learning_epochs,
        ),
        "evaluation": save_evaluation_trace(
            run.result_frame,
            asset_dir / "picid_evaluation_trace.png",
        ),
        "validation unroll": save_checkpoint_unroll_trace(
            checkpoint_unroll,
            asset_dir / "picid_checkpoint_unroll_validation.png",
            split="val",
        ),
        "test unroll": save_checkpoint_unroll_trace(
            checkpoint_unroll,
            asset_dir / "picid_checkpoint_unroll_test.png",
            split="test",
        ),
    }
    return trace_paths
