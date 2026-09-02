from pathlib import Path

from .define import PosData, PosWithTime
from .resolve import resolve

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "three"


def smooth_trajectory(
    trajectory: list[PosWithTime],
    window: int = 3,
) -> list[PosWithTime]:
    """Apply a centered moving average to a 10 Hz trajectory.

    The timestamps are preserved. At the two ends, the available points are
    averaged instead of padding the trajectory with artificial coordinates.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    if not trajectory:
        return []

    radius = window // 2
    result = []
    for index, item in enumerate(trajectory):
        left = max(0, index - radius)
        right = min(len(trajectory), index + radius + 1)
        points = trajectory[left:right]
        result.append(
            PosWithTime(
                item.time,
                PosData(
                    sum(point.pos.posx for point in points) / len(points),
                    sum(point.pos.posy for point in points) / len(points),
                ),
            )
        )
    return result


def plot_10hz_trajectory(
    trajectory: list[PosWithTime] | None = None,
    save_path: str | None = None,
    show: bool = False,
    smooth_window: int | None = None,
    show_raw: bool = True,
):
    """Plot the 10 Hz fused trajectory and coordinate-time curves.

    Pass an existing trajectory to avoid running the solver again. When
    ``save_path`` is provided, the requested file plus editable SVG/PDF and
    600 dpi TIFF companions are written under ``outputs/three``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Nature-style defaults: editable vector text, compact journal typography,
    # and a restrained palette that remains legible when printed in grayscale.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )

    if trajectory is None:
        trajectory, _ = resolve()
    if not trajectory:
        raise ValueError("The 10 Hz trajectory is empty")

    smoothed = (
        smooth_trajectory(trajectory, smooth_window)
        if smooth_window is not None
        else None
    )

    times = [item.time for item in trajectory]
    xs = [item.pos.posx for item in trajectory]
    ys = [item.pos.posy for item in trajectory]

    width_mm = 183
    height_mm = 82
    figure = plt.figure(
        figsize=(width_mm / 25.4, height_mm / 25.4), constrained_layout=True
    )
    grid = GridSpec(1, 3, figure=figure, width_ratios=(1.35, 1, 1), wspace=0.28)
    axes = np.asarray(
        [figure.add_subplot(grid[0, index]) for index in range(3)], dtype=object
    )

    raw_color = "#66788a"
    signal_color = "#007c83"
    grid_color = "#d9e0e5"
    accent_start = "#2f7d5a"
    accent_end = "#b24c4c"
    raw_style = {"color": raw_color, "linewidth": 0.65, "alpha": 0.55, "zorder": 1}
    signal_style = {"color": signal_color, "linewidth": 1.45, "zorder": 2}

    for axis, label in zip(axes, ("a", "b", "c")):
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        axis.grid(True, color=grid_color, linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.7, colors="#39434d")
        for spine in axis.spines.values():
            spine.set_color("#39434d")
            spine.set_linewidth(0.8)

    if show_raw:
        axes[0].plot(xs, ys, label="10 Hz fused", **raw_style)
    if smoothed is not None:
        axes[0].plot(
            [item.pos.posx for item in smoothed],
            [item.pos.posy for item in smoothed],
            label=f"smoothed ({smooth_window}-point)",
            **signal_style,
        )
    displayed = smoothed if smoothed is not None and not show_raw else trajectory
    axes[0].scatter(
        displayed[0].pos.posx,
        displayed[0].pos.posy,
        color=accent_start,
        s=24,
        edgecolor="white",
        linewidth=0.7,
        label="start",
        zorder=3,
    )
    axes[0].scatter(
        displayed[-1].pos.posx,
        displayed[-1].pos.posy,
        color=accent_end,
        s=24,
        edgecolor="white",
        linewidth=0.7,
        label="end",
        zorder=3,
    )
    axes[0].annotate(
        "Start",
        (displayed[0].pos.posx, displayed[0].pos.posy),
        xytext=(7, -11),
        textcoords="offset points",
        color=accent_start,
        fontsize=6,
        fontweight="bold",
    )
    axes[0].annotate(
        "End",
        (displayed[-1].pos.posx, displayed[-1].pos.posy),
        xytext=(7, 8),
        textcoords="offset points",
        color=accent_end,
        fontsize=6,
        fontweight="bold",
    )
    axes[0].set_title("Fused trajectory", loc="left", pad=8, fontweight="bold")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    axes[0].axis("equal")

    if show_raw:
        axes[1].plot(times, xs, label="10 Hz fused", **raw_style)
    if smoothed is not None:
        axes[1].plot(
            [item.time for item in smoothed],
            [item.pos.posx for item in smoothed],
            label=f"smoothed ({smooth_window}-point)",
            **signal_style,
        )
    axes[1].set_title("X over time", loc="left", pad=8, fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("X coordinate")

    if show_raw:
        axes[2].plot(times, ys, label="10 Hz fused", **raw_style)
    if smoothed is not None:
        axes[2].plot(
            [item.time for item in smoothed],
            [item.pos.posy for item in smoothed],
            label=f"smoothed ({smooth_window}-point)",
            **signal_style,
        )
    axes[2].set_title("Y over time", loc="left", pad=8, fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y coordinate")

    if smoothed is not None:
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 1.08),
            ncol=len(labels),
            frameon=False,
            handlelength=2.4,
        )
    if save_path is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / Path(save_path).name
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        stem = output_path.with_suffix("")
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
