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
    ``save_path`` is provided, the figure is saved under ``outputs/two``
    using the supplied file name.
    """
    import matplotlib.pyplot as plt

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

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    if show_raw:
        axes[0].plot(xs, ys, linewidth=1.0, label="10 Hz fused trajectory")
    if smoothed is not None:
        axes[0].plot(
            [item.pos.posx for item in smoothed],
            [item.pos.posy for item in smoothed],
            linewidth=1.4,
            label=f"smoothed ({smooth_window}-point)",
        )
    displayed = smoothed if smoothed is not None and not show_raw else trajectory
    axes[0].scatter(
        displayed[0].pos.posx,
        displayed[0].pos.posy,
        color="green",
        s=40,
        label="start",
        zorder=3,
    )
    axes[0].scatter(
        displayed[-1].pos.posx,
        displayed[-1].pos.posy,
        color="red",
        s=40,
        label="end",
        zorder=3,
    )
    axes[0].set_title("Fused trajectory")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if show_raw:
        axes[1].plot(times, xs, linewidth=0.8, label="raw")
    if smoothed is not None:
        axes[1].plot(
            [item.time for item in smoothed],
            [item.pos.posx for item in smoothed],
            linewidth=1.0,
            label=f"smoothed ({smooth_window}-point)",
        )
    axes[1].set_title("X over time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("X")
    axes[1].grid(True, alpha=0.3)

    if show_raw:
        axes[2].plot(times, ys, linewidth=0.8, label="raw")
    if smoothed is not None:
        axes[2].plot(
            [item.time for item in smoothed],
            [item.pos.posy for item in smoothed],
            linewidth=1.0,
            label=f"smoothed ({smooth_window}-point)",
        )
    axes[2].set_title("Y over time")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y")
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    if save_path is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / Path(save_path).name
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
