import csv
import math
from pathlib import Path

import numpy as np


def configure_plot_fonts() -> None:
    """Use Latin Modern Math for Latin/math glyphs and SimSun for Chinese."""
    import matplotlib as mpl

    mpl.rcParams["font.family"] = ["Latin Modern Math", "SimSun", "STSong"]
    mpl.rcParams["font.sans-serif"] = ["Latin Modern Math", "SimSun", "STSong"]
    # Route automatic tick labels through mathtext so negative values use the
    # mathematical minus glyph from Latin Modern Math instead of a text dash.
    mpl.rcParams["axes.unicode_minus"] = True
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Latin Modern Math"
    mpl.rcParams["mathtext.it"] = "Latin Modern Math"
    mpl.rcParams["mathtext.bf"] = "Latin Modern Math"
    mpl.rcParams["mathtext.cal"] = "Latin Modern Math"
    mpl.rcParams["mathtext.sf"] = "Latin Modern Math"
    mpl.rcParams["mathtext.tt"] = "Latin Modern Math"
    mpl.rcParams["mathtext.fallback"] = "stix"


def fixed_mathtext_formatter(precision: int):
    """Return a fixed-point formatter whose signs are rendered in mathtext."""
    from matplotlib.ticker import FormatStrFormatter

    if precision < 0:
        raise ValueError("precision must be non-negative")
    return FormatStrFormatter(f"$%.{precision}f$")


def chinese_font_properties():
    """Provide a CJK font for text objects that also contain mathtext."""
    from matplotlib.font_manager import FontProperties

    return FontProperties(family="SimSun")


def mathtext_number(value: float, precision: int) -> str:
    """Format a signed number for use inside a mixed-language mathtext label."""
    if precision < 0:
        raise ValueError("precision must be non-negative")
    return f"${value:.{precision}f}$"


def load_sensor_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.reader(Path(path).open(encoding="utf-8")))[1:]
    values = np.asarray([[float(cell) for cell in row[:3]] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(f"invalid sensor CSV: {path}")
    return values[:, 0], values[:, 1], values[:, 2]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    if window == 1:
        return values.copy()
    radius = window // 2
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        result[index] = values[left:right].mean()
    return result


def smooth_sensor(
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return times.copy(), moving_average(xs, window), moving_average(ys, window)


def comparison_times(
    t1: np.ndarray,
    t2: np.ndarray,
    delta: float,
    hz: float = 10.0,
) -> np.ndarray:
    start = max(float(t1[0]), float(t2[0] - delta))
    end = min(float(t1[-1]), float(t2[-1] - delta))
    if start > end:
        return np.empty(0, dtype=float)
    first = math.ceil(start * hz - 1e-9)
    last = math.floor(end * hz + 1e-9)
    if first > last:
        return np.empty(0, dtype=float)
    return np.arange(first, last + 1, dtype=float) / hz


def aligned_differences(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delta: float,
    hz: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = comparison_times(t1, t2, delta, hz)
    if len(times) == 0:
        return times, np.empty(0), np.empty(0)
    ix = np.interp(times, t1, x1)
    iy = np.interp(times, t1, y1)
    jx = np.interp(times + delta, t2, x2)
    jy = np.interp(times + delta, t2, y2)
    return times, jx - ix, jy - iy


def time_objective(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delta: float,
    min_fraction: float = 0.8,
) -> float:
    """Evaluate the time objective on sensor 1's original sampling instants.

    This deliberately differs from :func:`aligned_differences`, which uses a
    10 Hz grid for spatial comparison. The formal solvers estimate the time
    offset at sensor 1's observed instants and interpolate sensor 2; keeping
    that convention here makes the default sensitivity and formal results
    directly comparable.
    """
    start = max(float(t1[0]), float(t2[0] - delta))
    end = min(float(t1[-1]), float(t2[-1] - delta))
    mask = (t1 >= start) & (t1 <= end)
    if int(mask.sum()) < min_fraction * len(t1):
        return math.nan
    query = t1[mask] + delta
    dx = np.interp(query, t2, x2) - x1[mask]
    dy = np.interp(query, t2, y2) - y1[mask]
    return float(np.mean((dx - dx.mean()) ** 2 + (dy - dy.mean()) ** 2))


def estimate_delta_time(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    search_min: float,
    search_max: float,
    coarse_step: float = 0.1,
    fine_step: float = 0.001,
    min_fraction: float = 0.8,
) -> tuple[float, float]:
    coarse = np.arange(search_min, search_max, coarse_step)
    values = np.asarray(
        [
            time_objective(t1, x1, y1, t2, x2, y2, float(delta), min_fraction)
            for delta in coarse
        ]
    )
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("no valid time offset candidate")
    coarse_best = float(coarse[np.nanargmin(values)])
    fine = np.arange(
        coarse_best - coarse_step,
        coarse_best + coarse_step + fine_step / 2,
        fine_step,
    )
    fine_values = np.asarray(
        [
            time_objective(t1, x1, y1, t2, x2, y2, float(delta), min_fraction)
            for delta in fine
        ]
    )
    fine_valid = np.isfinite(fine_values)
    if not fine_valid.any():
        return coarse_best, float(values[np.nanargmin(values)])
    best_index = int(np.nanargmin(fine_values))
    return float(fine[best_index]), float(fine_values[best_index])


def block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    count = len(values) // block_size
    if count < 1:
        return np.empty(0)
    return values[: count * block_size].reshape(count, block_size).mean(axis=1)


def bias_confidence(
    dx: np.ndarray,
    dy: np.ndarray,
    block_size: int = 20,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    bx = block_means(dx, block_size)
    by = block_means(dy, block_size)
    if len(bx) < 2 or len(by) < 2:
        raise ValueError("at least two complete blocks are required")
    # Normal critical values are enough for the sensitivity comparison;
    # scipy is intentionally not required by the project.
    z_table = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    z = z_table.get(round(confidence, 2), 1.959964)
    mean_x, mean_y = float(bx.mean()), float(by.mean())
    se_x = float(bx.std(ddof=1) / math.sqrt(len(bx)))
    se_y = float(by.std(ddof=1) / math.sqrt(len(by)))
    x_low, x_high = mean_x - z * se_x, mean_x + z * se_x
    y_low, y_high = mean_y - z * se_y, mean_y + z * se_y
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "x_low": x_low,
        "x_high": x_high,
        "y_low": y_low,
        "y_high": y_high,
        "has_x_bias": bool(x_low > 0 or x_high < 0),
        "has_y_bias": bool(y_low > 0 or y_high < 0),
        "block_count": len(bx),
    }
