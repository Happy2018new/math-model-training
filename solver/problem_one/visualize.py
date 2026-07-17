"""Render the problem-one drone route as a modern route-planning graphic."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from resolve import INPUT_POINT_PATH, PROJECT_ROOT, Point, load_points, solve_tsp


OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "one" / "visual.png"
MAP_PATH = PROJECT_ROOT / "others" / "input_picture.png"
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1000
HORIZONTAL_PADDING = 110
TOP_PADDING = 205
BOTTOM_PADDING = 90
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

BACKGROUND = (250, 248, 247)
PANEL = (255, 255, 255)
GRID = (237, 235, 232)
TEXT = (38, 49, 65)
MUTED_TEXT = (119, 132, 150)
ROUTE_SHADOW = (231, 221, 210)
ROUTE = (205, 95, 50)
DEPOT = (35, 148, 245)
DELIVERY = (75, 165, 66)


def project_points(points: list[Point]) -> list[tuple[int, int]]:
    """Project mathematical coordinates with the same transform as the map."""
    map_image = cv2.imdecode(np.fromfile(MAP_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if map_image is None:
        raise FileNotFoundError(f"Cannot read map image: {MAP_PATH}")
    map_height, map_width = map_image.shape[:2]
    scale_x = (CANVAS_WIDTH - 2 * HORIZONTAL_PADDING) / map_width
    scale_y = (CANVAS_HEIGHT - TOP_PADDING - BOTTOM_PADDING) / map_height
    scale = min(scale_x, scale_y)
    offset_x = (CANVAS_WIDTH - scale * map_width) / 2
    offset_y = TOP_PADDING + (
        CANVAS_HEIGHT - TOP_PADDING - BOTTOM_PADDING - scale * map_height
    ) / 2

    return [
        (
            round(offset_x + scale * float(point["x_math"])),
            round(offset_y + scale * ((map_height - 1) - float(point["y_math"]))),
        )
        for point in points
    ]


def draw_grid(canvas: np.ndarray) -> None:
    """Draw a restrained background grid for spatial context."""
    for x in range(0, CANVAS_WIDTH, 120):
        cv2.line(canvas, (x, 0), (x, CANVAS_HEIGHT), GRID, 1)
    for y in range(0, CANVAS_HEIGHT, 120):
        cv2.line(canvas, (0, y), (CANVAS_WIDTH, y), GRID, 1)


def draw_map_background(canvas: np.ndarray) -> None:
    """Draw a softened map background aligned with the mathematical coordinates."""
    encoded = np.fromfile(MAP_PATH, dtype=np.uint8)
    map_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if map_image is None:
        raise FileNotFoundError(f"Cannot read map image: {MAP_PATH}")

    map_height, map_width = map_image.shape[:2]
    scale = min(
        (CANVAS_WIDTH - 2 * HORIZONTAL_PADDING) / map_width,
        (CANVAS_HEIGHT - TOP_PADDING - BOTTOM_PADDING) / map_height,
    )
    output_width = round(map_width * scale)
    output_height = round(map_height * scale)
    map_image = cv2.resize(map_image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    softened = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    softened = cv2.addWeighted(map_image, 0.35, softened, 0.65, 0)
    white = np.full_like(softened, 255)
    softened = cv2.addWeighted(softened, 0.58, white, 0.42, 0)
    offset_x = round((CANVAS_WIDTH - output_width) / 2)
    offset_y = round(
        TOP_PADDING
        + (CANVAS_HEIGHT - TOP_PADDING - BOTTOM_PADDING - output_height) / 2
    )
    canvas[offset_y : offset_y + output_height, offset_x : offset_x + output_width] = softened


def draw_route(
    canvas: np.ndarray, route: list[int], screen_points: list[tuple[int, int]]
) -> None:
    """Draw a restrained route with direction markers."""
    for start, end in zip(route, route[1:]):
        start_point = screen_points[start]
        end_point = screen_points[end]
        cv2.line(canvas, start_point, end_point, ROUTE_SHADOW, 6, cv2.LINE_AA)
        cv2.line(canvas, start_point, end_point, ROUTE, 2, cv2.LINE_AA)

        direction = np.array(end_point, dtype=float) - np.array(start_point, dtype=float)
        length = np.linalg.norm(direction)
        if length >= 16:
            unit_direction = direction / length
            midpoint = (np.array(start_point, dtype=float) + np.array(end_point, dtype=float)) / 2
            arrow_length = min(30.0, length * 0.45)
            arrow_start = tuple(
                np.round(midpoint - unit_direction * arrow_length / 2).astype(int)
            )
            arrow_end = tuple(
                np.round(midpoint + unit_direction * arrow_length / 2).astype(int)
            )
            cv2.arrowedLine(canvas, arrow_start, arrow_end, BACKGROUND, 4, cv2.LINE_AA, tipLength=0.45)
            cv2.arrowedLine(canvas, arrow_start, arrow_end, ROUTE, 2, cv2.LINE_AA, tipLength=0.45)


def draw_points(canvas: np.ndarray, screen_points: list[tuple[int, int]]) -> None:
    """Draw delivery stops and the depot with contrasting visual treatments."""
    for point in screen_points[1:]:
        cv2.circle(canvas, point, 10, (224, 239, 220), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 6, DELIVERY, -1, cv2.LINE_AA)

    depot = screen_points[0]
    cv2.circle(canvas, depot, 18, (222, 237, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, depot, 11, DEPOT, -1, cv2.LINE_AA)
    cv2.circle(canvas, depot, 4, PANEL, -1, cv2.LINE_AA)


def draw_header(canvas: np.ndarray, delivery_count: int, total_length: float) -> None:
    """Add title and compact route metrics."""
    cv2.rectangle(canvas, (40, 35), (650, 150), PANEL, -1, cv2.LINE_AA)
    cv2.line(canvas, (70, 52), (70, 126), ROUTE, 3, cv2.LINE_AA)

    legend_x = CANVAS_WIDTH - 300
    cv2.rectangle(canvas, (legend_x, 35), (CANVAS_WIDTH - 40, 125), PANEL, -1, cv2.LINE_AA)
    cv2.circle(canvas, (legend_x + 35, 67), 7, DEPOT, -1, cv2.LINE_AA)
    cv2.circle(canvas, (legend_x + 35, 101), 7, DELIVERY, -1, cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(FONT_PATH, 26)
    body_font = ImageFont.truetype(FONT_PATH, 17)
    draw.text((92, 57), "无人机配送路线", font=title_font, fill=TEXT[::-1])
    draw.text(
        (92, 105),
        f"{delivery_count} 个配送点  |  总距离 {total_length:.1f} 像素",
        font=body_font,
        fill=MUTED_TEXT[::-1],
    )
    draw.text((legend_x + 55, 55), "仓库", font=body_font, fill=TEXT[::-1])
    draw.text((legend_x + 55, 89), "配送点", font=body_font, fill=TEXT[::-1])
    canvas[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save_image(output_path: Path, canvas: np.ndarray) -> None:
    """Save PNG bytes without OpenCV's Windows Unicode-path limitation."""
    success, encoded = cv2.imencode(output_path.suffix, canvas)
    if not success:
        raise OSError("Cannot encode route visualization as PNG")
    encoded.tofile(str(output_path))


def main() -> None:
    """Solve the TSP and save its route visualization."""
    depot, deliveries = load_points(INPUT_POINT_PATH)
    route, total_length = solve_tsp(depot, deliveries)
    screen_points = project_points([depot] + deliveries)

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    draw_map_background(canvas)
    draw_grid(canvas)
    draw_route(canvas, route, screen_points)
    draw_points(canvas, screen_points)
    draw_header(canvas, len(deliveries), total_length)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_image(OUTPUT_PATH, canvas)
    print(f"Saved route visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
