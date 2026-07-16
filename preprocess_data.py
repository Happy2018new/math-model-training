import csv
import json

import numpy as np


RAW_CENTERS_CSV_PATH = "raw_centers.csv"
POINTS_CSV_PATH = "points.csv"
POINTS_JSON_PATH = "points.json"
DISTANCE_MATRIX_PATH = "distance_matrix.csv"


def load_raw_centers(input_path):
    with open(input_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return [], None, None

    points = []
    for row in rows:
        points.append(
            {
                "color": row["color"],
                "x_pixel": float(row["x_pixel"]),
                "y_pixel": float(row["y_pixel"]),
            }
        )

    width = int(float(rows[0]["image_width"]))
    height = int(float(rows[0]["image_height"]))
    return points, width, height


def preprocess_points(raw_points, image_width, image_height):
    raw_points = sorted(
        raw_points,
        key=lambda p: (p["y_pixel"], p["x_pixel"], p["color"]),
    )

    color_counts = {}
    processed = []
    for idx, point in enumerate(raw_points, start=1):
        color = point["color"]
        color_counts[color] = color_counts.get(color, 0) + 1
        x_pixel = float(point["x_pixel"])
        y_pixel = float(point["y_pixel"])

        processed.append(
            {
                "id": idx,
                "color": color,
                "color_id": color_counts[color],
                "x_pixel": round(x_pixel, 2),
                "y_pixel": round(y_pixel, 2),
                "x_norm": round(x_pixel / (image_width - 1), 6),
                "y_norm": round(y_pixel / (image_height - 1), 6),
                "x_math": round(x_pixel, 2),
                "y_math": round((image_height - 1) - y_pixel, 2),
            }
        )

    return processed


def save_points_csv(points, output_path):
    fieldnames = [
        "id",
        "color",
        "color_id",
        "x_pixel",
        "y_pixel",
        "x_norm",
        "y_norm",
        "x_math",
        "y_math",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def save_points_json(points, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(points, file, ensure_ascii=False, indent=2)


def save_distance_matrix(points, output_path):
    coords = np.array(
        [(point["x_pixel"], point["y_pixel"]) for point in points],
        dtype=np.float64,
    )
    if len(coords) == 0:
        matrix = np.empty((0, 0), dtype=np.float64)
    else:
        diff = coords[:, None, :] - coords[None, :, :]
        matrix = np.sqrt(np.sum(diff * diff, axis=2))

    header = ",".join(["id"] + [str(point["id"]) for point in points])
    rows = []
    for point, distances in zip(points, matrix):
        values = [str(point["id"])] + [f"{distance:.4f}" for distance in distances]
        rows.append(",".join(values))

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(header + "\n")
        file.write("\n".join(rows))
        if rows:
            file.write("\n")


def main():
    raw_points, image_width, image_height = load_raw_centers(RAW_CENTERS_CSV_PATH)
    if image_width is None or image_height is None:
        raise ValueError(f"No raw centers found in {RAW_CENTERS_CSV_PATH}")

    points = preprocess_points(raw_points, image_width, image_height)
    save_points_csv(points, POINTS_CSV_PATH)
    save_points_json(points, POINTS_JSON_PATH)
    save_distance_matrix(points, DISTANCE_MATRIX_PATH)

    print(f"Loaded raw centers from: {RAW_CENTERS_CSV_PATH}")
    print(f"Total count: {len(points)}")
    print(f"Saved preprocessed points to: {POINTS_CSV_PATH}")
    print(f"Saved preprocessed points to: {POINTS_JSON_PATH}")
    print(f"Saved distance matrix to: {DISTANCE_MATRIX_PATH}")


if __name__ == "__main__":
    main()
