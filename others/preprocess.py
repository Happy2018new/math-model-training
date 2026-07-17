import cv2
import csv
import json
from pathlib import Path

import numpy as np

IMAGE_PATH = r"others/input_picture.png"
OUTPUT_DIR = Path("output/processed")
OUTPUT_PATH = OUTPUT_DIR / "detected_centers.png"
POINTS_CSV_PATH = OUTPUT_DIR / "points.csv"
POINTS_JSON_PATH = OUTPUT_DIR / "points.json"
DISTANCE_MATRIX_PATH = OUTPUT_DIR / "distance_matrix.csv"


def find_circle_centers(mask, min_area=20, max_area=500, split_factor=1.55):
    """Return circle centers as (x, y) pixel coordinates.

    Adjacent dots may touch each other and become one connected component.
    Components whose area is much larger than a normal dot are split with
    k-means on their foreground pixels, so close dots are counted separately.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    centers = []
    areas = [
        stats[label, cv2.CC_STAT_AREA]
        for label in range(1, num_labels)
        if min_area <= stats[label, cv2.CC_STAT_AREA] <= max_area
    ]

    if not areas:
        return centers

    normal_area = float(np.median(areas))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        expected_count = max(1, int(round(area / normal_area)))
        if expected_count == 1 or area < split_factor * normal_area:
            cx, cy = centroids[label]
            centers.append((round(float(cx), 2), round(float(cy), 2)))
            continue

        ys, xs = np.where(labels == label)
        points = np.column_stack([xs, ys]).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.05,
        )
        cv2.setRNGSeed(0)
        _compactness, _cluster_labels, cluster_centers = cv2.kmeans(
            points,
            expected_count,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        for cx, cy in cluster_centers:
            centers.append((round(float(cx), 2), round(float(cy), 2)))

    return sorted(centers, key=lambda p: (p[1], p[0]))


def preprocess_points(green_centers, blue_centers, image_shape):
    image_height, image_width = image_shape[:2]
    raw_points = [
        {"color": color, "x_pixel": x, "y_pixel": y}
        for color, centers in (("green", green_centers), ("blue", blue_centers))
        for x, y in centers
    ]
    raw_points.sort(key=lambda p: (p["y_pixel"], p["x_pixel"], p["color"]))

    color_counts = {}
    points = []
    for idx, point in enumerate(raw_points, start=1):
        color = point["color"]
        color_counts[color] = color_counts.get(color, 0) + 1
        x_pixel = float(point["x_pixel"])
        y_pixel = float(point["y_pixel"])
        points.append(
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
    return points


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
    coords = np.array([(point["x_pixel"], point["y_pixel"]) for point in points])
    diff = coords[:, None, :] - coords[None, :, :]
    matrix = np.sqrt(np.sum(diff * diff, axis=2)) if len(coords) else np.empty((0, 0))

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(",".join(["id"] + [str(point["id"]) for point in points]) + "\n")
        for point, distances in zip(points, matrix):
            file.write(
                ",".join(
                    [str(point["id"])] + [f"{distance:.4f}" for distance in distances]
                )
                + "\n"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower = np.array([35, 80, 80])
    green_upper = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    blue_lower = np.array([95, 80, 60])
    blue_upper = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    green_centers = find_circle_centers(green_mask)
    blue_centers = find_circle_centers(blue_mask)

    print("Green centers:")
    for idx, center in enumerate(green_centers, start=1):
        print(f"{idx}: {center}")

    print("\nBlue centers:")
    for idx, center in enumerate(blue_centers, start=1):
        print(f"{idx}: {center}")

    print(f"\nGreen count: {len(green_centers)}")
    print(f"Blue count: {len(blue_centers)}")
    print(f"Total count: {len(green_centers) + len(blue_centers)}")
    points = preprocess_points(green_centers, blue_centers, img.shape)
    save_points_csv(points, POINTS_CSV_PATH)
    save_points_json(points, POINTS_JSON_PATH)
    save_distance_matrix(points, DISTANCE_MATRIX_PATH)

    vis = img.copy()

    for x, y in green_centers:
        point = (int(round(x)), int(round(y)))
        cv2.circle(vis, point, 4, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"({point[0]},{point[1]})",
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    for x, y in blue_centers:
        point = (int(round(x)), int(round(y)))
        cv2.circle(vis, point, 4, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"({point[0]},{point[1]})",
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(OUTPUT_PATH, vis)
    print(f"\nSaved annotated image to: {OUTPUT_PATH}")
    print(f"Saved preprocessed points to: {POINTS_CSV_PATH}")
    print(f"Saved preprocessed points to: {POINTS_JSON_PATH}")
    print(f"Saved distance matrix to: {DISTANCE_MATRIX_PATH}")


if __name__ == "__main__":
    main()
