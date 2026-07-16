import cv2
import csv
import numpy as np


IMAGE_PATH = r"C:\Users\29532\AppData\Local\Temp\codex-clipboard-d53bd2d9-3d93-4230-afd4-49ab75070304.png"
OUTPUT_PATH = "detected_centers.png"
RAW_CENTERS_CSV_PATH = "raw_centers.csv"


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


def save_raw_centers_csv(green_centers, blue_centers, image_shape, output_path):
    height, width = image_shape[:2]
    fieldnames = [
        "color",
        "x_pixel",
        "y_pixel",
        "image_width",
        "image_height",
    ]
    rows = [
        {
            "color": "green",
            "x_pixel": x,
            "y_pixel": y,
            "image_width": width,
            "image_height": height,
        }
        for x, y in green_centers
    ] + [
        {
            "color": "blue",
            "x_pixel": x,
            "y_pixel": y,
            "image_width": width,
            "image_height": height,
        }
        for x, y in blue_centers
    ]
    rows.sort(key=lambda p: (p["y_pixel"], p["x_pixel"], p["color"]))

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
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
    save_raw_centers_csv(green_centers, blue_centers, img.shape, RAW_CENTERS_CSV_PATH)

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
    print(f"Saved raw centers to: {RAW_CENTERS_CSV_PATH}")


if __name__ == "__main__":
    main()
