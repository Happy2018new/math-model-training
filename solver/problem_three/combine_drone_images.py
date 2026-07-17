"""将问题三的单机详情图按每四张合并为一张 2x2 图片。"""

from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "output" / "problems" / "three" / "drone_routes"
OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "three" / "drone_routes_combined"
GROUP_SIZE = 4
GRID_SIZE = 2


def combine_group(image_paths: list[Path], output_path: Path) -> None:
    """将四张图片缩放后排列成 2x2，并保持单张原图的总尺寸。"""
    if len(image_paths) != GROUP_SIZE:
        raise ValueError(f"每组必须包含 {GROUP_SIZE} 张图片。")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("同一组合中的图片尺寸必须一致。")

    cell_width = width // GRID_SIZE
    cell_height = height // GRID_SIZE
    canvas = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(images):
        resized = ImageOps.contain(image, (cell_width, cell_height))
        left = (index % GRID_SIZE) * cell_width + (cell_width - resized.width) // 2
        top = (index // GRID_SIZE) * cell_height + (cell_height - resized.height) // 2
        canvas.paste(resized, (left, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    """读取 8 张单机图并生成两张四图合成图。"""
    image_paths = sorted(INPUT_DIR.glob("drone_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"未找到单机图片: {INPUT_DIR}")
    if len(image_paths) % GROUP_SIZE != 0:
        raise ValueError(f"图片数量 {len(image_paths)} 不能被每组数量 {GROUP_SIZE} 整除。")

    for start in range(0, len(image_paths), GROUP_SIZE):
        group = image_paths[start : start + GROUP_SIZE]
        group_number = start // GROUP_SIZE + 1
        output_path = OUTPUT_DIR / f"drone_group_{group_number:02d}.png"
        combine_group(group, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
