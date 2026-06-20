import argparse
from pathlib import Path

import cv2
import numpy as np


def create_toy_segmentation_data(output_dir, num_samples=12, image_size=128, seed=42):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(num_samples):
        base_color = rng.integers(115, 185, size=3, dtype=np.uint8)
        image = np.full((image_size, image_size, 3), base_color, dtype=np.uint8)
        noise = rng.normal(0, 8, size=image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        center = (
            int(rng.integers(image_size * 0.30, image_size * 0.70)),
            int(rng.integers(image_size * 0.30, image_size * 0.70)),
        )
        axes = (
            int(rng.integers(image_size * 0.12, image_size * 0.28)),
            int(rng.integers(image_size * 0.08, image_size * 0.22)),
        )
        angle = float(rng.integers(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        lesion_color = np.array([90, 55, 75], dtype=np.uint8) + rng.integers(-15, 16, size=3)
        lesion_color = np.clip(lesion_color, 0, 255).astype(np.uint8)
        image[mask > 0] = (
            0.65 * image[mask > 0].astype(np.float32) + 0.35 * lesion_color.astype(np.float32)
        ).astype(np.uint8)
        name = f"toy_{index:03d}.png"
        cv2.imwrite(str(images_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_dir / name), mask)
    return images_dir, masks_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Create lightweight toy segmentation data.")
    parser.add_argument("--output-dir", default="examples/toy_segmentation_demo")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir, masks_dir = create_toy_segmentation_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Created toy images: {images_dir}")
    print(f"Created toy masks: {masks_dir}")


if __name__ == "__main__":
    main()
