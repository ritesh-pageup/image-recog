
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_pipeline(width: int, height: int) -> A.Compose:
    # This intentionally uses stochastic combinations rather than a fixed
    # transform repeated 50 times.
    return A.Compose(
        [
            A.SmallestMaxSize(
                max_size=max(width, height),
                p=1.0,
            ),
            A.RandomCrop(
                height=height,
                width=width,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.10),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-20, 20),
                shear=(-8, 8),
                p=0.45,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.20,
                        contrast_limit=0.20,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=12,
                        sat_shift_limit=20,
                        val_shift_limit=15,
                        p=1.0,
                    ),
                    A.CLAHE(
                        clip_limit=(1, 3),
                        tile_grid_size=(8, 8),
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(85, 115),
                        p=1.0,
                    ),
                ],
                p=0.65,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=(3, 7),
                        p=1.0,
                    ),
                    A.MotionBlur(
                        blur_limit=(3, 7),
                        p=1.0,
                    ),
                    A.MedianBlur(
                        blur_limit=(3, 7),
                        p=1.0,
                    ),
                    A.GaussNoise(
                        std_range=(0.02, 0.08),
                        p=1.0,
                    ),
                ],
                p=0.25,
            ),
            A.OneOf(
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 5),
                        hole_height_range=(0.03, 0.12),
                        hole_width_range=(0.03, 0.12),
                        p=1.0,
                    ),
                    A.GridDropout(
                        ratio=0.20,
                        p=1.0,
                    ),
                ],
                p=0.15,
            ),
            A.ImageCompression(
                quality_range=(65, 95),
                p=0.20,
            ),
            A.Resize(
                height=height,
                width=width,
                p=1.0,
            ),
        ],
        p=1.0,
    )


def generate_augmentations(
    image: np.ndarray,
    count: int,
    width: int,
    height: int,
):
    pipeline = create_pipeline(width, height)

    for index in range(1, count + 1):
        result = pipeline(image=image)
        yield index, result["image"]
