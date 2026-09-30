
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    model_name: str
    weights_path: str | None
    device: str
    dtype: str
    batch_size: int
    normalize_embeddings: bool
    augmentation_count: int
    augmentation_width: int
    augmentation_height: int
    save_augmented_images: bool
    chroma_path: str
    collection_name: str
    distance: str
    input_dir: str
    augmented_dir: str


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    m = raw["model"]
    a = raw["augmentation"]
    s = raw["storage"]
    d = raw["data"]

    return AppConfig(
        model_name=m["name"],
        weights_path=m.get("weights_path") or None,
        device=m.get("device", "auto"),
        dtype=m.get("dtype", "auto"),
        batch_size=int(m.get("batch_size", 16)),
        normalize_embeddings=bool(m.get("normalize_embeddings", True)),
        augmentation_count=int(a.get("count_per_image", 50)),
        augmentation_width=int(a.get("width", 224)),
        augmentation_height=int(a.get("height", 224)),
        save_augmented_images=bool(a.get("save_images", True)),
        chroma_path=s["chroma_path"],
        collection_name=s["collection_name"],
        distance=s.get("distance", "cosine"),
        input_dir=d["input_dir"],
        augmented_dir=d["augmented_dir"],
    )
