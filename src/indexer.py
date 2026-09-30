
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .augmentation import generate_augmentations, list_images, load_rgb
from .chroma_store import ChromaImageStore
from .config import AppConfig
from .embedding import DINOv3Embedder


def _file_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_rgb_jpeg(path: Path, image: np.ndarray, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        str(path),
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )


def index_dataset(
    config: AppConfig,
    embedder: DINOv3Embedder,
    store: ChromaImageStore,
):
    input_images = list_images(config.input_dir)

    if not input_images:
        print(f"No images found in {config.input_dir}")
        return

    print(f"Found {len(input_images)} source image(s).")
    print(f"Chroma records before indexing: {store.count()}")

    total_added = 0
    skipped_sources = 0

    for source_path in tqdm(input_images, desc="Indexing images"):
        source_sha256 = store.sha256_file(source_path)

        # A source already represented in the collection is considered
        # completely indexed. This makes reruns idempotent.
        if store.source_exists(source_sha256):
            skipped_sources += 1
            continue

        original = load_rgb(source_path)

        images: list[np.ndarray] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        # Index original image as augmentation_index=0.
        images.append(original)
        ids.append(store.make_id(source_sha256, 0))
        metadatas.append({
            "source_sha256": source_sha256,
            "source_filename": source_path.name,
            "source_path": str(source_path.resolve()),
            "augmentation_index": 0,
            "augmentation_type": "original",
            "image_path": str(source_path.resolve()),
            "record_type": "original",
        })
        documents.append(str(source_path.resolve()))

        augmented_output_dir = (
            Path(config.augmented_dir) / source_path.stem
        )

        for augmentation_index, augmented in generate_augmentations(
            original,
            config.augmentation_count,
            config.augmentation_width,
            config.augmentation_height,
        ):
            images.append(augmented)

            record_id = store.make_id(
                source_sha256,
                augmentation_index,
            )

            if config.save_augmented_images:
                output_path = (
                    augmented_output_dir
                    / f"{source_path.stem}_aug_{augmentation_index:03d}.jpg"
                )
                _save_rgb_jpeg(output_path, augmented)
                image_path = str(output_path.resolve())
            else:
                image_path = ""

            ids.append(record_id)
            metadatas.append({
                "source_sha256": source_sha256,
                "source_filename": source_path.name,
                "source_path": str(source_path.resolve()),
                "augmentation_index": augmentation_index,
                "augmentation_type": "albumentations_random_pipeline",
                "image_path": image_path,
                "record_type": "augmented",
            })
            documents.append(image_path or str(source_path.resolve()))

        # Batch DINOv3 inference.
        embeddings = embedder.encode(
            images,
            batch_size=config.batch_size,
        )

        added = store.add(
            record_ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        total_added += added

    print()
    print(f"Added records: {total_added}")
    print(f"Skipped already-indexed sources: {skipped_sources}")
    print(f"Chroma records after indexing: {store.count()}")
