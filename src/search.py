
from __future__ import annotations

from pathlib import Path

import numpy as np

from .augmentation import load_rgb
from .chroma_store import ChromaImageStore
from .config import AppConfig
from .embedding import DINOv3Embedder


def search_image(
    image_path: str,
    embedder: DINOv3Embedder,
    store: ChromaImageStore,
    top_k: int = 10,
    unique_originals: bool = False,
    include_original: bool = True,
):
    image = load_rgb(Path(image_path))
    embedding = embedder.encode([image], batch_size=1)[0]

    # Fetch more than top_k if we need to collapse many augmentations from
    # the same source image.
    raw_k = max(top_k * 10, top_k) if unique_originals else top_k

    result = store.search(
        embedding=embedding,
        top_k=raw_k,
        include=["metadatas", "distances", "documents"],
    )

    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]

    results = []
    seen_sources = set()

    for record_id, metadata, distance, document in zip(
        ids,
        metadatas,
        distances,
        documents,
    ):
        if not include_original and metadata.get("record_type") == "original":
            continue

        source_sha = metadata.get("source_sha256")

        if unique_originals and source_sha in seen_sources:
            continue

        seen_sources.add(source_sha)

        # Chroma cosine distance is converted into a convenient similarity
        # score. For normalized vectors this is equivalent to cosine similarity
        # up to numerical precision.
        similarity = 1.0 - float(distance)

        results.append({
            "id": record_id,
            "similarity": similarity,
            "distance": float(distance),
            "source_filename": metadata.get("source_filename"),
            "source_path": metadata.get("source_path"),
            "augmentation_index": metadata.get("augmentation_index"),
            "augmentation_type": metadata.get("augmentation_type"),
            "image_path": metadata.get("image_path"),
            "document": document,
        })

        if len(results) >= top_k:
            break

    return results
