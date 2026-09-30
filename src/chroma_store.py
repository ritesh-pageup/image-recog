
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


class ChromaImageStore:
    def __init__(
        self,
        path: str,
        collection_name: str,
        distance: str = "cosine",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": distance,
                }
            },
        )

    @staticmethod
    def sha256_file(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def make_id(
        source_sha256: str,
        augmentation_index: int,
        pipeline_version: str = "v1",
    ) -> str:
        raw = (
            f"{source_sha256}:"
            f"{augmentation_index}:"
            f"{pipeline_version}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def exists(self, record_id: str) -> bool:
        result = self.collection.get(ids=[record_id], include=[])
        return bool(result["ids"])

    def source_exists(self, source_sha256: str) -> bool:
        result = self.collection.get(
            where={"source_sha256": source_sha256},
            include=[],
            limit=1,
        )
        return bool(result["ids"])

    def add(
        self,
        record_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> int:
        if len(record_ids) == 0:
            return 0

        existing = self.collection.get(
            ids=record_ids,
            include=[],
        )
        existing_ids = set(existing["ids"])

        keep = [
            i for i, record_id in enumerate(record_ids)
            if record_id not in existing_ids
        ]

        if not keep:
            return 0

        self.collection.add(
            ids=[record_ids[i] for i in keep],
            embeddings=embeddings[keep].tolist(),
            metadatas=[metadatas[i] for i in keep],
            documents=[documents[i] for i in keep],
        )

        return len(keep)

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        include = include or ["metadatas", "distances", "documents"]

        return self.collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=top_k,
            where=where,
            include=include,
        )

    def count(self) -> int:
        return self.collection.count()
