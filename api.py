
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.chroma_store import ChromaImageStore
from src.config import load_config
from src.embedding import DINOv3Embedder
from src.search import search_image


config = load_config()

embedder: DINOv3Embedder | None = None
store: ChromaImageStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store

    embedder = DINOv3Embedder(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
        normalize=config.normalize_embeddings,
    )

    store = ChromaImageStore(
        path=config.chroma_path,
        collection_name=config.collection_name,
        distance=config.distance,
    )

    yield

    # Release model memory when the process exits.
    embedder = None
    store = None


app = FastAPI(
    title="DINOv3 Image Similarity API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(embedder.device) if embedder else None,
        "indexed_records": store.count() if store else 0,
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=100),
    unique_originals: bool = Query(False),
    include_original: bool = Query(True),
):
    if embedder is None or store is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not initialized.",
        )

    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:
            temp.write(data)
            temp_path = temp.name

        results = search_image(
            image_path=temp_path,
            embedder=embedder,
            store=store,
            top_k=top_k,
            unique_originals=unique_originals,
            include_original=include_original,
        )

        return {
            "query_filename": file.filename,
            "count": len(results),
            "results": results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
