
from __future__ import annotations

import argparse
import json

from src.chroma_store import ChromaImageStore
from src.config import load_config
from src.embedding import DINOv3Embedder
from src.indexer import index_dataset
from src.search import search_image


def build_components():
    config = load_config()

    embedder = DINOv3Embedder(
        model_name=config.model_name,
        weights_path=config.weights_path,
        device=config.device,
        dtype=config.dtype,
        normalize=config.normalize_embeddings,
    )

    store = ChromaImageStore(
        path=config.chroma_path,
        collection_name=config.collection_name,
        distance=config.distance,
    )

    return config, embedder, store


def cmd_index():
    config, embedder, store = build_components()
    index_dataset(config, embedder, store)


def cmd_search(args):
    _, embedder, store = build_components()

    results = search_image(
        image_path=args.image,
        embedder=embedder,
        store=store,
        top_k=args.top_k,
        unique_originals=args.unique_originals,
        include_original=args.include_original,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 image similarity with ChromaDB"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index")
    index_parser.set_defaults(func=lambda _: cmd_index())

    search_parser = sub.add_parser("search")
    search_parser.add_argument("--image", required=True)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument(
        "--unique-originals",
        action="store_true",
        help="Return at most one result per original source image.",
    )
    search_parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the original source record in search results.",
    )
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
