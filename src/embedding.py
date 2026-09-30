
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

# ImageNet stats used by Meta's DINOv3 torch.hub backbones (LVD-1689M weights).
_HUB_MEAN = (0.485, 0.456, 0.406)
_HUB_STD = (0.229, 0.224, 0.225)
_HUB_REPO = "facebookresearch/dinov3"


class DINOv3Embedder:
    def __init__(
        self,
        model_name: str,
        weights_path: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        normalize: bool = True,
        image_size: int = 224,
    ):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.normalize = normalize
        self.use_hub = weights_path is not None

        print(f"Loading DINOv3: {model_name}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        if self.use_hub:
            # Local checkpoint from Meta's facebookresearch/dinov3 release
            # (e.g. dinov3_vitl16_pretrain_lvd1689m-*.pth). Model code is
            # fetched from the public GitHub repo; only the weights are
            # local, so this bypasses the gated HF Hub entirely.
            weights_file = str(Path(weights_path).resolve())
            print(f"Local checkpoint: {weights_file}")

            self.processor = None
            self.model = self._load_hub_backbone(model_name, weights_file)

            from torchvision.transforms import v2

            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=_HUB_MEAN, std=_HUB_STD),
            ])
        else:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(model_name)

            # device_map="auto" can be convenient for large checkpoints, but for
            # this application explicit device placement is easier to reason about.
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.dtype if self.device != "cpu" else torch.float32,
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_hub_backbone(model_name: str, weights_file: str):
        # hubconf.py unconditionally imports segmentation/detection heads
        # (torchmetrics, termcolor, ...) that plain backbone inference does
        # not need. Trigger torch.hub's repo download/cache once, then
        # import the lightweight dinov3.hub.backbones module directly.
        import importlib
        import sys

        hub_dir = Path(torch.hub.get_dir())
        repo_dir = hub_dir / "facebookresearch_dinov3_main"

        if not repo_dir.exists():
            try:
                torch.hub.load(
                    _HUB_REPO,
                    model_name,
                    source="github",
                    weights=weights_file,
                )
            except ModuleNotFoundError:
                pass

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, model_name)
        return factory(weights=weights_file, pretrained=True, check_hash=False)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but CUDA is not available.")
            return torch.device("cuda")

        if device == "cpu":
            return torch.device("cpu")

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        raise ValueError("device must be one of: auto, cuda, cpu")

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
        if device.type == "cpu":
            return torch.float32

        if dtype == "float32":
            return torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16

        # auto
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(image).convert("RGB")

    @torch.inference_mode()
    def encode(
        self,
        images: Iterable[np.ndarray | Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:
        images = list(images)
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = [
                self._to_pil(img)
                for img in images[start:start + batch_size]
            ]

            autocast_enabled = self.device.type == "cuda"
            autocast_dtype = self.dtype

            context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                )
                if autocast_enabled
                else nullcontext()
            )

            if self.use_hub:
                pixel_values = torch.stack(
                    [self.transform(img) for img in batch]
                ).to(self.device)

                with context:
                    outputs = self.model(pixel_values)

                # Meta's torch.hub backbone entrypoints return the pooled
                # CLS-token embedding directly as a (B, embed_dim) tensor.
                embeddings = (
                    outputs["x_norm_clstoken"]
                    if isinstance(outputs, dict)
                    else outputs
                )
            else:
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with context:
                    outputs = self.model(**inputs)

                # DINOv3 documentation exposes the CLS token as the global
                # image representation used for retrieval.
                embeddings = outputs.last_hidden_state[:, 0, :]

            if self.normalize:
                embeddings = torch.nn.functional.normalize(
                    embeddings.float(),
                    p=2,
                    dim=1,
                )

            all_embeddings.append(
                embeddings.detach().cpu().numpy().astype(np.float32)
            )

        return np.concatenate(all_embeddings, axis=0)
