# DINOv3 + Albumentations + ChromaDB Image Similarity

A runnable local image similarity system:

1. Reads images from `data/input/`
2. Generates exactly 50 stochastic Albumentations variants per unique source image
3. Embeds images in batches with DINOv3
4. Stores normalized vectors + metadata in persistent local ChromaDB
5. Deduplicates source files by SHA-256 and generated records by deterministic IDs
6. Supports CLI indexing/search
7. Includes a FastAPI image-similarity API
8. Automatically uses CUDA when available, otherwise CPU

## Architecture

```text
             data/input
                  |
                  v
          SHA-256 duplicate check
                  |
                  v
           Albumentations
             50 variants
                  |
                  v
       DINOv3 batched inference
                  |
                  v
        L2-normalized embeddings
                  |
                  v
          Persistent ChromaDB
                  |
          +-------+-------+
          |               |
       CLI search       FastAPI
```

## 1. Create environment

Python 3.10+ is recommended.

### CPU

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### NVIDIA GPU

Install a PyTorch build matching your CUDA version first, then install the project requirements.

Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The project itself does not hard-code a CUDA version.

## 2. DINOv3 model access

The default model is:

`facebook/dinov3-vits16-pretrain-lvd1689m`

DINOv3 checkpoints can have Hugging Face access/license requirements. If the model cannot be downloaded, authenticate with Hugging Face as required by the checkpoint:

```bash
huggingface-cli login
```

You can change `model.name` in `config.yaml`.

## 3. Add images

Put images into:

```text
data/input/
```

Supported formats:

- jpg
- jpeg
- png
- webp
- bmp
- tif
- tiff

## 4. Index dataset

```bash
python main.py index
```

For each unique input image, the application generates:

```text
image.jpg
  -> aug_001.jpg
  -> aug_002.jpg
  ...
  -> aug_050.jpg
```

Then all 51 images (original + 50 variants) can be embedded. By default the original is indexed as well.

If you only want augmented images, change `include_original` in `src/indexer.py`.

## 5. Search

Search using another image:

```bash
python main.py search --image path/to/query.jpg --top-k 10
```

You can suppress duplicate variants from the same original:

```bash
python main.py search --image path/to/query.jpg --top-k 10 --unique-originals
```

This is useful because 50 variants of one source image otherwise may occupy many top-K positions.

## 6. Run API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Search endpoint

`POST /search`

Form fields:

- `file`: image
- `top_k`: number of raw nearest vectors
- `unique_originals`: return at most one result per original image
- `include_original`: whether original-image records can be returned

Example:

```bash
curl -X POST "http://127.0.0.1:8000/search?top_k=10&unique_originals=true" \
  -F "file=@query.jpg"
```

### Health

```text
GET /health
```

## Duplicate handling

There are two levels:

### Source duplicate

A SHA-256 hash of the original file is stored as:

```text
source_sha256
```

If the same file is indexed again, the application skips regeneration.

### Record duplicate

Every stored record gets a deterministic ID based on:

```text
source_sha256 + augmentation_index + pipeline_version
```

Chroma `get_or_add`-style behavior is implemented by checking existing IDs before insertion, so rerunning the index command does not create duplicate records.

## Embedding

DINOv3's CLS/global representation is used as the image embedding.

The vectors are L2-normalized before storage. Chroma uses cosine distance.

For normalized vectors:

```text
cosine similarity = 1 - cosine distance
```

The API exposes both the Chroma distance and a converted similarity score.

## GPU memory

If you hit CUDA OOM:

```yaml
model:
  batch_size: 4
```

or:

```yaml
model:
  batch_size: 2
```

For CPU, start with:

```yaml
model:
  batch_size: 4
```

## Important augmentation note

The project uses stochastic Albumentations transforms. It does not reuse one fixed augmentation 50 times.

The 50 outputs therefore represent different random transform combinations.

For domains where flips, rotations, color changes, or occlusion are not semantically valid, edit:

```text
src/augmentation.py
```

## Production considerations

For a large dataset, do not necessarily save every augmented JPEG forever. You can configure:

```yaml
augmentation:
  save_images: false
```

and only store embeddings + metadata.

For very large collections, consider batching inserts and periodically compacting/maintaining the Chroma deployment.

## License / model

Check the license and access conditions of the exact DINOv3 checkpoint you use. The code in this repository is separate from the model weights.
