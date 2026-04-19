# COLMAP Pipeline Repro Guide

This guide keeps the COLMAP dataset folder clean and makes pipeline runs reproducible in host or Docker setups.

## 1) Recommended dataset layout

Use a single dataset root (default: `colmap_dataset`):

- `images/` (input images)
- `custom_matches.txt` (pair list)
- `sparse/model/` (PX4 prior model: `cameras.txt`, `images.txt`, `points3D.txt`)

Generated artifacts (safe to regenerate):

- `db.db`
- `dense/`
- `sparse/0/`
- `sparse/triangulated/`

## 2) Clean before reruns

Preserve source images and priors:

```bash
./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset
```

Full reset including source images and priors:

```bash
./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset --full
```

## 3) Portable COLMAP binary selection

`launch/colmap_pipeline.py` resolves COLMAP in this order:

1. `COLMAP_BIN` environment variable
2. local build: `../colmap_cuda_src/build_cuda/src/colmap/exe/colmap`
3. `/opt/colmap/bin/colmap`
4. `/usr/local/bin/colmap`
5. `/usr/bin/colmap`
6. `colmap` from `PATH`

For Docker, set `COLMAP_BIN` explicitly to remove ambiguity.

## 4) Example Docker run

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace \
  -v "$PWD/colmap_dataset":/data/colmap_dataset \
  -e COLMAP_BIN=/opt/colmap/bin/colmap \
  your-image:tag \
  bash -lc "cd /workspace/PX4-Autopilot && ./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset && python3 launch/colmap_pipeline.py --reconstruct-existing --dataset-dir /data/colmap_dataset"
```

## 5) Repro tips

- Pin your Docker image tag and COLMAP version/commit.
- Keep only one active dataset root per experiment run.
- Archive `images/`, `custom_matches.txt`, and `sparse/model/` as immutable inputs.
