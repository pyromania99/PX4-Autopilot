#!/usr/bin/env bash
set -euo pipefail

# Clean generated COLMAP artifacts while preserving source images and PX4 priors by default.

DATASET_DIR="colmap_dataset"
FULL_CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --full)
      FULL_CLEAN=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: cleanup_colmap_dataset.sh [--dataset-dir PATH] [--full]

Default cleanup removes generated reconstruction artifacts and preserves:
- images/
- sparse/model/ priors
- custom_matches.txt

--full additionally removes images/ and sparse/model/.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

DATASET_DIR="$(realpath -m "$DATASET_DIR")"
if [[ "$DATASET_DIR" == "/" || "$DATASET_DIR" == "." ]]; then
  echo "Refusing to clean unsafe dataset path: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$DATASET_DIR"

rm -f "$DATASET_DIR/db.db" "$DATASET_DIR/db.db-shm" "$DATASET_DIR/db.db-wal"
rm -rf "$DATASET_DIR/dense"
rm -rf "$DATASET_DIR/sparse/0" "$DATASET_DIR/sparse/triangulated"

if [[ "$FULL_CLEAN" -eq 1 ]]; then
  rm -rf "$DATASET_DIR/images" "$DATASET_DIR/sparse/model"
fi

echo "Cleanup complete for dataset: $DATASET_DIR"
if [[ "$FULL_CLEAN" -eq 0 ]]; then
  echo "Preserved: images/, sparse/model/, custom_matches.txt"
fi
