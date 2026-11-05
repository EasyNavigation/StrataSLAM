#!/usr/bin/env bash
# Rebuilds the visual vocabulary shipped as `vocabulary: default` (strataslam_ros/vocabularies/tum_rgbd.bin, ADR-0054):
# every fifth colour image of TUM RGB-D fr1/xyz and fr2/xyz (CC BY 4.0), 892 images, in name order. The training is
# deterministic (k-means with a fixed seed), so the same images give the same file.
#
#   scripts/download_datasets.sh data
#   scripts/train_default_vocabulary.sh data out.bin
set -euo pipefail
data="$1"; out="$2"
images="$(mktemp -d)"
trap 'rm -r -- "$images"' EXIT
for seq in rgbd_dataset_freiburg1_xyz rgbd_dataset_freiburg2_xyz; do
  [ -d "$data/$seq/rgb" ] || { echo "missing $data/$seq (scripts/download_datasets.sh)" >&2; exit 1; }
  i=0
  for f in "$data/$seq"/rgb/*.png; do
    i=$((i + 1))
    if [ $((i % 5)) -eq 0 ]; then ln -s "$(realpath "$f")" "$images/${seq}_$(basename "$f")"; fi
  done
done
strataslam vocab train "$images" --out "$out"
