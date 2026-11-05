#!/usr/bin/env bash
# Downloads public datasets for the nightly sanity runs, verifying SHA-256. Data is never committed.
# Usage: scripts/download_datasets.sh [dest_dir]   (default: ./data)
set -euo pipefail
dest="${1:-data}"
mkdir -p "$dest"

# name | url | sha256 | license note   (the name is the directory the archive creates)
datasets=(
  "rgbd_dataset_freiburg1_xyz|https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz|a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98|CC BY 4.0 (TUM RGB-D)"
  "rgbd_dataset_freiburg2_xyz|https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz|a4903027a7fe6bc6573a1ce96a8afa5689f15d74f4a3f0627e91b5799f62e40b|CC BY 4.0 (TUM RGB-D)"
  # KITTI raw (Velodyne HDL-64 + OXTS GPS/INS ground truth): CC BY-NC-SA 3.0, evaluation only, never redistributed.
  "2011_09_30/2011_09_30_drive_0027_sync|https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip|614a8df95033a9dfecae82c410df5990d9af0a3f2d0b3467e5fe3f8645176c38|CC BY-NC-SA 3.0 (KITTI), evaluation only"
  "2011_09_30/calib_imu_to_velo.txt|https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip|2ed01563b616baec14f25659671bd9a77810bd87752d709d184fe304ff7258e3|CC BY-NC-SA 3.0 (KITTI), evaluation only"
)

for entry in "${datasets[@]}"; do
  IFS='|' read -r name url sha license <<<"$entry"
  if [[ -e "$dest/$name" ]]; then echo "$name: already present"; continue; fi
  tmp="$(mktemp -d)"
  archive="$tmp/$(basename "$url")"
  echo "$name: downloading ($license)"
  curl -fL --retry 3 -o "$archive" "$url"
  echo "$sha  $archive" | sha256sum -c -
  case "$archive" in
    *.tgz) tar -xzf "$archive" -C "$dest" ;;
    *.zip) unzip -q -o "$archive" -d "$dest" ;;
  esac
  rm -rf "$tmp"
done
echo "done"
