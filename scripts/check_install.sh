#!/usr/bin/env bash
# Follows docs/user/install.md in clean Docker containers, so the guide is tested as written.
#
#   scripts/check_install.sh core [log]   # ubuntu:24.04, no ROS: core + simulator build and ctest
#   scripts/check_install.sh ros  [log]   # ROS 2 Jazzy image: colcon build of the five packages, CLI, colcon test
#
# The repository is copied (read-only mount, then cp) into the container; nothing on the host changes.
set -euo pipefail

mode="${1:-}"
log="${2:-/dev/stdout}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
ros_image="${STRATASLAM_ROS_IMAGE:-osrf/ros:jazzy-desktop}"

case "$mode" in
  core)
    docker run --rm -v "$repo":/repo:ro ubuntu:24.04 bash -c '
      set -euxo pipefail
      cp -r /repo /strataslam && cd /strataslam
      scripts/install_deps.sh core > /tmp/apt.log 2>&1 || { tail -20 /tmp/apt.log; exit 1; }
      cmake -S superbuild -B build -DCMAKE_BUILD_TYPE=Release
      cmake --build build -j"$(nproc)"
      ctest --test-dir build --output-on-failure -j4
      ldd build/strataslam_core/libstrataslam_core.so | grep -c /opt/ros && exit 1 || echo "no ROS library linked"
      echo CHECK_INSTALL_CORE_OK' > "$log" 2>&1
    ;;
  ros)
    docker run --rm -v "$repo":/repo:ro "$ros_image" bash -c '
      set -exo pipefail   # no -u: the ROS setup scripts read unset variables
      source /opt/ros/jazzy/setup.bash
      mkdir -p /ws/src && cp -r /repo /ws/src/strataslam
      /ws/src/strataslam/scripts/install_deps.sh ros > /tmp/apt.log 2>&1 || { tail -20 /tmp/apt.log; exit 1; }
      cd /ws
      colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
      test "$(ls install | grep -c ^strataslam)" -eq 5
      source install/setup.bash
      strataslam --help
      ros2 pkg executables strataslam_ros
      colcon test --event-handlers console_direct+ || true
      colcon test-result --verbose
      echo CHECK_INSTALL_ROS_OK' > "$log" 2>&1
    ;;
  *)
    echo "usage: $0 core|ros [log]" >&2
    exit 2
    ;;
esac
