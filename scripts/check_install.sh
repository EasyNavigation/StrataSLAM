#!/usr/bin/env bash
# Follows docs/user/install.md in clean Docker containers, so the guide is tested as written.
#
#   scripts/check_install.sh core [log]   # ubuntu:24.04, no ROS: core + simulator build and ctest
#   scripts/check_install.sh ros  [log]   # ROS 2 Jazzy image: colcon build of the five packages, CLI, the
#                                         # first-map tutorial of docs/user/mapping.md, colcon test
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
      mkdir -p /ws/src && cp -r /repo /ws/src/StrataSLAM
      /ws/src/StrataSLAM/scripts/install_deps.sh ros > /tmp/apt.log 2>&1 || { tail -20 /tmp/apt.log; exit 1; }
      cd /ws
      colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
      test "$(ls install | grep -c ^strataslam)" -eq 5
      source install/setup.bash
      strataslam --help
      ros2 pkg executables strataslam_ros
      # docs/user/mapping.md, "A first map, start to finish", in a new shell that sources only the workspace
      bash -c "set -exo pipefail; source /ws/install/setup.bash; cd ~
        P=\$(ros2 pkg prefix strataslam_ros)/share/strataslam_ros/profiles/handheld_camera_imu.yaml
        ros2 run strataslam_ros strataslam_sim_export_bag S03 ~/strataslam_demo --profile \$P
        strataslam doctor \$P --bag ~/strataslam_demo/bag
        ros2 launch strataslam_ros strataslam.launch.py profile:=\$P vocabulary:=\$HOME/strataslam_demo/vocab.bin \
            features_topic:=/strataslam/features > /tmp/launch.log 2>&1 &
        L=\$!
        for i in \$(seq 60); do ros2 service list | grep -q /strataslam_node/save_map && break; sleep 1; done
        ros2 bag play ~/strataslam_demo/bag > /dev/null
        timeout 600 ros2 service call /strataslam_node/save_map std_srvs/srv/Trigger
        kill -TERM \$L; wait \$L || true   # a background job ignores SIGINT in a script: TERM
        MAP=\$(ls -dt ~/strataslam_maps/*/ | head -1)
        strataslam export \$MAP && strataslam export \$MAP --format grid && strataslam export \$MAP --format ply
        strataslam fuse \$MAP && test -f \$MAP/fused/floor_transforms.yaml
        strataslam eval \$MAP --gt ~/strataslam_demo"
      colcon test --event-handlers console_direct+ || true
      colcon test-result --verbose
      echo CHECK_INSTALL_ROS_OK' > "$log" 2>&1
    ;;
  *)
    echo "usage: $0 core|ros [log]" >&2
    exit 2
    ;;
esac
