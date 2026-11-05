#!/usr/bin/env bash
# Installs the system packages strataslam needs on Ubuntu 24.04 (single source of truth for the
# install guide, the CI and scripts/check_install.sh).
#
#   scripts/install_deps.sh ros    # full system: core, simulator, ROS 2 node, messages, CLI (needs ROS 2 Jazzy apt repo)
#   scripts/install_deps.sh core   # core + simulator only, plain Ubuntu, no ROS at all
#
# Runs apt through sudo when not root.
set -euo pipefail

mode="${1:-}"
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
# A mirror that stops answering fails the download after 30 s and is retried, instead of hanging.
apt=(apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
# Only what the build needs: the recommended packages (GTSAM's documentation, MathJax, video drivers...) made 306 MB in
# 407 packages, more than an hour on a slow mirror in the CI.
install=("${apt[@]}" install -y --no-install-recommends)

common=(cmake g++ git libeigen3-dev libyaml-cpp-dev libopencv-dev libpng-dev libjpeg-dev libgtest-dev libbenchmark-dev
        python3-yaml python3-numpy python3-pytest)

case "$mode" in
  ros)
    $SUDO "${apt[@]}" update
    # The launch/launch_testing packages are listed together so that apt brings them to one version:
    # a newer launch_testing_ros next to an older launch_testing breaks pytest's plugin loading.
    $SUDO "${install[@]}" "${common[@]}" python3-dev pybind11-dev ros-jazzy-gtsam ros-jazzy-rosbag2 \
      ros-jazzy-rosbag2-storage-default-plugins ros-jazzy-launch-ros ros-jazzy-launch-testing \
      ros-jazzy-launch-testing-ros ros-jazzy-launch-testing-ament-cmake ros-jazzy-ament-cmake-pytest \
      python3-colcon-common-extensions python3-tk  # Tk: the viewer's question when its window is closed
    ;;
  core)
    $SUDO "${apt[@]}" update
    # libgtsam-dev's CMake export needs Boost, METIS and SuiteSparse (CCOLAMD) at configure/link
    # time, and references a libCppUnitLite.a that the Ubuntu package does not ship (ADR-0007).
    $SUDO "${install[@]}" "${common[@]}" libgtsam-dev libboost-all-dev libmetis-dev libsuitesparse-dev
    lib=/usr/lib/$(uname -m)-linux-gnu/libCppUnitLite.a
    # Empty placeholder for the missing file: it is checked for existence, never linked.
    [ -e "$lib" ] || $SUDO touch "$lib"
    ;;
  *)
    echo "usage: $0 ros|core" >&2
    exit 2
    ;;
esac
