# Source me: environment for building against ROS Jazzy's GTSAM (see docs/decisions.md ADR-0007).
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export GTSAM_DIR=/opt/ros/jazzy/lib/cmake/GTSAM
