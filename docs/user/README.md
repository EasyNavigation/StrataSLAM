# StrataSLAM user guide

StrataSLAM builds 3D maps of multi-floor buildings: one point cloud per floor and a global one, without
the floors collapsing into each other. It works with an RGB-D camera, a 3D LiDAR or both, or a 2D laser scanner
on a ground robot (one level at a time, with or without a camera), with an IMU, a barometer and external odometry as
options, and is configured with a single YAML profile. Besides the clouds, a map exports an occupancy grid per floor
for navigation (nav2).

1. [Install and build](install.md)
2. [Write a profile](profiles.md)
3. [Check your setup with `strataslam doctor`](doctor.md)
4. [Map a building](mapping.md)
5. [Fuse the floors, export and evaluate](fusion.md)
6. [Measure the noise of an IMU](imu-noise.md)
7. [Troubleshooting](troubleshooting.md)

Reference: [map format](../data_format.md) · [design decisions](../decisions.md) · [Pose4 mathematics](../math/pose4.md) ·
[benchmarks](../benchmarks.md) · [scenario results](../scenarios.md).
