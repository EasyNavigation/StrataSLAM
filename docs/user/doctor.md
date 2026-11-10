# `strataslam doctor`

Run it before mapping, against the live robot or against a rosbag2:

```bash
strataslam doctor profile.yaml                 # listens for ~6 s
strataslam doctor profile.yaml --bag my_bag --json
```

Every check prints ✔ (correct), ⚠ (warning) or ✘ (error) and, when not correct, how to fix it. Exit code:
0 all ✔, 1 some ⚠, 2 some ✘, 3 the doctor itself could not run (no profile, no bag, unreadable bag).

| Check | ✔ | ⚠ | ✘ |
| --- | --- | --- | --- |
| profile | valid (the core's own validator, as the launch) | a validation warning (e.g. aerial without IMU), or not validated because `strataslam_ros` is not installed | invalid |
| topics | all publish (with `external_odometry.frame`, the TF odom -> base exists) | | one missing |
| camera rate | >= 15 Hz | 5-15 | < 5 |
| LiDAR rate (PointCloud2, or LaserScan with `lidar.type: 2d`) | >= 5 Hz | 2-5 | < 2 |
| IMU rate | >= 100 Hz | 50-100 | < 50 |
| barometer rate | >= 5 Hz | 1-5 | < 1 |
| camera_info | valid K, size matches | non-standard distortion | missing, null or NaN K |
| depth encoding | 16UC1 / 32FC1 | | other |
| depth registered to color | same frame | | not registered |
| valid depth pixels | >= 50 % | 20-50 % | < 20 % |
| RGB-depth offset | < 10 ms | 10-30 | > 30 |
| LiDAR-camera offset | < 20 ms | 20-50 | > 50 |
| TF base -> sensor | static transform | through moving joints (/tf): the node reads the mounting once at start | missing |
| LiDAR-camera extrinsic | present | | missing |
| IMU at rest | \|a\| = 9.81 ± 0.3, +z up in the base frame | slightly off | wrong axis or units |
| barometer | 80-110 kPa, low noise | high noise | out of range |
| time | monotonic stamps | | zero stamps / going backwards |
| disk in `map_dir` | >= 10 GB | 2-10 | < 2 |

Streams of the same rate are paired by index for the offsets (matching to the nearest stamp would hide an offset of
one whole period). If one has a frame or two more (dropped by the driver or not recorded), every shift up to that
difference is tried and the median of the pair differences is used, so a drop does not read as a one-period offset.
Streams at different rates are matched to the nearest stamp, which only tells the offset modulo the slower period
(100 ms for a 10 Hz LiDAR): the LiDAR-camera row says so, since a whole-period offset cannot be seen from the stamps.

