# Benchmarks and sanity runs

Recorded per release. Real-data runs do not block CI (nightly or manual).

## TUM RGB-D (visual frontend, ORB 1000 features, local-map tracking, no loop closure)

`strataslam_tum_run <sequence> out.tum` — ATE RMSE after SE(3) alignment.

| Sequence | Frames tracked | ATE RMSE | ATE max | Version | Date |
| --- | --- | --- | --- | --- | --- |
| fr1/xyz | 798 / 798 | 0.0137 m | 0.0570 m | 0.2.0-dev (H4) | 2026-11-01 |
| fr1/xyz | 798 / 798 | 0.0137 m | 0.0570 m | unreleased (audit follow-up) | 2026-11-03 |
| fr2/xyz | 3669 / 3669 | 0.0150 m | 0.0397 m | unreleased (audit follow-up) | 2026-11-03 |
| fr1/xyz | 798 / 798 | 0.0143 m | 0.0551 m | unreleased (ORB on valid depth only) | 2026-11-03 |
| fr1/xyz | 798 / 798 | 0.0143 m | 0.0551 m | unreleased (audit fixes, `make_results.sh`) | 2026-11-05 |
| fr2/xyz | 3669 / 3669 | 0.0152 m | 0.0326 m | unreleased (audit fixes, `make_results.sh`) | 2026-11-05 |
| fr2/xyz | 3669 / 3669 | 0.0152 m | 0.0326 m | unreleased (ORB on valid depth only) | 2026-11-03 |

A rise above 10 % relative to the previous release is a regression.

## KITTI raw (full LiDAR pipeline: odometry, Scan Context loop closure, backend)

`strataslam_kitti_run <drive_sync> <calib_dir> out` — ATE RMSE of the optimized keyframes after SE(3) alignment
against the OXTS GPS/INS solution. KITTI is CC BY-NC-SA 3.0: used for evaluation, never redistributed.

| Drive | Length | Scans | ATE RMSE | ATE max | Loops | Time per scan | Version | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2011_09_30_drive_0027 (odometry 07) | 692 m | 1106 | 0.82 m | 2.03 m | 0 | 58 ms (1 thread) | unreleased | 2026-11-03 |
| 2011_09_30_drive_0027 (odometry 07) | 692 m | 1106 | 0.40 m | 0.76 m | 1 | 58 ms (1 thread) | unreleased (LiDAR loop gate, ADR-0057) | 2026-11-06 |
| 2011_09_30_drive_0027 (odometry 07) | 692 m | 1106 | 0.77 m | 1.79 m | 0 | 54 ms (1 thread) | unreleased (audit fixes, `make_results.sh`) | 2026-11-05 |

No loop closure was accepted at the revisit near the end of the drive: Scan Context is configured for buildings
(50 m range, indoor rooms). That is an open item, not a tuned result.

## Performance budgets (Google Benchmark, `strataslam_benchmarks`)

Host: 20-thread x86-64 @ up to 5.2 GHz, Release build, CPU scaling enabled (timings noisy).
Not measured on Jetson-class hardware (not available here).

| Benchmark | Measured (`make_results.sh`, 2026-11-05) | Budget (spec 12.8) |
|---|---|---|
| ORB extraction, 640×480, 783 features | 13.8 ms | ≤ 33 ms |
| Visual tracking (simulated RGB-D, per frame) | 0.62 ms | — |
| LiDAR frontend (64 beams, GICP, per scan) | 4.0 ms | ≤ 50 ms |
| iSAM2 update (per keyframe, 400-keyframe chain) | 0.081 ms | ≤ 20 ms |

The ORB benchmark warms OpenCV up outside the timed loop: the first call in a process costs about 2 s
(initialisation of its backends), which a single measured iteration reported once as the cost of an extraction.

The iSAM2 figure is for a mostly sequential graph with sparse loops; graphs with dense loop
closures cost more (relinearisation), see the S01–S12 reports for end-to-end runtimes.
The tracking number uses simulated feature-level frames (descriptor matching and PnP, no
image processing), so it is a lower bound for real images.

What these microbenchmarks do not show (audit AUD-048, 2026-11-04): the ORB input is a synthetic 640x480 image, the
LiDAR one a 64 x 360 simulated scan registered with 4 threads, and the iSAM2 chain has no loop closures. On real data
the LiDAR frontend is slower: KITTI raw 2011_09_30_drive_0027 (64 beams, about 120 000 points per scan) measured
82.7 ms per scan with one thread and 36 ms with four, against the 50 ms budget; see the KITTI row above (58 ms on an
unloaded machine). The budgets of section 12.8 are met by the microbenchmarks, and by the real 64-beam scans only with
several threads.

## Quality gates

- Line coverage of `strataslam_core` (gcovr, `scripts/check_coverage.py`, audit follow-up): geometry 96.1 %,
  backend/factors 91.9 %, floors 88.2 %, config 89.7 %, overall 67.5 %.
- All 145 core tests pass under ASan + UBSan (`-DSTRATASLAM_ENABLE_SANITIZERS=ON`), with no sanitizer report.
- clang-tidy (`.clang-tidy`): no warning on `strataslam_core/src`.

## Summit rosbags: StrataSLAM vs lidarslam_ros2 (informal, not reproducible, do not cite)

The scripts of this comparison were not kept in the repository and the bags are private, so it cannot be
regenerated (audit AUD-011); it stays as a development note only.

lidarslam_ros2 (branch `jazzy`, default NDT parameters of its `lidarslam.yaml` with only the frames changed, played at 1x)
against StrataSLAM (LiDAR only, default profile, played at 2x), one run each. Trajectories are aligned to the
reference `pose_graph.g2o` of the bag with a 2D ICP, so the figures are distances after alignment, in metres
(median / maximum of reference poses to the nearest point of each path). The provenance and settings of the
reference graph are unknown (its file names match lidarslam_ros2's output).

| Bag | lidarslam_ros2 | StrataSLAM |
| --- | --- | --- |
| `2024_03_21-09_57_48_wp` | 0.33 / 1.3 | 0.23 / 2.5 |
| `2024_03_21-20_07_45_wp` | 1.11 / 24.3 | 0.56 / 4.0 |
| `2024_03_11-09_58_47_0` (no reference) | between the two paths: 2.56 / 20.6 | |

Three bags and one run each do not support a ranking.
