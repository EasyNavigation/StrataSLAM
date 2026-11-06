# Changelog

## [Unreleased]

### Removed

- The example profiles `handheld_rgbd`, `lidar_only` and `combined_aerial`: replaced by one template per kind of robot
  (see Added).
- Camera presets (`camera: {preset: ...}`, `imu: {noise_preset: ...}`, `strataslam_ros/presets/`): the profile is the
  whole configuration and a saved map keeps all of it (ADR-0053). A profile with `preset:` is refused with what to
  write instead; the values of the tested cameras are in the example profiles and in `docs/user/profiles.md`.
  `strataslam doctor --preset-dir` and the node's `preset_dir` parameter are gone.

### Added
- Live view: a second 3D tab, **3D: height**, with the map coloured by height (blue low, red high) next to
  **3D: floors**, each floor in its colour.
- README: how to run it with ROS 2 (a real robot, a recorded rosbag, next to a navigation stack). The project logo and the CI, coverage (Codecov), licence, ROS 2 and C++ badges in the README (`docs/media/logo.png`); `scripts/check_links.py` also checks the `src` and `href` of HTML tags.
- One template profile per kind of robot (ADR-0056): `ground_camera`, `ground_lidar`, `ground_lidar_camera_imu`,
  `handheld_camera_imu`, `aerial_lidar_camera_imu` (and `handheld_oak_d_lite`). The simulator bag exporter writes the
  wheel odometry when the profile enables it.
- A visual vocabulary ships with `strataslam_ros` (`vocabularies/tum_rgbd.bin`, trained on TUM RGB-D, CC BY 4.0):
  `fusion.vocabulary: default`, used by the example profiles with a camera (ADR-0054); `fusion.vocabulary` in the profile;
  `save_map` reports the loop closures and repeats that visual loop closure was off without a vocabulary.
- `output.publish_tf` and `map -> floor_<k>` (ADR-0052); profiles as templates with every option.
- `namespace:=` for the launch file; a duplicate node in the same domain refuses to run (`strataslam_msgs/Instance`).
- The node completes the TF tree: `odom -> base` when the robot has no odometry of its own, otherwise
  `floor_<k> -> odom` as a correction on top of the robot's (its own odometry in `strataslam_odom`).
- `sensors.barometer.reference_topic` (a fixed base station) is used; `sensors.barometer.settle_time_s`.
- Ctrl+C on a live run saves the map, as with `bag:=`.
- `strataslam::image_io` (libpng, libjpeg) for the TUM runner and the vocabulary trainer, which no longer load OpenCV's
  imgcodecs and with it GPL code (ADR-0047).
- Tests for every mutant the audit found surviving in the barometer, gravity, floor gate and ablation code.
- Live view of a run in Rerun (`viewer:=true`, online or offline): 3D with each floor in its colour, top view, level line
  and the log (transitions going up or down, new floors and floors recognised as already mapped). Offline, the
  node publishes the floor cloud from the bag loop and waits for the viewer; the map is unchanged by it. What the
  sensors see right now is published on `/strataslam/scan` and `/strataslam/depth_cloud` (odometry frame).
- Same-level candidates while mapping: the node checks each floor against the others as `fuse` does
  (`same_level_candidate()`, ADR-0031) in a background thread and logs the pairs that could be one level; nothing
  is applied and the map is unchanged. On `prueba_03` it reports the same three pairs as the fusion. The height
  between two floors comes from the floor manager (`SlamSystem::floor_altitude()`), not from the odometry frame,
  which misses an elevator.
- The node says at start which sensor tracks the motion and which closes loops, with `auto` resolved
  (`pipeline started: odometry from the lidar (auto), loop closures from the camera and lidar`).
- Profiles of the PAL Tiago: `tiago_lidar_camera.yaml` (the profile of the corridor rosbag) and
  `tiago_camera_only.yaml` (to study the visual tracking).
- `multifloor.mode: floors | single | planar` (ADR-0045). `planar` keeps a ground robot on one plane: no drift in
  height reaches the map (the Tiago corridor: 0.12 m in z, 0.75 m in xy with the camera, wheels and IMU).
  `enabled: false` still means `single`.
- `multifloor: {enabled: false}` also holds through a tracking loss: a session that relocalization cannot place stays
  on the only level instead of opening a floor (ADR-0042).
- The IMU is used however it is mounted (ADR-0044): its rotation, sign convention and scale are learned from the
  motion when the TF or the driver are wrong (the Tiago's base IMU is upside down, missing from its TF); its gravity
  (a Mahony filter with gyro bias) bounds the camera tracking's roll and pitch. `imu.floor_changes: false` uses it for
  gravity only. The Tiago without its LiDAR: 1 floor, 1.1 m xy and 0.43 m z from the LiDAR map (was 6 floors).
- The external odometry (`sensors.external_odometry`, from TF with `frame: odom` or from a topic) is the motion
  prior of the camera tracking: it predicts each frame and carries the pose through tracking losses, which open no
  session (ADR-0043). Before, it was parsed and chosen by `auto` but nothing used it. `odometry_from: external` is
  now an error. It also vetoes camera motions it could not have made. Scenario S14; `tiago_camera_only.yaml` uses
  the Tiago's wheel odometry (24 losses became 0).
- Ctrl+C in the offline mode (`bag:=`) stops reading the bag and saves the map of what was processed; before, the
  node ignored it until launch killed it, with nothing saved.
- Closing the Rerun window ends the viewer node, so the launch ends with the SLAM node instead of waiting forever.
- Camera and barometer in the live view. `/strataslam/camera_view` (`strataslam_msgs/CameraView`, at most 5 Hz,
  320 px wide) carries the image, the depth used and each keypoint with what the tracking did with it (used,
  rejected, unmatched, or kept for loop closures under LiDAR odometry; `VisualTracker::last_keypoint_use()`,
  `SlamSystem::keypoint_use()`); the viewer shows them next to the log. `FloorState` gains `has_baro` and
  `baro_height` (`SlamSystem::baro_height()`), drawn as a line of its own in the level line.
- The offline mode (`bag:=`) processes cameras: colour (or feature frames) paired with depth by stamp.
- ORB keypoints are detected only where the depth is valid (RGB-D): on the Tiago corridor bag the camera-only
  tracking lost 18 % of the frames instead of 42 %; TUM fr1/xyz 0.0143 m (was 0.0137), fr2/xyz 0.0152 m (0.0150).
- Layer 3 of the floor gate is a graph-membership check with its own test; the 30-inlier PnP threshold of visual loops has a test (ADR-0040).
- `scripts/run_seeds.py` and `strataslam_scenario_runner`: every scenario on many seeds, with a regression gate (ADR-0035); `floor_accuracy_all` reported next to `floor_accuracy`.
- `scripts/make_results.sh`: every published result regenerated with one command (nightly CI); `docs/results.md`.
- Real data: TUM fr2/xyz and KITTI raw 2011_09_30_drive_0027 (`strataslam_kitti_run`, full LiDAR pipeline) next to TUM fr1/xyz.
- Synthetic fusion ablation (`strataslam_fusion_ablation_synthetic`): the Euler formulation fails at the yaw wrap without GNC (ADR-0040).
- Camera-noise study (1 px at 640x480 and at 320x240) through `--pixel-sigma` / `--image` of the scenario runner.
- `scripts/install_deps.sh` and `scripts/check_install.sh` (install guide tested in clean containers), `.clang-tidy` with a CI job (ADR-0037), the `ldd` check of linked libraries in `license_audit.py` (ADR-0036).

### Changed
- A LiDAR loop closure whose GICP residual is above the fixed bound (`loop_max_mean_error`, tuned indoors) is accepted
  when the residual is at most four times that of the scan registered against its own predecessors (ADR-0057). Outdoors
  most right loops were rejected: KITTI 07 now closes its loop (ATE 0.81 to 0.41 m), 05 gains 12 loops; 06 and 09 are
  unchanged, and on the Summit bags the 10 new loops leave the fit to GPS unchanged.
- The project is now **StrataSLAM** (it was easyslam): packages `strataslam_core`, `strataslam_sim`, `strataslam_msgs`,
  `strataslam_ros`, `strataslam_tools`; the C++ namespace and headers `strataslam`; the `strataslam` command; topics
  `/strataslam/...`; environment and CMake variables `STRATASLAM_*`; default map folder `~/strataslam_maps`.
  Profiles with the old `easyslam:` header still load, with a warning; maps whose manifest has `easyslam_version`
  still read. The binary magic numbers (`ESLM`, `ESVB`) are unchanged, so saved maps and vocabularies keep loading.
- `strataslam_sim` is optional for `strataslam_ros` (a test dependency): `colcon build --packages-ignore strataslam_sim`
  builds the node, launch files and tools without the simulator; only the bag exporter and the tests on simulated
  bags need it.
- Simulated anchors are marked on the estimated map with 5 cm of error, building corners included; the reference placement of a floor uses the keyframes on its level (ADR-0034).
- Validation messages are in English (ADR-0038).
- Developer traces (`STRATASLAM_DEBUG_*`) go through the Logger (ADR-0042).
- `-Wall -Wextra -Werror` on the ROS node and tools.
- `THIRD_PARTY.md` registers the transitive libraries; `planned` rows removed; matplotlib is the optional extra `plot` (ADR-0036).
- `docs/user/install.md` rewritten and tested; `docs/data_format.md` documents `fused/`, the optional payload folders and the integrity checks.

### Fixed
- The node with a bag that lacks a topic of the profile: it crashed at the end (no pipeline was ever started). It now
  names the missing topic and the bag's topics of that type, and says that no map was saved.
- The live view drew the path of the raw odometry, never corrected by loop closures, so a closed loop looked collapsed;
  it now draws the optimized keyframe path (`/strataslam/path`). A large map (KITTI 06: 9 million points) froze the
  machine: the node publishes the floor cloud at most a tenth of the time, and the view sends at most 1.5 million points
  per floor to each 3D tab.
- `strataslam::image_io` declares GTSAM, which its header includes: it compiled against whatever GTSAM was on the
  default include path (Ubuntu's, next to the core's from ROS) and failed in the `ros:jazzy` container. The CI builds
  with two compile jobs: `-j` alone ran its runners out of memory.
- clang-tidy's `performance-inefficient-string-concatenation` in `find_duplicate_keys` (the CI's `lint-cpp`).
- The live view with ROS 2 Jazzy: `rerun-sdk==0.22.1` is installed with `pyarrow==16.0.0` (docs, CI); the latest pyarrow
  refuses to load with numpy 1.x, and `import rerun` failed with it. The launch and the viewer say why Rerun did not load.
- `scripts/install_deps.sh` installs without the recommended packages (306 MB in 407 packages before, more than an
  hour on a slow mirror), gives up on a mirror that stops answering after 30 s and retries; every CI job has a
  time limit (a hung apt download held two jobs for over an hour).
- Codecov counts a line that ran as covered (`partials_as_hits`), as the coverage gate does: gcov's exception branches
  made it report 37 % for the gate's 73 %.
- The CI's Python job installs ROS 2 Jazzy's numpy (1.26.4), since mypy 1.9 cannot read numpy 2's stubs, and rerun-sdk 0.22.1 for the live view's tests; it finds the core's executables, which the artifact keeps in subfolders.
- After the independent audit of 2026-11-04 (`audit/AUDIT_REPORT.md`; every change, with its test, in
  `audit/FIXES.md`): S11 lost its third floor in 4 of 20 seeds (the IMU stopped watching for an elevator during a
  long tracking loss); a new level was placed at the drifted absolute altitude instead of the measured climb (S03 2 of
  20); the ground plane at start-up could be a slab of wall (S02/S10 seed 7017 left 94 % of the keyframes out of the
  map); results changed with the number of threads (now bit-identical with any, ADR-0048); a malformed PointCloud2
  crashed the node; two launches in one ROS domain loaded into each other's container and hung; a node that could
  not start left its launch waiting; S10's "no worse than each modality" compared the wrong value (ADR-0046); the
  scenarios' ATE ignored keyframes left out of the map (map coverage required, ADR-0046); `converged` was always
  true in the fusion; `strataslam eval` printed an ATE of 0 when nothing matched; `.tum` stamps kept two decimals;
  the doctor accepted a NaN K, read a dropped frame as a 50 ms offset, never listened to the external odometry and
  failed robots whose sensors are mounted on joints; manifests listing a floor 100 000 times exhausted the memory;
  duplicate profile keys and extra YAML documents were ignored; `colcon test` collected no Python test with ROS
  sourced and no Open3D.
- `loop_closure_from: both` never tried the camera: it behaved as `lidar`.
- A `camera_info` without K (NaN, as some drivers publish for depth) dropped every frame silently; the node now warns.
- `strataslam fuse` rejected `--same-level` and `--stacking`, which the guide documents.
- A camera loss half-way up a ramp opened a spurious floor (S13: ATE 4.4 m on 5 of 20 seeds); the climb is now kept and split at the restart (ADR-0033).
- Floors after an elevator were tilted by the attitude drift of the restarted frontend; every restart is re-levelled from the IMU at rest or the ground plane (ADR-0033).
- `AnchorFactor` underestimated the noise of a correspondence by sqrt(2), so GNC rejected correct anchor sets (ADR-0034).
- Profiles with a camera preset and no `imu` section raised a YAML exception instead of validating; scalar sensor sections (`imu: 5`) were accepted (AUD-007, AUD-018).
- A truncated `keyframes.csv` aborted the reader; PCD/PLY files declaring huge sizes allocated before checking the file length; non-numeric PLY counts threw (AUD-008).
- `colcon build` on a cloned workspace built one package: the superbuild moved to `superbuild/` (ADR-0041). `strataslam_tools` now builds with `--symlink-install`, installs the `strataslam` command and runs its tests under `colcon test`; `strataslam_sim` builds its tests standalone.
- `multifloor.enabled: false` was ignored (ADR-0042).
- Benchmark counters printed milliseconds with the unit "s".
- An exception inside a ROS callback could stop the node; callbacks, services and the offline reader are guarded.

## [0.2.0] - 2026-11-02

First release. Highlights: one submap per floor with a three-layer gate against cross-floor loops (the
negative-control scenario S04 shows the collapse the gate prevents), floor transitions detected from the
odometry, barometer and IMU, offline fusion in SE(2)×ℝ, camera/LiDAR/combined frontends, a ROS 2 node, the
`easyslam` command line with `doctor`, and a synthetic simulator with exact ground truth. Twelve integration
scenarios pass (docs/scenarios.md). Known limits and deviations from the plan: docs/decisions.md.

### Added
- `easyslam imu-noise`: the four IMU noise values of a preset from a rosbag of the IMU at rest (Allan deviation), with plot and warnings (docs/user/imu-noise.md).
- `handheld_oak_d_lite.yaml` example profile and `oak_d_lite` camera preset (Luxonis OAK-D Lite, depthai_ros_driver topics, BMI270 noise).
- `summit_lidar.yaml` profile (Robotnik Summit XL, LiDAR only, floor height prior off).
- `fuse` lists floors that could be one level (`same_level_candidates`) and accepts the user's confirmation (`--same-level A,B`); never applied on its own (ADR-0031).
- Offline processing of a rosbag (`bag:=` in the launch file): every message in order, no dropped scans, reproducible with one thread (ADR-0029).
- `save_map` logs how many LiDAR scans the node received, to tell a run that kept up with the sensor from one that dropped scans.
- Structural registration of the floors in `fuse` (`floor_stacking`, ADR-0027) and transition uncertainty that grows with the distance travelled; `--stacking on|off` in the `fuse` tool.

By milestone of the development plan:

#### H10 · Release
- User guide, release checklist, CI (core, coverage, sanitizers, ROS 2, python, license audit, docs, nightly),
  Google Benchmark suite, coverage gate script, Zenodo metadata, `CITATION.cff` 0.2.0.

#### H9 · ROS 2 and doctor
- `easyslam_msgs`, `easyslam_node` (composable; QoS sensor data, TF `floor_<k> -> odom`, services
  `save_map`, `mark_floor_change`, `new_session`), launch file with profile validation through pybind11,
  `easyslam_sim_export_bag` (rosbags with optional corruptions), `easyslam_vocab_train`.
- `easyslam` command line: `doctor`, `map`, `fuse`, `export`, `eval`, `vocab train`.
- `easyslam doctor`: the 17 checks of the plan with hints, table / JSON, exit codes 0/1/2.
- ROS 2 tests on simulator rosbags: the node with each example profile (>= 10 Hz), `save_map`, manual floor
  change, invalid profile aborting the launch, the doctor on a correct bag and the five corrupted ones.

#### H8 · Offline fusion
- `fuse()`: one Pose4 per floor, `TransitionFactor`, `BaroHeightFactor`, `CrossFloorMatchFactor`,
  `AnchorFactor`, `FloorHeightPrior` (all with exact Jacobians and `clone()`), Levenberg-Marquardt +
  GNC (truncated least squares), marginal covariances, accepted/rejected report, warnings.
- Formulations `lie4`, `se3soft`, `euler` over identical inputs; `fusion_ablation` tool + Python script.
- `align_pose4_gnc` (Umeyama restricted to z + graduated non-convexity) instead of TEASER++ (ADR-0011).
- Restart of a frontend from the optimized pose; relocalization and altitude policies (ADR-0012..0016).
- Scenarios S07 (anchors + height prior) and S11 (three floors, stairs + elevator, full pipeline).

#### H7 · Combined camera + LiDAR
- LiDAR odometry with camera loop candidates (BoW + PnP) verified and refined by GICP; LiDAR cloud
  colorization from the camera (`colorize_points`). Scenario S10.

#### H6 · IMU and barometer
- Hypsometric height and `BaroFilter` (differential altitude with a base station), `BaroDeltaZFactor`
  (binary, exact Jacobians) in the floor graphs.
- Gravity alignment from a static IMU window or, without IMU, from the dominant ground plane
  (seeded RANSAC); gravity-attitude priors on quasi-static keyframes.
- `ImuPreintegrator` (PreintegratedCombinedMeasurements); elevator detection and height from the
  IMU alone (sustained vertical acceleration with static odometry, double integration with a
  zero-velocity update).
- Frontend restart when a height change is invisible to the odometry; relocalization restricted to
  the current floor when height sensors are active.
- Scenarios S05 (elevator + barometer), S06 (elevator + IMU), S09 (vertical flight).

#### H5 · Multi-floor
- `FloorManager` (OnFloor / InTransition / Lost) classifying stairs, ramps, elevators and flights;
  mid-landing rejection; level assignment; unit tests on synthetic signals.
- Per-floor databases and graphs, pending sessions (relocalization or a new floor with an `Unknown`
  transition), transition records with relative pose, `Backend::promote_to_floor`.
- Scenarios: S03 (collapse 0, ATE 0.027 m, height error 0.10 m), S04 negative control (collapse 86 %),
  S08 (ramp), S12 (camera blackout, session reconnected by loops).

#### H4 · Visual, one floor
- `VisualFrontend`/`VisualTracker`: depth back-projection with range and discontinuity checks,
  guided matching by projection, PnP-RANSAC with mirror-solution rejection, joint reprojection +
  depth refinement, landmark averaging, keyframe selection.
- Own binary bag-of-words (vocabulary tree, k-majority) with inverted index; ORB extractor.
- Visual loop closure (BoW per floor + PnP with >= 30 inliers) through the gated backend.
- Scenario S01: ATE RMSE 0.032 m. TUM fr1/xyz ATE RMSE 0.0137 m (docs/benchmarks.md).

#### H3 · LiDAR, one floor
- `LidarFrontend`: scan-to-local-map GICP (small_gicp), deskew, degeneracy detection from the
  translation Hessian, keyframe selection.
- Scan Context (from the paper), `FloorGraph` (iSAM2 + batch LM), `Backend` with the three-layer
  floor gate and Mahalanobis consistency check, `SlamSystem` (LiDAR path), per-floor voxel map.
- Scenario S02: ATE RMSE 0.015 m, 9 loops accepted.

#### H2 · Simulator and evaluation
- `easyslam_sim`: procedural multi-floor building (stairs, elevator with moving cabin, ramp,
  atrium), scripted trajectories, LiDAR/RGB-D/IMU/barometer/odometry with exact ground truth.
- Metrics (ATE, RPE, SE(3)/4-DoF alignment, collapse rate, height and fusion errors) in C++ and
  Python, cross-checked by a test.

#### H1 · Geometry
- `Pose4` = SE(2)×ℝ with exp/log (Taylor branch), Adjoint, left/right Jacobians and inverses,
  interpolation, SE(3) conversion with tilt check, and GTSAM `LieGroup` traits.
- `docs/math/pose4.md`; ADR-0007 (GTSAM packaging, ASan/Eigen) and ADR-0008.

#### H0 · Skeleton
- Repository layout, CMake build of the ROS-free `easyslam_core`, colcon package manifest.
- YAML profile schema, preset merging, validation with accumulated errors and typo suggestions,
  and `auto` mode resolution (`easyslam_core/config`).
- On-disk map format v1 with readers/writers and round-trip tests: manifest, transitions, floor
  metadata, keyframes and edges CSV, binary PCD/PLY, versioned binary payloads.
- `scripts/license_audit.py` and `THIRD_PARTY.md`.
- CI workflows (core, sanitizers, ROS 2 Jazzy, python, license audit, docs, nightly).

### Changed
- Point clouds default to PCD: `fuse` writes `fused/global_cloud.pcd` (it was PLY) and `easyslam export` defaults to `--format pcd` (`ply` stays available; older maps with `global_cloud.ply` still export).

### Fixed
- The vertical acceleration of the IMU is a trimmed mean: impulses of the sensor no longer open elevators that do not exist (ADR-0032).
- A slow sustained climb (under about 0.06 m/s) no longer passes for drift: the floor reference stays put while the long slope window sees a rise (ADR-0029).
- A calm stretch that lasts and covers distance after a climb is recognized as a level even when the climb is not a multiple of the floor height prior (ADR-0028).
- The relative pose of a transition is measured in gravity-aligned frames, which gives the right floor heights on ramps (ADR-0026).
- The keyframes of a ramp or stairs that changes floor are kept in the map, split between the two floors (ADR-0025).
- A ramp whose slope flickers around the calm threshold no longer closes mid-way; the floor reference follows slow drift at any sample rate; a keyframe linked to nothing no longer aborts ISAM2 (ADR-0024).
- A transition force-closed at a height that is not a whole number of floors no longer opens a floor (ADR-0023).
- Gentle ramps (below the calm vertical speed) are measured over their whole climb (ADR-0021); new scenario S13 and `BuildingConfig::ramp_length`.
- Keyframes created during a transition that ends on the same floor (terrain slopes) were dropped from the map (ADR-0020).
