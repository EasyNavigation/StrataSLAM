# Decision log

Short ADRs: context, decision, alternatives, consequences.

## ADR-0001 · Presets live in `strataslam_ros/presets`, not in the core (superseded by ADR-0053)

- **Context:** profiles reference camera presets (`realsense_d435i`). The core must validate
  profiles without ROS, but the spec places presets in `strataslam_ros/presets/`.
- **Decision:** `LoadOptions::preset_dir` tells the loader where `<preset>.yaml` files are. The core
  embeds no presets, so there is one source of truth. Preset names must match `[a-z0-9_]+`
  (no path traversal). Core tests point at `strataslam_ros/presets`.
- **Alternatives:** embed presets as C++ strings (duplicated data); copy presets into the core.
- **Consequences:** the launch file and CLI must pass the installed preset directory.

## ADR-0002 · Sections default to enabled when present

- **Context:** the spec example lists `enabled` for every sensor but does not say what a section
  without it means.
- **Decision:** a present sensor section is enabled unless `enabled: false`; an absent section is
  disabled. This keeps lidar-only profiles short.
- **Consequences:** an empty `imu:` is an IMU with its defaults. (Before ADR-0053, `camera: {preset: realsense_d435i}` was a
  complete camera definition.)

## ADR-0003 · DBoW3 is not used; the bag-of-words is implemented in-house

- **Context:** the plan asked to verify DBoW3's license in H0. Its `LICENSE.txt` (checked
  2026-11-01) is a modified BSD with a fourth clause requiring notification of the original author on
  redistribution. That is not among the allowed licenses.
- **Decision:** do not depend on DBoW3; implement a hierarchical vocabulary tree and inverted index
  from the original papers in milestone H4. Removed from the dependency table.
- **Consequences:** more work in H4, no license risk.

## ADR-0004 · Binary map files use little-endian only

- **Context:** the map format must be portable and readable without StrataSLAM.
- **Decision:** all binary files (payloads, PCD, PLY) are little-endian; the build fails on a
  big-endian host (`static_assert`). Text files use shortest round-trip number formatting
  (`std::to_chars`), so reads reproduce doubles exactly.

## ADR-0005 · Edge CSV layout

- **Decision:** `edges.csv` has columns `type,from,to,measurement,info_dim,information`; vector
  fields are `;`-separated; `information` is the row-major upper triangle with
  `info_dim*(info_dim+1)/2` entries. The measurement dimension may differ from `info_dim`
  (e.g. 7 values for a Pose3 between factor with a 6×6 information matrix).

## ADR-0006 · Provisional IMU noise values

- **Decision:** the D435i preset uses nominal BMI055 noise densities (`TODO(decision)`). They must
  be replaced with a calibrated Allan-variance result before H6 acceptance tests are tuned.

## ADR-0007 · GTSAM packaging and Eigen/ASan ABI

- **Context:** Ubuntu 24.04's `libgtsam-dev` (4.2.0) exports a CMake target `CppUnitLite` whose
  `libCppUnitLite.a` is not shipped, so `find_package(GTSAM)` fails. GTSAM also pulls in Boost.
- **Decision:** use `ros-jazzy-gtsam` (same 4.2.0, complete CMake export) as the reference install:
  `-DGTSAM_DIR=/opt/ros/jazzy/lib/cmake/GTSAM`, with `/opt/ros/jazzy/lib/x86_64-linux-gnu` on
  `LD_LIBRARY_PATH` for libmetis. The core only needs the *package*, not a ROS runtime. On plain
  Ubuntu CI, `libgtsam-dev` plus an empty placeholder `libCppUnitLite.a` is used (never linked).
- **ASan:** with sanitizers Eigen stops assuming glibc `malloc` is aligned and uses its own aligned
  allocator, while the non-instrumented `libgtsam` frees with `free()` (bad-free). The sanitizer
  build defines `EIGEN_MALLOC_ALREADY_ALIGNED=1` to keep both sides consistent.
- **Consequences:** GTSAM becomes a hard dependency of the core from H1 (geometry traits).

## ADR-0008 · `Pose4` derives from `gtsam::LieGroup`

- **Decision:** one header for the group, its algebra and the GTSAM traits, instead of a separate
  adapter. The core already depends on GTSAM and a single type avoids conversion layers.
- **Note:** GTSAM's stock `BetweenFactor`/`PriorFactor` use `J_r⁻¹(error) ≈ I` unless built with
  `GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR`; their Jacobians are exact only near zero error. Our own
  factors (H6/H8) implement exact chain-rule Jacobians; `Pose4Gtsam.ExactErrorChainJacobians*`
  covers the building blocks.

## ADR-0009 · Statistic for the PnP accuracy criterion

- **Context:** spec 12.2 asks for a pose error below 1 cm and 0.2 degrees for PnP with 20 % outliers
  and 1 px noise. Over random geometries the error is noise-limited (mostly along the optical axis):
  with 500 correspondences (the simulator's keypoint budget; 400 inliers) the median is ~6 mm and
  2 of 50 draws land at 1.0-1.1 cm.
- **Decision:** the criterion is applied to the RMS error over 50 seeded draws (translation < 1 cm,
  rotation < 0.2 degrees), plus hard bounds on every single draw (2 cm, 0.5 degrees). The spec's
  numbers are unchanged; what it left open is the statistic. No algorithmic threshold (RANSAC,
  matching) was relaxed. Measured: RMS translation ~0.6 cm, RMS rotation ~0.08 degrees.
- **Consequence:** the real tracker also uses depth, which constrains the optical axis better than
  pixel-only PnP; scenario ATE thresholds (S01, S03, ...) are the end-to-end check.

## ADR-0010 · XFeat/ONNX Runtime not built in this milestone

- **Decision:** `FeatureExtractor` is an interface with ORB as the only implementation. XFeat needs
  exported ONNX weights and ONNX Runtime, neither available offline here; both stay optional behind
  `STRATASLAM_WITH_ONNX` (not defined). The system works with ORB, as the spec requires.

## ADR-0011 · TEASER++ is replaced by a Pose4-specialised GNC estimator

- **Context:** the plan lists TEASER++ for robust registration of anchors. Its max-clique / certification
  stack is heavy for registrations with 4 degrees of freedom and a handful of correspondences.
- **Decision:** `align_pose4_gnc` (graduated non-convexity with a truncated least-squares loss, Yang et
  al., RA-L 2020 - the estimator family behind TEASER's robust steps) written from the paper;
  `align_pose4_closed_form` is the Umeyama alignment restricted to rotations about z. Test: it recovers
  the transform with 50 % gross outliers over 20 seeds. TEASER++ is not vendored.
- **Consequence:** THIRD_PARTY lists TEASER++ under "Rejected". If a full 6-DoF robust registration
  is ever needed (e.g. LiDAR loops from large initial errors), TEASER++ can be added then.

## ADR-0012 · LiDAR degeneracy: no pinning to the prediction by default

- **Context:** in a degenerate direction GICP's solution is poorly determined; the first implementation
  kept the constant-velocity prediction along it. In scenario S11 that diverged to 14 m at the turn of the
  U-shaped stairwell (the velocity model extrapolates through the turn), while the unpinned GICP solution
  stayed within 0.25 m.
- **Decision:** `pin_degenerate` is off. `Degraded` still (a) widens the odometry covariance of the
  keyframe by `degraded_scale` in the backend and (b) vetoes LiDAR loop closures. Along the weak direction the
  registration is bounded by a physical acceleration instead (ADR-0061).

## ADR-0013 · Relocalization policy after a tracking loss

- **Decision:** a loss opens a new session whose floor is pending. With a barometer or an IMU a floor
  change cannot go unnoticed, so the session stays on the current floor (relocalization is only tried
  there, and failure keeps the floor). Without height sensors every floor is a relocalization candidate;
  failing to relocalize opens a new floor with an `Unknown` transition, as the plan prescribes for an
  elevator ride that nothing can observe. Identical floors, an elevator and no height sensors are
  fundamentally ambiguous: the floors may be merged by relocalization against the wrong floor.

## ADR-0014 · Floor altitude and the elevator

- **Decision:** the floor manager owns the altitude. For an elevator (height seen by the barometer or
  the IMU, not by the odometry) the frontend is restarted *without* shifting its z; the manager adds
  the measured height, and the transition record's relative z is that measurement. Shifting both would
  count the height twice (found in S05).

## ADR-0015 · Height of continuous transitions and the floor height prior

- **Context:** in S11 (camera + LiDAR; odometry from the LiDAR as the profile table dictates) the
  LiDAR underestimates the climb of a U-shaped stairwell by 0.17-0.3 m (weak z constraint in a narrow
  shaft; not cured by a finer voxel or a 32-beam sensor, which only halve it).
- **Decision:** the z standard deviation of stairs/ramp records is 0.15 m, and `floor_height_sigma`
  defaults to 0.10 m. The prior is the user's statement that floors are about `floor_height_prior`
  apart; it corrects such a bias, while the barometer (0.05 m) or good odometry still dominate it.
  A profile with `floor_height_prior: null` gets no height prior and keeps the raw bias.

## ADR-0016 · S11 is evaluated with structural anchors; the figures without them are reported

- **Context:** with LiDAR odometry (the profile table's choice for camera + LiDAR) the stairs transition
  carries 0.14 m / 0.9 degrees of error, so the transition-only fusion misses the S11 criterion
  (4-DoF error <= 0.10 m and 1 degree).
- **Decision:** S11 marks the corners of the stairwell and of the elevator shaft (structural anchors that
  repeat on every floor, the plan's `AnchorFactor`) and fuses with them. The scenario report also prints
  `without_anchors_trans_err_m` / `without_anchors_rot_err_deg` so the limitation is visible.

## ADR-0017 · The simulator works at feature level; its bags carry features

- **Context:** the simulator generates keypoints with descriptors and depth (the plan's "camera at feature
  level"), not rendered images. The ROS node must still be tested on simulator rosbags.
- **Decision:** `strataslam_msgs/FeatureFrame` carries keypoints + 256-bit descriptors; the node has a
  `features_topic` parameter (not a profile key, so the profile schema stays the one of the plan). With it
  the node reads those features and the depth image; without it, it extracts ORB from the RGB image
  (the production path). The simulator bag also holds a flat grey RGB image so that topic rates, the
  RGB-depth synchronisation and the doctor see a complete camera.
- **Consequence:** the ORB-from-image path of the node is exercised by the TUM sanity run
  (`strataslam_tum_run`, same extractor and frontend) but not by a ROS test.

## ADR-0018 · Packaging

- `strataslam_core` and `strataslam_sim` are shared libraries (the static build would have to export its private
  dependencies: small_gicp, OpenCV). The installed targets are `strataslam::core` and `strataslam::sim`, the same
  names as inside the source tree.
- `strataslam_tools` ships `setup.py` only as a colcon shim; `pyproject.toml` has no `requires-python` because
  colcon's setup.py parser cannot read it (Python 3.12 is fixed by CI and the docs).
- The launch file validates profiles through a pybind11 module (`strataslam_ros._strataslam_py`) that calls the
  core's own loader, so the node, the launch file and the tests cannot disagree.

## ADR-0019 · What the release does not include

- XFeat/ONNX Runtime (ADR-0010), TEASER++ (ADR-0011), a Jetson benchmark run, clang-tidy and the GitHub CI
  run could not be done in the development environment; the Zenodo DOI needs the repository owner.
  See `docs/release_checklist.md`.

## ADR-0020 · Transitions that end on the floor they started on keep their keyframes

Found with a real outdoor rosbag (Summit robot, LiDAR only): a gentle slope of the
terrain (about 1 m of rise over 30 m) opened a floor transition (`depart_dz` 0.30 m), the manager never saw the
calm needed to close it, and it was force-closed after `max_transition_s` (120 s) as a ramp 0→0. Keyframes created
while a transition is open belong to no floor by design, so 129 keyframes (28 m of the route) were missing from
the map.

- When a transition closes with `from == to`, continuous tracking and no frontend reset, it was not a floor
  change: no transition record is written and its keyframes are promoted to that floor, chained to the last
  keyframe before it by their odometry (`Backend::promote_to_floor(..., bridge_is_odometry=true)`).
- Transitions between different floors are unchanged (stairs, ramps, elevators, aerial: S03, S05–S11).
- Not changed: the departure threshold itself. A route with long gentle slopes still pauses floor assignment for
  up to 120 s; the map is complete now, but those keyframes are only part of the graph after the transition ends.
- Regression tests: `FloorManager.GentleSlopeThatReturnsToTheSameLevelIsNotAFloor` (manager level) and S13 (below). The
  keyframe promotion in `SlamSystem` is only covered by the Summit rosbag run, not by a simulator scenario.

## ADR-0021 · A gentle ramp slower than `calm_vz` is still a floor change

Found while adding the gentle-slope scenario S13 (2.9 degrees, 30 m flights, 3 m rise; `BuildingConfig::ramp_length`).
At 1 m/s such a ramp climbs 0.05 m/s, under `calm_vz` (0.06), and the slow tracking of the floor reference
(2 % per sample) follows it with a lag below `depart_dz`. The transition therefore opened late, on the second flight,
measured its height from the already-followed reference (2.2 m instead of 3.0 m) and closed at the first calm.

- `calm` now also requires that the climb per metre travelled over the last `calm_slope_window_s` (2 s) stays under
  `calm_slope` (0.04, 2.3 degrees), measured only while the robot moves at least 1 m in that window.
- The floor reference stops following the odometry while that climb is sustained, so the departure and the
  measured `dz` start where the climb started. After an elevator (restarted odometry, which keeps creeping in S11)
  the reference follows regardless for 30 s.
- Limits: a gradient between 0.025 and 0.04 (1.4 to 2.3 degrees, like the Summit terrain slope) is still treated as
  drift/terrain; with S06 and S11 the creep of the odometry after an elevator is of the same size as a gentle
  ramp, which is why the threshold is not lower. The keyframes of a 60 s ramp still belong to no floor until it closes.

## ADR-0022 · LiDAR loop gate left as is after the Summit experiment

On the Summit `_wp` bag (`STRATASLAM_DEBUG_LOOPS=1` prints every candidate) Scan Context proposes loops and GICP
fitness is 0.9 or more for all 132 verifications, but only one passes `loop_max_mean_error` (0.12 m²): real
outdoor data has residuals of 0.13 to 0.35. Relaxing it to 0.4 accepted 6 loops, three of them wrong (one pulled
part of the map 23 m away from the reference). The gate is therefore kept. A per-sensor, noise-aware
threshold (for example scaled by the registration voxel) is open work and needs more than one bag to tune.
Superseded by ADR-0057.

## ADR-0023 · A forced transition close at an invalid height stays on the floor

The Summit bag `2024_03_11` (outdoor, no reference map) opened a second floor on a terrain slope: the transition
never calmed, was force-closed after 120 s with dz = -1.47 m (half a floor) and `assign_level` created a level at
that altitude. 183 keyframes were left without floor and the map split in two. When the close is forced and `dz` is not
a whole number of floors (`valid_level_change`), the manager now keeps the floor (`to == from`); `SlamSystem` then
keeps the keyframes (ADR-0020). Test: `FloorManager.ForcedCloseOfAnInvalidHeightStaysOnTheFloor`. An outdoor route with
a real change of about one floor height would still be taken as a floor; `multifloor` can be disabled in the profile.

## ADR-0024 · Ramps with a flickering slope, drift-following reference, unlinked keyframes

The Summit bag `prueba_00` (building with two interior floors joined by two ramps through an outdoor stretch) exposed
three faults, seen over repeated runs:

- A gentle ramp whose odometry z alternates between about 1.5 and 3.5 degrees dropped under `calm_slope` on the flat
  parts, so the transition closed mid-ramp and another opened. The transition is now also kept open while the climb over
  the last `slope_long_window_s` (8 s, at least 3 m travelled) exceeds `calm_slope_long` (0.025) and the last 2 s still
  rise (half that slope), so the flag does not linger once the ramp ends. Test:
  `FloorManager.RampWhoseSlopeFlickersAroundTheThresholdIsOneTransition`.
- The floor reference followed the altitude at 0.02 per sample, i.e. a 5 s time constant only at 10 Hz. At the 2 Hz of
  the keyframe rate a drift of 0.02 m/s lagged by 0.5 m, over `depart_dz`, and opened a false transition after the ramp.
  The reference now tracks with a 5 s time constant in time.
- That false transition ended with a keyframe linked to nothing and ISAM2 threw `IndeterminantLinearSystemException`,
  aborting the node. `FloorGraph::add_keyframe` adds a weak prior when no factor links the keyframe.

The split of a long ramp into one or two transitions still varies from run to run (the sum of the `delta_z` of the two
ramps was 4.4 to 6.3 m over six runs at 2x playback; lidarslam_ros2 reaches about 7.3 m): the outdoor stretch between the
ramps is a floor of its own for the manager, not a mezzanine (open work).

## ADR-0025 · Keyframes of a ramp or stairs stay in the map

Keyframes created while a transition was open joined no graph; ADR-0020 only rescued them when the transition ended
on the same floor. On the Summit bags `prueba_00` and `prueba_03` every real ramp lost its keyframes (about 180 and
510 of roughly 800), so the map had no climb and the chained path came out short. When a ramp or stairs tracked
continuously ends on another floor, the keyframes of the climb are now split by altitude: the lower half goes to
the floor it left, chained forward by odometry from the last keyframe there, and the rest to the floor it arrived
at, chained backwards from that floor's first keyframe (`Backend::promote_transit_before`), so the floor frame keeps
its origin on the floor. They are not loop candidates (a scan taken mid-climb is a poor place). Elevators and
restarted sessions keep dropping them. In the scenario metrics a keyframe without a true floor (mid-climb) is counted
as correct on either floor. Result: `prueba_03` went from 324 to 883 keyframes in the map, `prueba_00` to 869.

## ADR-0026 · The relative pose of a transition is measured in gravity-aligned frames

`TransitionRecord::relative` was the odometry between the last keyframe on the old floor and the first on the new
one, expressed in the full 3D body frame of the first. A ramp starts with that keyframe already tilted by the slope,
so with 16 m of travel the climb leaked into x and the height came out wrong (about -0.3 m for a real 2.4 m
ramp, `prueba_00`, `prueba_03` and the Gazebo garage). `fuse` takes `relative` as a Pose4 (yaw and translation in the
yaw-only frames), so it is now computed in those frames. Against the true poses of the Gazebo parking garage
(`/ground_truth`, three floors, 2.33 m apart) the floor frames of `fuse` came out at 0.05 and 0.15 m in xy, 0.04 and
0.01 m in z and 0.3 and 0.7 degrees from the truth, where the heights had been wrong by 2.6 and 5.0 m. The scenario
check for S08/S13 compares the recorded relative height with the climb; the simulated ramps are too gentle to fail
on the old code, so the garage run is the real regression evidence.

## ADR-0027 · Structural registration of the floors (`floor_stacking`)

Floors of a building share structure along the vertical, so `fuse` registers the structure (points between 0.4 and
3 m above the floor frame, rasterized at 0.4 m) of every pair of floors that has a floor cloud, in a window around
the pose chained through the transitions (25 m, +-30 degrees), by correlating rasters (`cv::matchTemplate`) and
refining on a finer raster. The result becomes a cross-floor constraint only if the structures overlap (score at
least 0.35 of the smaller one) and the best pose is clearly better than the best at a distinct pose (ratio 1.25);
otherwise a warning says why it was not used. Nothing in it depends on a particular shape, and floors that do not
stack (an outdoor level between two interior ones) or repetitive layouts simply contribute nothing. To let the
constraint count when it is good, the covariance of a transition grows with the distance the odometry travelled
(`transition_drift_per_m` 3 %, `transition_yaw_drift_per_m` 0.0005 rad/m). On the Gazebo garage, whose floors are
nearly identical, every pair was declared ambiguous (best 0.996 against 0.97), correctly, and the odometry chain
alone was already within 0.15 m; `--stacking off` in the `fuse` tool disables it. Tests: `stacking_test.cpp`.
On the Summit bags `prueba_00` and `prueba_03` (outdoor stretches between the floors) every pair was also declared
ambiguous (best 0.57 against 0.57, 0.68 against 0.62, 0.66 against 0.59), so on real data the registration has not yet
contributed a constraint; whether a sharper score (finer cells, weighting of distinctive features) separates them
is open work. The height fix of ADR-0026 is what changed those results.

## ADR-0028 · A plateau is a level, whatever its height

A change of height was a level only if, with a `floor_height_prior`, it was within 0.4 priors of a whole number of
floors (or, without one, at least `min_vertical_change`). Buildings have levels that are not whole floors (split
levels, a car park's mezzanine): a climb of 4.2 m followed by one of 1.5 m with a prior of 3 m kept the first by 0.01
m and dropped the second, and a run that measured 4.6 m merged both into one transition that ended as terrain. A
transition now also closes as a level when the robot stays calm for `plateau_s` (20 s) while travelling
`plateau_travel_m` (10 m) after a change of at least `plateau_min_change` (1 m): a stair landing lasts seconds, a
floor is lived on. Levels closer than half of that change are one. Tests:
`ALongPlateauAfterAClimbThatIsNotAWholeFloorIsALevel`, `AShortLandingIsNotAPlateau`,
`ALevelNearAnExistingOneIsNotANewLevel`, and the forced-close test now feeds terrain that never calms.

On real bags without ground truth (the Summit's) the split of the same climbs varied from run to run, so no single run
says which configuration is right; the Gazebo garage, which has one, is unchanged (0.05 and 0.15 m from the truth).
The thresholds are heuristics: open work.

## ADR-0029 · Offline bag processing, and the floor reference does not follow a sustained climb

Runs of `ros2 bag play` against the node were not reproducible even at 1x: the subscription queue drops the scans
that arrive while the previous one is being processed, and which ones depends on the timing (one thread processes
about 4.5 scans/s for a 10 Hz LiDAR). The node has a `bag` parameter (launch argument `bag:=`) that reads the rosbag2
itself and processes every message in order, with TF taken from the bag. With one thread it is deterministic; with
several, rarely, a floor decision differs (floating-point sums in a different order).

With it, `prueba_00` showed a regression of ADR-0024: the reference altitude follows the signal with a 5 s time
constant, so a climb slower than 0.06 m/s (0.30 m of departure over 5 s) never leaves the reference behind and passes
for drift; the second ramp of the bag (0.03 per metre) was never detected. The reference is now also frozen while the
long slope window sees a sustained rise. Test: `ASlowSustainedClimbIsNotFollowedByTheFloorReference`.

Offline, one thread, `prueba_00`: 3 levels (+2.9 and +1.9 m) with and without the height prior; `prueba_03`: 3 levels
with the prior and 4 without (-1.0, -3.4 and +2.2 or +1.9 m); the Gazebo garage: 4 ramps (2.38, 2.41 to 2.46, -2.47 and
-2.42 m), the same as in real time.

## ADR-0030 · Drift in z cannot be told from a gentle ramp with the odometry alone (negative result)

On `prueba_03` the same level (the L-shaped lateral and the cross-shaped interior) came out as two floors 1.7 m apart:
the scan-matching odometry drifts in z by 0.5 to 0.7 m in a minute while the robot stays on one level. A real ramp tilts
the robot by its slope, so the tilt of the base against the slope measured along the travel direction was a candidate
test (`STRATASLAM_DEBUG_ALT` prints both). It does not work: over five stretches (a steep ramp, two gentle climbs and two
drifts) the tilt of the odometry agrees with the measured slope in all of them (for example +0.036 measured and +0.046
expected on the climb, -0.035 and -0.029 on a drift), because the drift tilts the whole map rigidly. An absolute
reference of gravity is needed. Enabling the IMU of the Gazebo garage bag instead produced eleven floors from
spurious elevator detections (the odometry is static at the foot of a ramp while the vertical acceleration is
sustained), so the IMU path is not a remedy yet; both facts are open work.

## ADR-0031 · Same-level candidates are reported, never applied

Registering two floors by their structure cannot tell one level seen twice from two identical levels stacked
(ADR-0027): on `prueba_03` the pair that matches best (0 and 3, residual 0.096) is two real levels. So the fusion
lists the pairs whose structure is compatible (score at least 0.3, ambiguity allowed) and whose heights differ by
at most `same_level_max_dz` (2 m, under a storey) in `same_level_candidates` and in the warnings, and applies
nothing. `FusionOptions::same_level` (`--same-level A,B;C,D` in the tool) is the user's confirmation: the frames get
the same height (sigma 0.10 m) and the height of the transition between them is released, keeping x, y and yaw.
On `prueba_03` it listed three candidates (0-1, 1-3, 2-3), one of them the pair that the user confirmed
(2 and 3 are the same level: heights -3.97 and -2.32 m before, -3.97 and -3.96 after); floor 3 then lies on floor 0 in
x and y, with the circle of the building in common, as the building has it. The other two are false: a candidate list
is advice, not a classification.

The node runs the same check while mapping (`same_level_candidate()`, the test that `fuse` applies to each pair): every
20 keyframes on a floor, that floor against the others within `same_level_max_dz` of it, and every pair at the end of a
bag. The relative pose comes in the plane from the odometry (the TF `floor_<k> -> odom` of each floor) and in height
from the floor manager's level altitudes (`SlamSystem::floor_altitude()`, the `altitude_m` that `fuse` reads): the
odometry frame misses the height of an elevator or of a change while lost, and with it the two identical floors of
the simulated elevator (S05, 2.9 m apart by the barometer) were reported 0.05 m apart. The clouds are copied
in the node's thread and registered in a background one, so the map does not change (keyframes identical with and
without it on `prueba_03`); a candidate is a warning in the log, reported once, and nothing is applied. On `prueba_03`
it reported the same three pairs as the fusion, each about two seconds after the transition to its later floor ended.

## ADR-0032 · The vertical acceleration of the IMU is a trimmed mean, not a mean

Enabling the IMU on the Gazebo garage produced eleven floors from spurious elevators (ADR-0030). The raw samples of the
simulated IMU are impulsive: while the robot rolls, single samples read 79 m/s^2 (or 7.6 instead of 9.8) in the
vertical axis because of wheel contacts, and the elevator detector took the 1 s mean (threshold 0.25 m/s^2): one
impulse among 30 samples adds 2.6 m/s^2. `FloorManager::az_mean` is now a mean without the 20 % highest and lowest
values, and the double integration of an elevator ride first passes a 5-point median. Tests:
`ImpulsesOfTheAccelerometerOnAParkedOrRollingRobotAreNotAnElevator` (fails with the plain mean) and
`AnElevatorIsStillMeasuredWhenTheAccelerometerHasImpulses`; the walking-bobbing test rules out the plain median, which
reads a sample of a symmetric oscillation instead of zero.

Effect on the garage with the IMU enabled (offline, one thread): 11 floors and 6 elevators before; now no elevator,
but only two of the four ramps are closed (+2.22 and +2.04 m, the floor frames 0.19 and 0.09 m in xy and 0.13 and 0.01 m
in z from the truth) against four without the IMU. Two causes remain: (1) at the foot of a ramp the real climbing
acceleration arrives before the odometry sees the height, so the first departure is still taken as unobserved (an
elevator-like restart of the frontend), and (2) the vibration of the simulated robot keeps the 1 s trimmed mean above
the calm threshold (0.10 m/s^2), so the descents never close. Open work.

## ADR-0033 · A tracking loss inside a climb, and every frontend restart, keep the geometry

The independent audit (AUD-026) ran the scenarios on 20 seeds other than the CI one; S08 and S13 (ramps, camera)
failed on 5 each and S07/S11 (elevators) on 8 and 13. Three defects of the pipeline, all after a frontend restart:

1. **A loss half-way up a ramp was taken as an unobservable floor change.** With no barometer or IMU, a session
   that cannot relocalize opens a new floor one `floor_height_prior` higher (`open_new_floor`). When the camera is
   lost for 2 s on a ramp the odometry was measuring, that put the altitude 1.5 m too high and opened a third floor
   for the rest of the ramp (S13: ATE 4.4 m). Now a pending session that starts inside an observed transition
   (`FloorManager::in_observed_transition`) is not resolved by relocalization or by opening a floor; its keyframes
   join the climb's transit keyframes when the transition ends, the climb is split at the restart (each part chained
   by its own odometry to the floor it touches, ADR-0026) and the transition keeps its relative pose with the wide
   covariance of a non-continuous one.
2. **The restart pose inherited the attitude of the degraded frames before the loss.** The frontend restarts from
   the last pose tracked as `Good`, not from the last (degraded) one, and the vertical motion during the blind time
   is extrapolated with the vertical speed of the last second inside a climb (zero elsewhere).
3. **A restarted floor was tilted.** The attitude at a restart carried the drift of the previous session (a LiDAR
   inside an elevator cabin drifts several degrees), and every keyframe of the new floor inherited it: a tilt of
   0.5-5 deg that a 4-DoF fusion cannot remove (floor heights 0.15-0.2 m off over 26 m). Every restart is now
   re-levelled: roll and pitch from the IMU when it is strictly at rest for 1 s (gyro < 0.02 rad/s, |a| within
   0.15 m/s^2 of g), else from the horizontal plane of the current depth image or scan, never from a plane on stairs
   or a ramp; yaw and position are kept. A ground-plane correction above 3 deg, or an IMU one above 10 deg, is not
   believed. Not at the start of an elevator ride (the cabin is about to accelerate: a 7 deg false correction was
   measured there), only at its end.

Effect over seeds 1001-1020: S08 15 -> 20 passes, S13 15 -> 19, S11 7 -> 18 (with ADR-0034), S07 12 -> 14; no
scenario lost a seed. The figures are in `docs/results.md`.

## ADR-0034 · Simulated anchors are marks on the estimated map; the reference placement uses level keyframes

The audit (AUD-004) found that the S07/S11 anchors were built from the ground-truth placement of each floor
(a 4-DoF fit of its keyframes to the truth) plus 2 cm of noise: the fusion received the answer. `make_anchors` now
simulates a user marking corners on each floor's cloud. Each physical corner is expressed relative to the keyframe of
that floor that saw it from closest and carried into the map by that keyframe's *estimated* pose, so the map's own
distortion is in the mark. Marking error: 5 cm. A corner no keyframe of a floor saw within 6 m is not marked. The
ground truth only says which physical corner is clicked, which the user knows. The marked corners are those a user
would pick: the elevator door and shaft, the stairwell in S11 and the four inner corners of the outer walls. S07
also reports the elevator shaft alone (`elevator_anchors_only_*`) and S11 the transitions alone (`without_anchors_*`).

Two defects surfaced on the way:

- `AnchorFactor` used the marking sigma for the difference of two marks, each with that error: the chi-square of a
  correct set doubled and GNC rejected it as an outlier (an elevator anchor set with residual 6.7 against 4 expected).
  The noise of the difference is sqrt(2)*sigma.
- The reference placement of a floor (`floor_frame_in_building`) fitted every keyframe of the floor map, including
  those recorded in an elevator cabin or on a stair before the change of floor was noticed. After an unobservable
  ride those keyframes are metres from the level, and they moved the reference height of S07's second floor
  0.17 m. The fit now uses the keyframes that were on the floor's level. Mislabelled keyframes are still measured,
  by `floor_accuracy` and the collapse rate.

## ADR-0035 · Multi-seed acceptance; the floor-accuracy metric

The thresholds of spec 12.4 are unchanged and every push runs S01-S14 on seed 42 with all of them. The audit
showed that one seed is not representative (AUD-026), so the nightly job runs every scenario on seeds 1001-1020
(`scripts/run_seeds.py`, one thread, bit-exact) and publishes the pass rate, mean, deviation and worst value of
every criterion with each failure and its seed (`docs/results.md`). The nightly fails when a scenario passes on
fewer seeds than the rate recorded in `docs/seeds_baseline.json`, or when the ATE of a real-data run rises more
than 10 % above `docs/real_data_baseline.json` (spec 12.6). That is a regression gate, not a claim that the
remaining failures are acceptable: `docs/results.md` lists each one with its seed and the criterion it failed.

`floor_accuracy` (S09, at least 0.95) ignores the keyframes within 2 s of a change of the true floor label: the
manager needs a moment to notice that a transition began or ended. That exclusion was undocumented (AUD-005). The
metric without it, `floor_accuracy_all`, is now reported next to it. On seed 42 both are 1.00 in S09. Keyframes whose
true floor is "in transition" count as correct on either floor (ADR-0025).

## ADR-0036 · Licenses of transitive libraries and of optional tools

The audit (AUD-014) listed libraries that `libstrataslam_core.so` loads without a row in `THIRD_PARTY.md`: Boost,
TBB, METIS, BLAS/LAPACK/ATLAS, plus OpenCV's system dependencies. They now have rows with status `transitive`, and
`scripts/license_audit.py --ldd <artifact>` checks every library the built core loads against the registry (CI
`core` job). For those libraries only, three permissive licenses equivalent to the allowed ones are accepted:
BSL-1.0 (Boost), Zlib and X11. The GCC runtime (libstdc++, libgcc_s, libgomp, libgfortran, under the GCC Runtime
Library Exception) and glibc are toolchain, not tracked. matplotlib (PSF-based license) is accepted only as an
optional plotting tool (`pip install strataslam_tools[plot]`), never linked nor distributed. The `planned` rows
(ONNX Runtime, XFeat, scipy) were removed because nothing uses them (AUD-015).

## ADR-0037 · clang-tidy configuration

`.clang-tidy` enables `bugprone-*`, `performance-*` and `clang-analyzer-*` as errors on the core (CI job `lint-cpp`,
zero warnings). Three checks are configured off, with reasons. `bugprone-easily-swappable-parameters` is an
API-design heuristic. `performance-enum-size` would change the public ABI of enums that are never stored in bulk.
The integer-to-floating-point part of `bugprone-narrowing-conversions` is off because nanosecond timestamps times
1e-9 and counts in means are intended throughout the numeric code. Floating-to-integer and unsigned-to-signed
conversions are still checked, and were made explicit. The 87 findings of the first run were fixed. Among them,
`solvePnPRefineLM` failures now restore the RANSAC pose instead of keeping a half-modified one.

## ADR-0038 · Validation messages in English

The specification gives Spanish examples for the validator (8.2) while it asks for English code and API
documentation (3.8); the messages were Spanish and the user guide quoted them inside English text (AUD-023). They are
English now (`config error: sensors.camra: unknown key; did you mean "camera"?`), like every log of the node.

## ADR-0039 · Point clouds are written as PCD by default

`strataslam export` and `strataslam fuse` write binary PCD (`fused/global_cloud.pcd`) instead of the PLY of spec 7.3
(AUD-009). PCD is the format of PCL and of the ROS tools (`pcl_ros`, RViz's loaders); PLY stays available with
`strataslam export --format ply` (the per-floor clouds; `fuse` writes the global cloud as PCD only, corrected after the
2026-11-04 audit, A6-09, which found this ADR claiming `fuse --format ply`). Open3D reads both (`test_cloud_compat.py`).

## ADR-0040 · Layer 3 of the floor gate is graph membership

The third layer of the floor gate re-compared the floor labels that layer 2 had just compared, so it could not be
reached by any test and removing it went unnoticed (AUD-001, mutant M2). It is now a check of what a factor
actually joins: `FloorGraph::add_loop` refuses a loop whose endpoints are not both nodes of that graph, whatever the
labels say. Test: `BackendGating.Layer3FloorGraphRefusesAFactorToANodeItDoesNotHold`; the mutant is now killed.
Likewise the 30-inlier PnP threshold of a visual loop (spec 9.7) moved into `verify_loop_pnp` with its own test
(AUD-025, mutant M13 killed).

The fusion ablation (spec 12.5) gained synthetic floor graphs with an exact truth
(`strataslam_fusion_ablation_synthetic`), because S07/S11 never exercise the differences between the formulations
(AUD-027). Cases: two floors turned by about 180 deg, ten chained floors, two sparse anchors. Without GNC, the
`euler` formulation, which does not wrap the yaw residual, fails at the wrap: 38 deg mean and 98 deg worst rotation
error, against 0.28 deg for `lie4` and 0.18 deg for `se3soft`. With GNC it survives by rejecting the constraint
across the wrap. In the other synthetic cases the three agree to within noise. On S11 with the realistic anchors of
ADR-0034 (seeds 2001-2020), `lie4` has the lowest mean rotation error: 0.19 deg against 0.69 (`se3soft`) and 0.75
(`euler`). The translation differences are within one standard deviation, and on S07 the three agree.
`docs/results.md` has the tables.

## ADR-0041 · Repository layout for colcon

The development superbuild was the repository root's `CMakeLists.txt`, so colcon treated the cloned repository as one
CMake package and built neither the node, nor the messages, nor the command line (AUD-002). It lives in `superbuild/`
now (`cmake -S superbuild -B build`), and `superbuild/` carries a `COLCON_IGNORE`. The packages that
colcon builds had their own gaps, all found by running the install guide in clean containers
(`scripts/check_install.sh`):

- `strataslam_tools` failed with `--symlink-install`.
- The `strataslam` command was never installed.
- `colcon test` ran no tool test.
- The simulator's tests were not built standalone.

`scripts/install_deps.sh` is the single list of system packages for the guide, the CI and the container checks. That
includes Boost, METIS, SuiteSparse and the placeholder that Ubuntu's `libgtsam-dev` needs (ADR-0007, AUD-003).

## ADR-0042 · Developer traces go through the Logger; `multifloor.enabled` is honoured

The core printed its debug traces with `fprintf(stderr)` when `STRATASLAM_DEBUG_*` was set (AUD-023). `debug_channel()`
reads each variable once and the traces go to the Logger at Debug level. In the node that is rclcpp's debug level
(`--ros-args --log-level strataslam_node:=debug`), in the simulator stderr. Every ROS callback, service and the
offline reader run inside a guard that logs an exception and drops the message instead of letting it reach the
executor.

`multifloor.enabled: false` was parsed but ignored. It now keeps a single level: the floor manager never runs, which
is what an outdoor drive with slopes needs (KITTI). Test: `Scenario.MultifloorDisabledKeepsOneFloor`. A tracking loss
that relocalization could not place still opened a new floor by the other path (the unobservable change): it is now
a session of the only level, as with height sensors (`Scenario.MultifloorDisabledKeepsOneFloorThroughAnUnrelocalizedLoss`;
the Tiago corridor with its camera alone: 11 unplaced sessions kept on the level, one floor, where it opened six).

## ADR-0043 · The external odometry is the motion prior of the camera tracking

`external_odometry` was parsed, and `auto` chose it as the odometry whenever it was enabled, but `add_odom()` was
empty and no frontend used it: with it enabled, nothing tracked the motion. It is now what a wheeled robot needs
from it: the prior of the camera tracking. The node reads it from TF (`frame: odom`, the transform `odom ->
base_frame` that the wheel odometry of most ROS robots publishes) or from a `nav_msgs/Odometry` topic; the core
keeps its last seconds and interpolates it at each camera frame. The visual frontend predicts the frame with the
odometry's increment instead of a constant velocity, and when the frame cannot be tracked it follows the increment
(a degraded pose, which the backend weighs less) and restarts its local map there: the loss never reaches the
pipeline, so it opens no new session and no floor. A camera frame waits for the odometry to cover its stamp (wheel
odometry trails the camera by a fraction of a second; at most 0.5 s), in arrival order, so a bag gives the same
result run after run. It is not an odometry source: `odometry_from: external` is an error, `auto` never picks it,
and with LiDAR odometry it is ignored with a warning (the LiDAR does not lose the motion).

The odometry also vetoes the camera: a tracked frame whose motion differs from the odometry's by more than 2 cm
plus 10 % of the step, or by more than 1 degree, is a wrong PnP solution and the odometry carries it (the local map
stays: only that solution was wrong). The bounds are about what wheel odometry can err in one frame plus the PnP
noise, not fitted to a bag.

Measured. Scenario S14 (S12's one-second camera blackout, with the simulator's wheel odometry): one session instead
of a second one reconnected by a loop, ATE 0.016 m (S12: 0.022 m). PAL Tiago corridor, camera only (wheel odometry
in TF at 15 Hz, 0.45 m off the LiDAR trajectory after 100 m), against the LiDAR map:

| | losses | floors | xy RMSE | largest z error |
| --- | --- | --- | --- | --- |
| camera | 24 | 6 | (one per floor) | 13 m (false ramps) |
| + wheel odometry as prior | 0 | 3 | 3.4 m | 10.8 m |
| + the veto and the IMU's gravity (ADR-0044) | 0 | 1 | 1.1 m | 0.43 m |

Without the veto the camera's roll and pitch went to -32 and -36 degrees in the corridor (the LiDAR measures under
2), which turned the forward motion into a climb; the IMU bounds them (ADR-0044) and the veto removes the wrong
solutions that remain.

## ADR-0044 · The IMU is used whatever its mounting; on a ground robot, for gravity

The PAL Tiago's base IMU is mounted upside down and the TF does not say so: at rest it reads -10.2 m/s^2 on its z, and
its gyroscope turns the opposite way to the wheels (correlation -0.97). Every use of the IMU took the TF and the ROS
sign convention for granted, so enabling it would have taken down for up. The mounting is now learned from the data
(`ImuMount`): the gyroscope is paired with the rotation the odometry measures, summed over windows of about a second
(where the turn stands out of the per-frame noise and a small time offset does not matter), and Wahba's problem gives
the rotation IMU -> base, the TF filling in only what the motion does not show (a ground robot only turns about the
vertical). With that rotation, gravity gives the accelerometer's sign convention and its scale. At start, on a ground
robot, gravity through the TF must point up within 30 degrees; if it does not, the IMU waits until the mounting is
learned (the robot must turn, 180 degrees in all). With a right TF the learned mounting only confirms it. On the
Tiago: learned from 182 degrees of turning, 179.6 degrees from the TF, gravity scaled by 0.965.

Gravity (`GravityTracker`, a Mahony filter on the up direction) is carried by the gyroscope and corrected by the
accelerometer with a 10 s time constant, so that the robot's own horizontal accelerations (which barely change the
magnitude of the specific force) do not tilt it; its integral term estimates the gyroscope bias, which a long time
constant would otherwise turn into a fixed tilt (the simulator's 0.002 rad/s gave 1.1 degrees). If the body was at
rest during the first second the bias is the gyroscope's mean there; if not, gravity is used only after three time
constants. The camera tracking's roll and pitch are bounded by it, not replaced: within 2 degrees (about the error of
an IMU on a moving body: 1.0 degrees mean in S06, walking) they are the tracking's own, beyond they are brought back
to the bound. Imposing gravity on every frame degraded good tracking (S09: ATE 0.011 -> 0.134 m; S06 broke); bounded,
the scenarios are unchanged. Frames that the camera did not measure (lost, or vetoed by the wheel odometry, ADR-0043)
take the IMU's attitude.

The vertical acceleration (elevators, ADR-0032) is a separate use, `imu.floor_changes` (default true): on a ground
robot's vibrating base it still produced false elevators, so `tiago_camera_only.yaml` turns it off and uses the IMU
for gravity only. Tests: `ImuMount.*` (a right TF, a driver with gravity pointing down, an upside-down mounting
missing from the TF, no turning) and `GravityTracker.FollowsATiltAndIgnoresTheRobotsOwnAcceleration`.

## ADR-0045 · Three level modes: floors, single, planar

`multifloor.enabled: false` kept one level but in 3D: a run that went down a ramp kept its heights, which is what a
level with slopes needs and not what a flat floor does, where every bit of height is drift. `multifloor.mode` now
says it: `floors` (each level its own map, the default), `single` (one map in 3D, the former `enabled: false`, which
still parses as it) and `planar`. In `planar` every frontend pose is moved onto the plane of the floor before the
pipeline sees it (x, y and heading kept, the height where the run started, roll and pitch from the IMU's gravity or
level), the camera tracking's own tilt is bounded by that up direction (ADR-0044), and every keyframe gets a height
prior (1 cm) and an attitude prior (0.01 rad) in its floor graph, so that loop closures, which are measured in six
degrees of freedom, cannot lift the map off the plane. It is the user's statement that the floor is flat, like
`floor_height_prior`: a real ramp is flattened and what is seen from it lands at the wrong height, so it is never the
default and it needs `robot.platform: ground`.

Measured on the PAL Tiago corridor (101 m, against the LiDAR map in 3D, whose own height wanders by 0.18 m): the
camera with wheel odometry and IMU went from 1.10 to 0.75 m in xy and from 0.43 to 0.12 m in z; the LiDAR profile
keeps its xy trajectory to the centimetre and becomes flat. `prueba_03` in the default mode is unchanged (keyframes
identical). Tests: `Rules.LevelModes`, `Scenario.PlanarKeepsTheBaseOnOnePlane` (S02 on a ground robot: every
keyframe within 3 cm of the plane and 1 degree of level, the scenario's criteria met).


## ADR-0046 · Scenario ATE: per floor, with a coverage requirement; criteria read values by name

The scenarios' ATE (`measure` in `strataslam_sim/src/scenarios.cpp`) aligns each floor of the map on its own (SE(3)
Umeyama) and pools the residuals over keyframes. Before fusion every floor has its own frame (`floor_<k>`, section 7
of the spec): there is no common frame to align them in, so the per-floor alignment is what the online map can be
measured in. Where the floors are placed relative to each other is measured separately: the floor height error, and
the 4-DoF fusion error of S07/S11.

The 2026-11-04 audit (AUD-005) showed what that hides: the ATE only sees keyframes that ended in the map. A run that
opened a transition and never closed it left 307 of its 328 keyframes out of the map (S02/S10, seed 7017) and reported
the best ATE of the 20 seeds. Every criterion that checks an ATE now also requires `map_coverage >= 0.90`: the fraction
of the keyframes taken on a floor (not mid-climb, which have no floor, ADR-0025) that ended in the map. Across S01-S14
with the default seed it is 1.000; in that run it is 0.064, and the scenario fails.

The same audit (AUD-004) found that S10's "no worse than the best single modality" read the ATE of the isolated runs
as `report.values.at(1)`, which is `floor_accuracy_all` (always about 1), so the criterion could not fail and the
published `ate_camera_only_m`/`ate_lidar_only_m` were not ATEs. Values are now looked up by name
(`report_value`, which throws on a missing name).

## ADR-0047 · Image files are read without OpenCV's imgcodecs

On Ubuntu, `libopencv_imgcodecs` loads GDAL, and with it poppler, jbig and rttopo (GPL): every process that links it
carries GPL code (audit AUD-002: `strataslam_tum_run` and `strataslam_vocab_train`, 146 libraries, 113 of them not
registered). The core never needed it; only those two tools read image files. They now use `strataslam::image_io`, a
small static library next to the core (`io/image_io.hpp`): PNG through libpng (8 and 16 bit; colour converted to
gray by libpng with the weights 0.299/0.587, as OpenCV's `IMREAD_GRAYSCALE` asked it to), JPEG through libjpeg, and
binary 8-bit PGM. No package asks OpenCV for imgcodecs any more, and `strataslam_ros` no longer depends on OpenCV.

Measured: the two tools load 36 libraries, none of GDAL/poppler/jbig/rttopo; the 8931 images of TUM fr1/xyz and
fr2/xyz decode byte for byte as `cv::imread` decoded them, and the fr1 trajectory is identical to the previous
binary's. libpng (libpng-2.0) and libjpeg-turbo (IJG, BSD-3, Zlib) are permissive but not on the specification's
allowed list; ADR-0051 accepts them (stb_image, MIT, was the alternative within the list). Tests: `ImageIo.*`.

## ADR-0048 · The LiDAR registration gives the same bits with any number of threads

Spec 12.9 asks for bit-identical runs with one thread and differences below 1e-6 m with several. The audit
(AUD-014) measured 5.3 mm between two 6-thread runs of S11 and 7.9 um on the Tiago bag; `threads: 0`, the default,
uses half the machine's cores, so results also changed from one computer to another. The only parallel code is
small_gicp's, and two of its OpenMP helpers hand out work dynamically (`schedule(guided)`): the reduction of the
GICP normal equations sums each thread's share in the order threads took them, and the parallel voxel grid writes
its output in the order blocks finish.

`lidar_frontend.cpp` now builds the same GICP from small_gicp's parts with two changes: the downsampling is the
sequential voxel grid (small_gicp's one-thread path; it is a small part of the time), and the reduction splits the
factors into 16 fixed contiguous chunks summed in chunk order, the threads only choosing who computes a chunk. Normal
estimation stays parallel: each point's is independent of the others. The grouping of every floating-point sum no
longer depends on the threads, so any thread count gives the same bits. Measured: S11 (1000 keyframes, three
floors) with 1, 3 and 6 threads, and twice with 6, writes 2019 identical files. Test:
`Scenario.MultiThreadIsIdenticalToSingleThread` (S02 with 1, 4 and 7 threads, equality, not a tolerance).

## ADR-0049 · Anchors enter every formulation as points

`lie4` received each anchor set as point correspondences (`AnchorFactor`), while `se3soft` and `euler` received one
pose pre-aligned from the set with a heuristic yaw covariance. The 2026-11-04 audit (AUD-016) showed with a mutant
that gave `lie4` that same pose that it then reproduced `euler` seed by seed: the published advantage of `lie4` came
from how the anchors entered, not from the formulation. `se3soft` and `euler` now have the same point residual in their
own parametrization (`ablation::Anchor3Factor`, `ablation::EulerAnchorFactor`; tests `AblationFactors.*`), and the
ablation table is regenerated with it.

Considered and rejected: making anchor sets known inliers of GNC. In S07 seed 7012 GNC drops the building-corner set
when it disagrees with the elevator-shaft set (each floor map is distorted by a few decimetres), and keeping both
improved that seed (0.367 m / 2.54 deg -> 0.255 m / 0.39 deg; S07 17 of 20 instead of 16). But it removes the
protection against a set that is wrong as a whole, which `Fusion.GncRejectsAFalseFiveMeterConstraintAmongThreeCorrectOnes`
requires, and it made S11 seed 7002 fail (0.103 m). S07's remaining failures are translation errors of camera maps
that are themselves distorted; they are open (AUD-008).

## ADR-0050 · Deviations from the development specification, in one place

The 2026-11-04 audit (A13-05, AUD-036) asked for every deviation from the development specification to be stated
with its reason. The specification itself is not part of this repository; this list is the reference.

| Deviation | Why | Where |
| --- | --- | --- |
| The project is called StrataSLAM: packages `strataslam_*`, namespace `strataslam`, topics `/strataslam/...`, command `strataslam`, variables `STRATASLAM_*` (the specification says `easyslam`) | Renamed on 2026-11-04 (name clash). Maps written before still load: the manifest's `easyslam_version` is read as `strataslam_version` | `MapIo.MapsSavedBeforeTheRenameStillLoad` |
| `fusion.odometry_from: external` is an error; the external odometry is the motion prior of the camera tracking | A wheel odometry alone drifts in heading and cannot close loops; as a prior it carries the camera through losses and vetoes wrong tracking | ADR-0043 |
| Two scenarios beyond the twelve of section 12.4: S13 (a gentle ramp, 2.9 deg, slower than the calm speed; criteria: a `Ramp` transition, ATE <= 0.15 m, two floors, floor height error <= 0.25 m) and S14 (S12 with wheel odometry: ATE <= 0.15 m, one floor, one session) | S13 found the gentle-ramp defect of ADR-0021; S14 is the external odometry of ADR-0043. The 0.25 m of S13 is S05's (an elevator measured by the barometer): a 3 m climb over 60 m of ramp is measured as much by the odometry's slow vertical drift as an elevator is by the barometer | ADR-0021, ADR-0043 |
| Anchor sets reach `se3soft` and `euler` as point correspondences, with factors of their own | So that the ablation compares formulations and not how the anchors enter | ADR-0049 |
| The ablation runs in the nightly CI, not in the per-commit CI | S07 and S11 over 20 seeds and three formulations take about an hour | nightly workflow |
| The scenarios' ATE aligns each floor on its own and requires a map coverage of 90 % | Before fusion every floor has its own frame | ADR-0046 |
| No camera presets: the profile is the whole configuration (the specification merges a `preset` into the profile) | A saved map keeps the profile text, which did not hold what the preset filled in; one merge in C++ and another in Python had to agree | ADR-0053 |
| Image files are read with libpng and libjpeg, not OpenCV's imgcodecs | imgcodecs loads GPL code on Ubuntu; libpng and libjpeg are permissive but outside the list of section 3.2 | ADR-0047, ADR-0051 |
| The licence rule of section 3.2 holds for what StrataSLAM builds, includes or redistributes; weak copyleft (LGPL, MPL-2.0) is accepted in system libraries that a dependency's package loads, and libpng-2.0, IJG and Zlib count as permissive | METIS and Spectra arrive inside the GTSAM of ROS 2 Jazzy and Ubuntu, unused; a GTSAM of our own would remove METIS only | ADR-0051, `scripts/license_audit.py` |

## ADR-0051 · Licences: weak copyleft in system libraries, permissive equivalents, Apache-2.0

The 2026-11-04 audit (AUD-001, AUD-002, A2-04) found four libraries that the built artifacts load outside the list of
section 3.2 of the specification (MIT, BSD, Apache-2.0, MPL-2.0 for Eigen only; GPL, LGPL and AGPL forbidden).

- **METIS** (Apache-2.0, with two GKlib files from glibc under LGPL-2.1-or-later) is loaded by
  `libstrataslam_core.so` because libgtsam lists it as needed, on both install paths. StrataSLAM never calls it:
  iSAM2 orders with COLAMD.
- **Spectra** (MPL-2.0) is compiled inside libgtsam (Shonan averaging); StrataSLAM never calls it.
- **libpng** (libpng-2.0) and **libjpeg-turbo** (IJG, BSD-3-Clause, Zlib) read image files since ADR-0047.

**Decision.** The rule of section 3.2 holds for everything StrataSLAM builds, includes, links statically or
redistributes: that code stays MIT, BSD, Apache-2.0 or MPL-2.0 (Eigen). Beyond it:

1. Weak copyleft (LGPL-2.1, LGPL-3.0, MPL-2.0) is accepted only in a library that a dependency's system package loads
   (`transitive`) or compiles in (`embedded`), linked dynamically and never redistributed by this project. The LGPL
   asks that a user can replace the library, which a shared system library already allows; MPL-2.0 binds only the
   files of Spectra.
2. Strong copyleft (GPL, AGPL) and non-commercial terms stay forbidden however a library arrives: that is why
   imgcodecs, which loads GDAL and poppler, was replaced (ADR-0047).
3. libpng-2.0, IJG and Zlib are permissive and compatible with Apache-2.0; they are accepted for any dependency.
4. The project licence is Apache-2.0 (no longer provisional).

**Alternatives rejected.** A GTSAM built with `GTSAM_SUPPORT_NESTED_DISSECTION=OFF` removes METIS but not Spectra,
adds a long build to every install, and conflicts with the `ros-jazzy-gtsam` that other ROS 2 packages use.
stb_image (MIT) instead of libpng/libjpeg: its author warns that it is not hardened against malformed files, and the
two system libraries are maintained.

**Consequence.** Whoever distributes binaries that include these libraries (a Docker image, a static build) must
meet the LGPL for METIS: offer its source, which Ubuntu's source package does. `scripts/license_audit.py` applies the
rule (`WEAK_COPYLEFT` only for `transitive` and `embedded` rows; `FORBIDDEN_PATTERNS` for every row), with tests in
`strataslam_tools/tests/test_license_audit.py`.

## ADR-0052 · TF: map above the floors, and `output.publish_tf`

The node published `floor_<k> -> odom` and nothing above it: `robot.map_frame` was only the header of `floor_state`, so
RViz with the fixed frame `map` could not place the floors. And with a localizer that owns `map -> odom` (AMCL in the
Summit simulation with EasyNav), `odom` had two parents and TF kept the localizer's: the floors hung loose.

**Decision.** As other SLAM stacks do (slam_toolbox, Cartographer, RTAB-Map publish `map -> odom`), the node owns the
frames above the robot's odometry, REP-105: `map -> floor_<k> -> odom -> base_frame`. `map` is the first floor; every
other floor hangs from it where the odometry entered it (height included), which `strataslam fuse` refines offline.
`output.publish_tf: false` (RTAB-Map's `publish_tf`, Cartographer's `publish_to_tf`) leaves `map` and `odom` to another
node: the node then publishes only its own branch `floor_<k> -> strataslam_odom`. The node warns, once, when `odom` has a
parent that is not one of its floors, and says which option to set; it does not change what it publishes on its own
(none of those stacks anchors itself to another node's `map`: two maps that need not agree would be overlaid).
With `publish_tf: false` the branch is left apart from `map` on purpose: the only place to join it without a second
parent is under `odom`, and a world frame (`floor_<k>`) hanging from a drifting one is against REP-105; RViz shows it
from the fixed frame `floor_0`.

**Tests.** `strataslam_ros/test/test_tf_tree.py`: `map -> floor_0` in both trees, the warning with a localizer publishing
`map -> odom`, and with `publish_tf: false` nothing on `map`, `odom` or the robot's frames.

## ADR-0053 · The profile is the whole configuration: no camera presets

Camera presets (`camera: {preset: realsense_d435i}`, files in `strataslam_ros/presets/`, ADR-0001) filled in a camera's
topics, depth range and IMU noise, merged under the profile's own values.

**Why they go.** (1) A saved map keeps the profile's text (`profile.yaml`), and that text did not hold what the preset
filled in: a map did not say with which topics, range and IMU noise it was made, and a later change of the preset
changed what an old profile meant. (2) The merge, with its rules for the IMU (an `imu:` section present took the
preset's IMU unless `noise_preset` said otherwise), was written twice, in the C++ loader and in the doctor's Python,
and both had to agree. (3) The launch, the doctor, the bag exporter and the tests each found the preset folder. (4)
Since the generic profiles became templates with every option (2026-11-05), what a preset said can be read in the
profile itself.

**Decision.** The profile is the whole configuration. The tested cameras' values are written in the example profiles
and listed with their drivers in `docs/user/profiles.md`. `preset:` and `noise_preset:` are removed keys: a profile
that has them is refused with what to write instead (not "unknown key"), so that no old profile runs silently with other
values. Without `imu.noise` the nominal defaults are used. `LoadOptions` and the `preset_dir` parameter of the node and
of `validate_profile` are gone; `strataslam imu-noise` prints the `sensors.imu.noise` block.

**Cost.** The D435i's values are written in every template that uses it (four since ADR-0056). Tests:
`Rules.PresetKeysOfOldProfilesSayWhatToWriteInstead`, `Rules.TheImuNoiseIsTheProfilesOrTheNominalDefaults`,
`Rules.ACameraNeedsItsThreeTopics`, and the template tests (`Profiles.*`).

## ADR-0054 · A visual vocabulary ships with the package (`fusion.vocabulary: default`)

Camera loop closures need a vocabulary, and none was shipped: a camera profile run as documented closed no loop (the
Tiago corridor, 2026-11-05), with one warning at start-up.

**Decision.** `strataslam_ros/vocabularies/tum_rgbd.bin` (9964 words, 527 kB) is trained with `strataslam vocab train`
on every fifth colour image of TUM RGB-D fr1/xyz and fr2/xyz (892 images of indoor offices). `fusion.vocabulary:
default` (or `vocabulary:=default`) names it; the example profiles with a camera use it. It is derived data of images
under CC BY 4.0, which allows redistribution with attribution: the attribution and the changes are in
`strataslam_ros/vocabularies/README.md` and `THIRD_PARTY.md`. `scripts/train_default_vocabulary.sh` rebuilds it byte for
byte: `strataslam_vocab_train` now reads the images in name order (the order of a directory depends on the file
system, and the k-means, with its fixed seed, starts from the descriptors in the order they come).

**Limits.** Offices are close to most indoor robots' rooms and corridors, not to warehouses or outdoors: a vocabulary of
one's own environment is documented as the better choice there. A forward camera still closes no loop on a route that
only comes back the other way (troubleshooting).

**Tests.** `Rules.TheVocabularyComesFromTheProfileRelativeToItsFolder` (`default` is a name, not a path);
`test_node_with_profiles.py` loads it from a profile (`vocabularies/tum_rgbd.bin (9964 words, from the profile)`).

## ADR-0055 · The C++ code is formatted by `.clang-format`, and CI checks it

`.clang-format` (Google style, 100 columns, the line length of the Python code too) existed but had never been applied
(audit AUD-097). It is applied to every C++ file in one commit of its own, listed in `.git-blame-ignore-revs` so that
`git blame` skips it, and the CI job `format-cpp` fails on any file it would change. The version is pinned
(clang-format 18, Ubuntu 24.04's): another version can format the same file differently. The commit changes no
behaviour: the build has no new warning, the unit and ROS tests pass, and S01, S07 and S11 give the same values bit
for bit before and after (seed 42, one thread).

## ADR-0056 · One template per kind of robot

The generic profiles were named after a camera or a mix (`handheld_rgbd`, `lidar_only`, `combined_aerial`), and
`lidar_only` also had an IMU and a barometer, so none started a ground robot with a camera, or with its LiDAR alone. The
templates are now one per kind of robot, named `<platform>_<sensors>`: `ground_camera` (with the wheel odometry as the
motion prior, TF `odom -> base_link`), `ground_lidar`, `ground_lidar_camera_imu` (LiDAR odometry, camera and LiDAR loop
closures, IMU for gravity only: ADR-0032), `handheld_camera_imu` and `aerial_lidar_camera_imu`; `handheld_oak_d_lite`
stayed as the example of another camera (removed later, like the robot profiles: its values are in
`docs/user/profiles.md`). Nothing was released with the old names, so they are not kept as aliases.
Every template holds every option (`Profiles.TheTemplatesHoldEveryOption`) and stays valid when any commented option
is uncommented; `external_odometry.topic` and `.frame` are alternatives, so uncommenting one comments out the other.
Each template runs the node on a simulator bag (`test_node_with_profiles.py`); the bag exporter writes the wheel
odometry when the profile enables it (TF, or `nav_msgs/Odometry` on its topic).

## ADR-0057 · LiDAR loops: the residual is judged against the scan's own

`loop_max_mean_error` (0.12) is the GICP residual per inlier, whose scale depends on the sensor, the voxel and the
scene. Outdoors it rejected nearly every right loop (ADR-0022): on KITTI right revisits registered with fitness 0.99
and residuals of 0.10-0.44. A right registration of the same scan sets the scale: the current keyframe against the
submap of its predecessors (as many keyframes as the loop submap), from their graph poses.

- A loop is accepted when the residual is under `loop_max_mean_error`, as before, or at most `loop_max_error_ratio`
  (4) times that reference. The reference is computed only for a candidate that passes fitness and degeneracy, once
  per keyframe.
- No indoor/outdoor setting in the profile: the residual depends on the sensor and the scene, not on a label, and the
  reference measures it on each scan.
- **Evidence.** On the KITTI bags with ground truth the right loops have a ratio of 0.4-2.9 (a few up to 8.6, still
  rejected) and every wrong candidate (repeated straight streets) 6.8 or more, and degenerate too. ATE improves where
  loops were missing (KITTI 07: 0.81 -> 0.41 m) and is unchanged elsewhere. Indoors (Summit, Tiago) 3-17 times more
  loops are accepted, with the same floors, each within 0.2 m of the distance the map already gave its keyframes; the
  simulated scenarios are unchanged.
- **Limits.** Without ground truth single loops cannot be labelled; a place that repeats with the same geometry (a car
  park of identical bays) can still give a low ratio, so fitness and the degeneracy check stay in front.

## ADR-0058 · The IMU's heading on a ground robot comes from its accelerations; its gravity bounds the LiDAR

A LiDAR odometry's pitch drifts, and on a long drive a degree of pitch is a climb of a metre per 60 m: on a ground
robot that is a false floor over the same ground. The IMU's gravity can bound it, but only the visual frontend used
it. On a ground robot there is a catch: its turns are all about the vertical, so learning the IMU's mounting from them
leaves its heading about the vertical free, and a TF with the IMU turned (a driver reading x and y backwards, for
example) reverses the sign of every slope the IMU reports.

- The LiDAR frontend keeps the base's roll and pitch within `max_tilt_error_deg` (1 deg) of the IMU's gravity
  (`GravityTracker`), as the visual frontend does with its own bound (2 deg); within it they are the registration's.
- `ImuMount` also pairs, over the same 1 s windows, the base's acceleration by the odometry (dv/dt + w x v in the base
  frame, the velocity at the window's edges a mean of the last 5 frames: differentiated per frame, the registration's
  jitter is ten times the signal) with the accelerometer's mean specific force minus gravity. The turn about the
  base's z that best maps one onto the other is the IMU's heading. It is decided once the horizontal acceleration adds
  up to 4 (m/s^2)^2 and the accelerometer explains half of it. An error e of the heading tilts a slope only by
  tilt * sin(e).
- Until the heading is seen, the IMU of a ground robot does not bound the LiDAR (its tilts have no known direction);
  the visual frontend keeps using it as before. The mounting check goes on until then (`imu_learned`), and a mounting
  learned from the turns alone is applied at once, as before, then refined.
- **Evidence.** On the Summit XL bags (an IMU whose TF has it turned half around) the false floor of `prueba_03`
  disappears (3 floors instead of 4) and the ramps of both bags come within 0.1-0.4 m of the climb given by the wheel
  speed times the IMU's pitch; once decided, the heading stays within about 10 deg of its final value. A 2 deg bound
  left a spurious step; 1 and 0.5 deg do not.
- **Limits.** A ramp driven before the heading is learned is the LiDAR's alone. A robot that only turns in place never
  shows its heading, and its IMU then stays out of the LiDAR.

## ADR-0059 · Elevator rides the IMU measures while the odometry is lost, or still in the cabin

An elevator ride is the IMU's to measure when the odometry does not see it. The odometry, though, often opens a
transition of its own just before the ride (its drift while walking, a jump as the doors close) and is then lost in
the cabin, or sees the cabin still. The IMU never took such a transition over, and the ride was lost, or closed as a
short ramp; the floor reached was mapped into the one left.

- **Takeover.** An odometry-opened transition is taken by the IMU when its 1 s mean of vertical acceleration passes
  the departure threshold for `imu_depart_s`, it has gained a cabin's speed (0.3 m/s) and the odometry did not follow:
  lost all along, or tracking at less than half that speed (a cabin is still to it), with the IMU smooth as a cabin
  is (`kCabinSpread`: under 0.45 m/s^2 of spread over 1 s; cabins measure 0.1-0.4, stairs and steps, which the
  odometry tracks half-way and whose 1 s means look like a push, 0.5-4.4). What the odometry's transition held until
  the onset ends on the floor it left (`ride_follows`), with any session lost before it; the ride counts from its onset.
- **The ride.** Push and brake are the integrals of the 1 s mean beyond 0.2 m/s^2 (a cruising cabin vibrates by about
  +-0.16), counted from the onset, from the mean of the 5 s before it (the slow bias estimate drifts while walking).
  It ends braked (60 % of the push) and calm for `ride_stop_s`, or when the odometry walks out once the cabin has
  slowed to half its top speed and its last pulse is over (not out of a cabin still moving or braking). A ride that
  never comes back to rest was one pulse: a jolt, or the brake of a ride whose push stayed under the threshold. It is
  dropped once the IMU jolts as steps do for 3 s while the odometry walks (a cabin does not, even when the odometry
  drifts in it), or after 30 m; first, the push is looked for in 60 s of history (an opposite pulse of most of its
  speed before a cabin's calm) and the ride measured from it. A ride whose height is no floor's goes back to the
  odometry. The level it reaches is known to 5 % of the ride (per-level sigma), so a stair back down closes on the
  level it left.
- **The map.** The frontend restarted after the ride is raised by its height, which the manager is told is no motion
  (`odometry_rebased`): the odometry's z is the height again. Left at the height before the ride, a floor reached
  again by the stairs was linked to its keyframes a ride's height off and every loop with them was rejected (scenario
  S15, three floors up by elevator and down the stairs: ATE 3.45 m and no loop, now 0.05 m and 22). The keyframes
  taken in the cabin of such a ride are dropped, as when the odometry is lost there: they have the height it left. A
  restart is turned by what the gyroscope measured while the odometry was blind.
- **The IMU itself.** A mounting learned from the motion turns the gravity tracker's state instead of resetting it
  (reset, a moving body waited tens of seconds for gravity and missed a ride); a heading under 20 deg from the TF keeps
  the TF (the learned heading is that noisy on an IMU that is the base frame).
- **Evidence** (offline, Elevator-VIGS: 16 real sequences with measured rises or round trips and 16 simulated ones
  with exact ground truth, CC BY-NC 4.0, not redistributed; against `aad8e44`). Every real sequence comes closer to
  its truth: one-way rises within 0.06-0.9 m, round trips closing within 0.07-2.6 m (3.2-13.9 m before). The simulated
  rides are measured (per-floor ATE 0.07-0.39 m, up to 23 m before). KITTI and the OAK bags are unchanged; of the four
  Summit ramps, which move by up to 0.4 m with the tracker turn, two come closer to the wheels-and-IMU estimate, one
  moves away and one is unchanged.
- **Limits.** On the simulator's walks without a ride its IMU's gravity errs by 2-5 deg (0.5-0.9 on real IMUs) and the
  LiDAR's 1 deg bound (ADR-0058) follows it: a simulated walk keeps a false floor of about 2.5 m. Loosening the bound
  or the tracker's gate fixed those walks and lost real rides.

## ADR-0060 · In a confined space the rotation is the gyroscope's

In a space a metre or two across, such as an elevator cabin, a LiDAR registration has little geometry, often reflective
walls and moving doors: as the person turns round in it, or walks in and out, its rotation can go wrong by tens of
degrees within a few scans. After an elevator ride the floor reached was then mapped turned against the one left, live
and in the saved map, with no loop closure to correct it. The gyroscope measures those turns to a fraction of a degree
over a minute. Registering the floors on each other (`floor_stacking`, ADR-0027) does not help: the area explored
around a lift lobby is too small for an unambiguous match.

- **Confined.** A scan whose points lie within `confined_range` (3 m, 95th percentile of their range) sees a confined
  space; it is out of it once it sees twice as far (a half-open door still shows mostly the cabin). 3 m separates a
  cabin (its scans mostly under 2.5 m) from open spaces: corridors, streets and car parks never come under 5.9 m on
  the bags checked, lift lobbies down to about 4 m; a rig laid on the floor can (see Limits).
- **The rotation.** There, with a trusted IMU, the scan's rotation is the gyroscope's turn since the last pose, with
  its heading drift against the odometry over the minute before taken out, projected on the rotations (nothing else
  re-estimates it, and a product of matrices drifts off them); the registration solves the translation only
  (small_gicp's `RestrictDoFFactor`), on a scan sampled twice as finely (a cabin has too few cells of 0.4 m to be
  tracked). A frontend started some scans after its restart is turned by the gyroscope since. For 10 s out of a
  confined space, a registration turned away from the gyroscope by more than 10 deg and half the turn takes the
  gyroscope's rotation: the local map is still the cabin. Elsewhere the gyroscope is not used, and the result is the
  same bit for bit.
- **The gyroscope's record.** Its attitude is kept every 20 ms; a turn asked for past the last record goes on to the
  last sample's attitude. Without it up to 20 ms of turn was lost at every scan, and in a fast turn those add up.
- **The floors.** A confined scan tells the floor manager that its odometry does not observe the vertical
  (`FloorSample::vertical_observed`): in a moving cabin it tracks the cabin, or sees the shaft move through the gaps
  of the door. The altitude is held there and taken up again from where it was held. Lost during a ride once its scans
  are no longer confined (out of the cabin), the frontend restarts turned by the gyroscope instead of waiting for the
  ride's end: its local map is the cabin, it would stay lost, and the ride would not see the walk out.
- **Evidence** (offline, Elevator-VIGS, against `9fcde3d`). On the real sequences whose arrival floor was turned by
  11-162 deg against the one left, it is now within 0-13 deg (but one, whose ground truth itself drifts in the cabin),
  and the per-floor ATE is lower where it was worst (up to 17.8 -> 1.0 m). Floor changes: 3 sequences closer, 28 the
  same, 1 worse (see Limits). Bags without a confined space (KITTI, the Summit and Tiago bags, the OAK bags, the
  simulated walks) give identical maps; the scenarios with an elevator (S11, S15) change by less than a millimetre.
- **Limits.** A confined scan does not always mean an unobserved vertical: a rig laid on the floor gives confined
  scans while its lift is visible, and the IMU can take that lift for a short ride (one sequence, a false floor of
  1.5 m). Telling them apart by the agreement of the odometry's vertical motion with the IMU's fails on glass lifts,
  whose scans see out. The gyroscope's residual heading drift accumulates on long rides (2-3 deg a minute in
  simulation); estimating it more finely (a line over the window, the bias at rest) did not hold on real data. The
  3 m range is checked on a few platforms only; the exit guard (10 deg, 10 s) and the restart out of the cabin rest on
  few cases and are the least validated parameters.

## ADR-0061 · A degenerate LiDAR registration moves no further than a physical acceleration allows

Where the registration is degenerate (`Degraded`, ADR-0012), the scan's geometry barely constrains one direction, and
along it GICP can slide to a structure that repeats, such as the next flight of a stairwell between identical floors.
Going down such a stairwell in scenario S15 the weak direction is the vertical: in two scans the registration left the
constant-velocity prediction by +0.6 and +0.5 m upwards (about 60 m/s^2), the velocity model took that as a climb at
over 5 m/s and carried it on, and the scans locked onto the flight above. The descent was not measured, the floor
below took the height of the one above, the walk back was not recognised. Pinning the weak direction to the prediction
(`pin_degenerate`) was rejected in ADR-0012: the prediction itself is wrong through a turn.

- **Decision.** When `Degraded` and not pinned, the registration leaves the prediction along the weak direction by at
  most `max_acceleration` x (scan interval)^2: 30 m/s^2, 0.3 m at 10 Hz. A sensor carried by a robot or a person does
  not accelerate at 3 g; a turn (1-2 m/s^2) or a step is taken as registered, and after a gap in the scans the bound
  grows with it. Elsewhere nothing changes. `0` turns it off.
- **Evidence** (seeds 1001-1020, against `c7eaa80`). S15 passes on 20 of 20 seeds (16 before; its failures, like the 3
  of `3fe1a90`, share one signature, and the five looked into were that slide); S11 20 of 20 (19), its fused error
  unchanged on average (0.069 m) and lower at worst (0.101 -> 0.092 m); every other scenario identical. 10 and 20
  m/s^2 also fix S15, but leave a seed of S11 just over its 0.10 m threshold. KITTI 0027, the Summit bags prueba_00
  and prueba_03 and the Tiago bag: identical maps. Elevator-VIGS: 19 of 32 sequences change, through a few bounded
  scans each (0.3-0.7 m, mostly in cabins), and their effect downstream goes both ways: floor changes 1 closer
  (Residential3, 3.4 -> 1.7 m), 29 the same, 2 further (Station3 2.2 -> 2.6 m; simulated Mall-M, 0.25 -> 1.3 m, an
  extra floor on its last ramp); the arrival floor's heading better on 4 sequences (Station2 13 -> 1 deg) and worse on
  3 (Station3 5.5 -> 13.6 deg).
- **Limits.** The bound acts along the weakest direction only, and only when the registration is `Degraded`; a slide
  slower than the bound, scan after scan, is not stopped. A platform that does accelerate harder (a drone's flip, a
  dropped sensor) needs a larger value.
