# Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `strataslam: command not found`, or `ros2` does not know `strataslam_ros` | the workspace is not sourced in this terminal: `source ~/strataslam_ws/install/setup.bash` (each new terminal, or once in `~/.bashrc`) |
| `strataslam: cannot find 'strataslam_fuse'` | the command line was installed with pip but the C++ build is not found: source the colcon workspace, or build the core (`cmake -S superbuild -B build`) in the same checkout, or put the folder holding `strataslam_fuse` on `PATH` |
| `strataslam doctor needs ROS 2` | doctor reads topics and rosbags through rclpy and rosbag2_py: source ROS 2 and the workspace |
| `config error: ...` at launch | the profile is wrong; the message names the key |
| "waiting for TF base_link -> ..." | the sensor mounting transforms are not published; run the doctor |
| RViz: "No transform from [floor_0]" / "TF: odom also has the parent 'map'" | another node (a localizer such as AMCL) publishes `map -> odom`: set `output.publish_tf: false` and use the fixed frame `floor_0`, or stop the other localizer |
| Odometry stops, "tracking lost: new session" | featureless view (blank wall, door closing) or fast motion; the session reconnects by loop closure |
| Two floors merged / a floor missing | identical floors with no IMU or barometer: add one, or fuse with anchors |
| Map is flat | an elevator with no height sensor; check `floor_state` and the fusion report warnings |
| "no vocabulary: visual loop closure is disabled" | set `fusion.vocabulary: default` in the profile (the one shipped), or train your own with `strataslam vocab train` and give its path (or pass `vocabulary:=`) |
| Camera only: few or no loop closures with a vocabulary | the camera recognises a place only when it sees it again facing the same way; a route that comes back the other way gives it nothing (`save_map` reports the count). A 360-degree LiDAR does not have this limit |
| LiDAR drifts in a corridor/stairwell | geometric degeneracy; the frontend reports `Degraded` and the backend widens its covariance; add the camera or use more beams |
| `libCppUnitLite.a` missing at CMake time | Ubuntu's `libgtsam-dev` without the placeholder: run `scripts/install_deps.sh core`, or use `ros-jazzy-gtsam` (ADR-0007) |
| colcon builds only one package called `strataslam` | an old checkout with a `CMakeLists.txt` at the root; the superbuild is in `superbuild/` now (ADR-0041) |
| The node drops frames when a bag is played at 2x | the subscriptions keep 5 messages (sensor QoS); a run that cannot keep up loses frames. Process the bag offline (`bag:=`), which never drops one |
| Few or no LiDAR loop closures outdoors | Scan Context is tuned for buildings (rooms of a few metres, 50 m range); on the KITTI drive no loop was accepted (docs/results.md) |
| LiDAR timestamps | provide a per-point `t`/`time` field (s or ns) for deskew; without it the scan is used as is. A 2D scanner's LaserScan gives it with `time_increment` (0: no deskew) |
| 2D scanner: `"2d" requires robot.platform: ground` | a planar sweep measures the motion in its plane only (x, y, heading): hand-held and aerial platforms need a camera or a 3D LiDAR |
| 2D scanner drifts along a long corridor | a sweep of two parallel walls does not see the motion along them; the frontend reports `Degraded` and the acceleration bound holds the estimate. Add the camera, or keep something across the corridor in view |
| 2D scanner: the map is a thin slice | without a camera it is the plane of the sweep, at the scanner's height; enable the camera for the 3D (its depth placed by the scan, ADR-0068). For navigation export the occupancy grid (`strataslam export --format grid`) |
| LiDAR and camera: most of the map grey | the camera sees a part of what a 360-degree LiDAR sees; the rest is coloured from other images only where the LiDAR point lies on the surface the camera's depth measured (within 0.3 m). Few colours with many keyframes near each other mean the two disagree: check the camera-LiDAR extrinsic and that the depth is registered to the colour image (`strataslam doctor`) |
| The map has no colour | `fusion.colorize_cloud: true` and a camera whose colour image reaches the node (with `features_topic:=`, the simulator's keypoints, there is none). Points the camera never saw stay grey |
| Occupancy grid: obstacles missing or the floor marked | adjust the band above the robot's base with `--min-height` / `--max-height` (default 0.05-2.0 m): a 2D scanner's sweep must lie inside it |
| `export --format grid`: "clouds are in the sensor's frame" | the map was written before `cloud_frame` (ADR-0062); map the bag again with this version |

Diagnostics for developers: `STRATASLAM_DEBUG_<CHANNEL>=1` enables per-frame traces of a channel (`TRACK`, `FLOORS`,
`ALT` (=n: every n-th sample), `BACKEND`, `LOOPS`, `IMU`). They go to the Logger at Debug level: launch the node with
`log_level:=debug` (for example `STRATASLAM_DEBUG_TRACK=1 ros2 launch strataslam_ros strataslam.launch.py profile:=... log_level:=debug`); the scenario tools print them to stderr (also with
`STRATASLAM_VERBOSE=1` for the info messages).
