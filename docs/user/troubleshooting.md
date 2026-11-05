# Troubleshooting

| Symptom | Likely cause |
| --- | --- |
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
| LiDAR timestamps | provide a per-point `t`/`time` field (s or ns) for deskew; without it the scan is used as is |

Diagnostics for developers: `STRATASLAM_DEBUG_<CHANNEL>=1` enables per-frame traces of a channel (`TRACK`, `FLOORS`,
`ALT` (=n: every n-th sample), `BACKEND`, `LOOPS`, `IMU`). They go to the Logger at Debug level: in the node run it with
`--ros-args --log-level strataslam_node:=debug`; the scenario tools print them to stderr (also with
`STRATASLAM_VERBOSE=1` for the info messages).
