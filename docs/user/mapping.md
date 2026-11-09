# Mapping

## A first map, start to finish (simulated building)

Everything below runs on a fresh install ([install](install.md)) without any sensor: the simulator writes a rosbag
of a two-storey building with identical floors and a staircase (scenario S03), as a hand-held RealSense would record it.

```bash
source ~/strataslam_ws/install/setup.bash
P=$(ros2 pkg prefix strataslam_ros)/share/strataslam_ros/profiles/handheld_camera_imu.yaml
# 1. A rosbag of the simulated building, its ground truth (trajectory.tum) and a visual vocabulary for it.
ros2 run strataslam_ros strataslam_sim_export_bag S03 ~/strataslam_demo --profile $P
# 2. Check the data against the profile: every row should be ✔ (exit code 0).
strataslam doctor $P --bag ~/strataslam_demo/bag
# 3. Map. The simulator publishes ORB keypoints instead of images, hence features_topic.
ros2 launch strataslam_ros strataslam.launch.py profile:=$P vocabulary:=$HOME/strataslam_demo/vocab.bin \
    features_topic:=/strataslam/features &
ros2 bag play ~/strataslam_demo/bag
ros2 service call /strataslam_node/save_map std_srvs/srv/Trigger   # prints the map folder
# 4. Clouds (PCD), the fusion of the two floors into one building, and the error against the ground truth.
MAP=$(ls -dt ~/strataslam_maps/*/ | head -1)   # the most recent map (the path save_map printed)
strataslam export $MAP      # -> $MAP/export/global_cloud.pcd (the building as saved), floor_000.pcd, floor_001.pcd
strataslam fuse $MAP        # -> $MAP/fused/global_cloud.pcd, floor_transforms.yaml, fusion_report.yaml
strataslam eval $MAP --gt ~/strataslam_demo   # ATE against trajectory.tum
```

With a real camera, step 1 is your robot or `ros2 bag record`, and step 3 drops `features_topic` (the node extracts ORB
from the colour images). The example profiles with a camera use `fusion.vocabulary: default`, the vocabulary shipped
with the package (trained on TUM RGB-D, indoor offices). Without any vocabulary the node starts and warns "no
vocabulary: visual loop closure is disabled", and says it again when it saves the map: maps still build, but drift is
not corrected.

## Reference

```bash
strataslam doctor profile.yaml
strataslam map profile.yaml --vocabulary vocab.bin        # = ros2 launch strataslam_ros strataslam.launch.py ...
ros2 service call /strataslam_node/save_map std_srvs/srv/Trigger
```

The launch file validates the profile with the core's own code first; an invalid profile prints its errors
and nothing starts.

- **Vocabulary.** Visual loop closure needs a vocabulary. `fusion.vocabulary: default` is the one shipped
  (`strataslam_ros/vocabularies/tum_rgbd.bin`: indoor offices, derived from TUM RGB-D under CC BY 4.0, attribution in
  its README). For other environments one trained on your own images can do better:
  `strataslam vocab train <image_folder> --out vocab.bin`, then `fusion.vocabulary: vocab.bin` in the profile
  (relative to it) or `vocabulary:=` at launch. Without one the system still works but visual loop closure is off;
  `save_map` reports how many loops closed and repeats the warning.
- **Topics.** `/strataslam/odometry`, `/strataslam/path`, `/strataslam/floor_state` (floor, state, transition
  type), `/strataslam/floor_cloud` (current floor).
- **TF.** The node owns the frames above the robot's odometry, as a SLAM does in REP-105:
  `map -> floor_<k> -> odom -> base_frame`. `map` (`robot.map_frame`) is the first floor; every other floor hangs from
  it where the odometry entered it, its height included (`strataslam fuse` refines that offline). If nothing else
  publishes `odom -> base_frame`, the node does, from its odometry; if the robot does (wheel odometry, as the Summit and
  the Tiago), the node publishes `floor_<k> -> odom` as the correction on top of it, and its own odometry
  (`/strataslam/odometry`) is in a sibling frame `strataslam_odom`. The start-up log says which.
  **With another localizer** that owns `map -> odom` (AMCL, a navigation stack's localizer), `odom` would have two
  parents: the node warns, and `output.publish_tf: false` in the profile makes it publish only its own branch
  `floor_<k> -> strataslam_odom` (as `publish_tf` in RTAB-Map or `publish_to_tf` in Cartographer). That branch is not
  joined to `map` on purpose: its only joint would be under `odom`, and a world frame such as `floor_<k>` hanging from
  a drifting frame is against REP-105 (it would slide in `map` with the wheels' drift). RViz with the fixed frame `map`
  shows it as "No transform from [floor_0]": view it with the fixed frame `floor_0` (a second view or window), or in
  the viewer (`viewer:=true`), which draws it on its own. To let StrataSLAM localize the navigation instead, stop the
  other localizer and keep `publish_tf: true`.
- **Viewers.** Best effort and latest
  message only: `/strataslam/scan` (the current LiDAR scan), `/strataslam/depth_cloud` (the current depth image as
  points) and `/strataslam/camera_view` (the image with its keypoints coloured by use).
- **Several instances in one ROS domain.** Each needs its own namespace: `namespace:=robot1` puts the node, its
  topics and its services under `/robot1` (`/robot1/strataslam/odometry`, `/robot1/strataslam_node/save_map`). A
  second node with the same name is refused: it prints "another /strataslam_node is running in ROS_DOMAIN_ID ..."
  and its launch ends, while the first keeps running. With `bag:=` it is only a warning, since both would just
  publish on the same topics. The frames are not namespaced: two robots in one domain need different
  `robot.odom_frame`/`base_frame` in their profiles.
- **Services.** `~/save_map`, `~/new_session`, `~/mark_floor_change` (tell it you just changed floor, with an
  optional transition type), `~/discard_map` (do not save the map when the node stops; an explicit `~/save_map`
  still saves).
- **Floors.** The system never closes a loop between two different floors. Stairs, ramps, elevators (with
  a barometer or an IMU) and vertical flight are detected and recorded. With neither an IMU nor a
  barometer an elevator ride is unobservable: a new floor is opened when the session cannot relocalize and the
  transition is recorded as `unknown`; give anchors when fusing (see [fusion](fusion.md)).
- **Sessions.** After a tracking loss a new session starts; it is reconnected to the floor by loop closure.
- **Stopping.** Ctrl+C saves the map of the run to `output.map_dir`, live as with `bag:=`, unless nothing was mapped
  since the last `~/save_map`. The map holds every floor in its own frame and, in `global_cloud.pcd`, the whole
  building as the live view showed it at the end: each floor where the odometry left it, with its drift.
  `strataslam fuse` refines it into `fused/global_cloud.pcd`.

## Processing a rosbag offline

```bash
ros2 launch strataslam_ros strataslam.launch.py profile:=profile.yaml bag:=/path/to/bag
```

With `bag:=` the node reads the rosbag2 and feeds the pipeline every message in order, one after the other, saves the
map to the profile's `output.map_dir` and exits. Ctrl+C stops it early and saves the map of what was processed so
far. Playing a bag with `ros2 bag play` instead goes through the
subscriptions' queues: when processing is slower than the sensor, scans are dropped and the result changes from run to
run (on `prueba_00` a single-thread run received 2179 and 2376 of 5038 scans, and the climbs came out different).
If you must play the bag (to watch it in another tool), `ros2 bag play -r 0.5` halves the rate and gives the node
time for every scan; the node logs how many scans it received when it saves the map.
Offline there is no loss, and the result is identical run after run whatever `runtime.threads` is: the LiDAR
registration groups its sums the same way with one thread or twenty (ADR-0048), so the map does not depend on the
machine's number of cores either. It supports LiDAR, camera, IMU, barometer and external odometry. Each colour image (or feature frame
with `features_topic:=`) is paired with the depth image of the closest stamp within 25 ms and with the latest
`camera_info`. Use the colour camera's `camera_info`: some drivers publish the depth one without K, and the node then
warns and drops the frames. The log ends with the number of messages, LiDAR scans and camera frames processed.

## Watching a run live

```bash
pip install --user --break-system-packages rerun-sdk==0.22.1 pyarrow==16.0.0   # once (see below)
ros2 launch strataslam_ros strataslam.launch.py profile:=profile.yaml viewer:=true [bag:=/path/to/bag]
```

`viewer:=true` opens a [Rerun](https://rerun.io) window, live or offline, with:

- **3D**: the cloud of each floor in its own colour, the path travelled in white and the robot in yellow, with a
  vertical pole through every level mapped so far that shows where it is from any angle; in magenta, what the
  sensors see right now (the LiDAR scan, the camera's depth) within the extent of the map, so that far returns do
  not move the framing, under `sensor` in the entity tree. Two tabs show it: **3D: floors**, each floor in its
  colour, and **3D: height**, the same cloud coloured by height from blue (lowest point mapped so far) to red
  (highest), as RViz's axis colour does; the colours are spread again over the whole map when it grows by more
  than half a metre. Each tab keeps its own camera.
- **Top view**: the map from above, each floor in its colour.
- **Level line**: the height of the robot over time, coloured by floor (grey while in a transition or lost). With a
  barometer, its filtered height is a magenta line, aligned with the robot's height at its first reading: in each
  transition it shows whether the barometer agrees with the odometry, and how much it drifts on a floor.
- **Log**: the node's log (transitions with their type and height change, loop closures, relocalizations, losses),
  when a transition starts going up or down, and whether the floor it ends on is new or an already mapped one
  recognised as the same level. Floors that *could* be one level (a drift of the odometry in z split it) appear as
  warnings, `floors 2 and 3 could be the same level ...`: the node checks each floor against the others as
  `strataslam fuse` does, and only reports it; confirm a pair with `strataslam fuse --same-level 2,3`.
- **Camera** and **Depth used**, next to the log when the profile has a camera (they appear with its first frame):
  the image with its keypoints, coloured by what the tracking did with each one, and the depth image as the
  tracking uses it (black where it is invalid or outside `depth_min`..`depth_max`, so no keypoint is taken there).
  Green: matched and used for the pose; orange: matched but rejected (a PnP outlier, or the frame was lost); red:
  not matched (the strongest become landmarks at the next keyframe); cyan: the camera is not the odometry source
  (LiDAR odometry) and its keypoints only serve loop closures. Hovering a keypoint shows its class. Few greens
  among many reds is what precedes a tracking loss: a corridor without texture, reflections, walls out of range.
  With feature-frame input (`features_topic:=`) there is no image, only the keypoints and the depth.

Everything is drawn in the odometry frame, the only one continuous across floors, so the floors sit where the
odometry left them (with its height drift); the building as fused is the output of `strataslam fuse`. The panel on
the left lists what is drawn, with an eye to show or hide each item: every floor cloud at once (`floors`), one
floor, the path or the robot. The timeline at the bottom rewinds the robot, the level line and the log; the clouds,
the path and the camera images show their latest state (keeping every refresh of them would take gigabytes on a long run). With a bag, processing waits for the window (up to 30 s) so it sees every pose, and
the floor cloud is refreshed every second; the map is the same as without the viewer.

The window lives with the launch. **Ctrl+C** closes it along with everything else (the SLAM saves its map). **Closing
the window while the SLAM runs** asks what to do: reopen the viewer (also when the question itself is closed; the
paths and floor clouds come back, the earlier timeline does not), stop the SLAM and save the map, or stop it without
saving (asked again, with the floors and keyframes that would be lost; the node's `~/discard_map`, after which an
explicit `~/save_map` still saves). To map without a window, launch without `viewer:=true`. **When a bag has been
processed** the window stays, to look
at the result, until you close it or press Ctrl+C. The question needs Tk (`python3-tk`, installed by
`scripts/install_deps.sh ros`); without it, closing the window closes only the viewer. A Rerun window that was
already open is reused and left open.

ROS 2 Jazzy ships numpy 1.x, and Rerun 0.23 and later need numpy 2: install 0.22.1, the last release that works
with it (installing a later one would replace the numpy that ROS's compiled packages use), with pyarrow 16.0.0, which
Rerun imports: pip would otherwise bring the latest pyarrow, which refuses to load with numpy 1.x ("pyarrow requires
NumPy 2.0 or newer"), and Rerun with it. `pip install -e "strataslam_tools[viewer]"` installs these
two. Outside ROS, with numpy 2, any recent `rerun-sdk` works.
