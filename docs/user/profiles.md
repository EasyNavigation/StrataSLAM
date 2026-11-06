# Robot profile

One YAML file describes the robot. Start from a file in `strataslam_ros/profiles/`: they are templates, one per kind
of robot, that hold every option, active when the example uses it and commented with its default otherwise, so you
uncomment what you need (a test keeps them complete). For your robot, copy the closest one and set its topics and
frames.

| Template | Platform | Sensors |
| --- | --- | --- |
| `ground_camera.yaml` | ground robot | RGB-D camera (RealSense D435i topics), the wheel odometry as its motion prior (TF `odom -> base_link`); no LiDAR, no IMU |
| `ground_lidar.yaml` | ground robot | 3D LiDAR only |
| `ground_lidar_camera_imu.yaml` | ground robot | 3D LiDAR (odometry), RGB-D camera (loop closures, colours), IMU (gravity only); camera and LiDAR loop closures |
| `handheld_camera_imu.yaml` | hand-held | RealSense D435i (RGB-D camera + its IMU) |
| `aerial_lidar_camera_imu.yaml` | drone | RealSense D435i (RGB-D camera + its IMU) and a 3D LiDAR |

```yaml
strataslam: {version: 1}
robot:   {platform: handheld, base_frame: base_link, odom_frame: odom, map_frame: map}
sensors:
  camera: {enabled: true, rgb_topic: /camera/color/image_raw, depth_topic: /camera/aligned_depth_to_color/image_raw,
           info_topic: /camera/color/camera_info, depth_min: 0.3, depth_max: 6.0, features: orb, max_features: 1000}
  lidar:  {enabled: false, topic: /points, type: 3d, min_range: 0.5, max_range: 50.0, deskew: auto}
  imu:    {enabled: true, topic: /camera/imu, floor_changes: true,
           noise: {accel_noise_density: 2.0e-3, gyro_noise_density: 1.7e-4, accel_random_walk: 3.0e-3, gyro_random_walk: 2.0e-5}}
  barometer: {enabled: false, topic: /pressure, reference_topic: "", temperature_k: 288.15, settle_time_s: 2.0}
  external_odometry: {enabled: false, topic: /odom}   # or frame: odom (TF odom -> base_frame)
fusion:     {odometry_from: auto, loop_closure_from: auto, colorize_cloud: true, vocabulary: ""}
multifloor: {mode: floors, floor_height_prior: 3.0, split_on_tracking_loss: true, tracking_loss_timeout: 2.0, min_vertical_change: 1.5}
output:     {map_dir: "~/strataslam_maps/{date}_{time}", voxel_size: 0.05, keyframe_translation: 0.25, keyframe_rotation_deg: 10.0, publish_tf: true}
runtime:    {seed: 42, threads: 0}
```

The profile is the whole configuration: what it says is what runs, and a saved map keeps it (`profile.yaml` in the
map folder). There are no camera presets any more (ADR-0053); a profile that still has `preset:` or `noise_preset:` is
refused with what to write instead. The values of the cameras that have been tested, with their drivers:

| Camera | Driver and launch | Topics | Depth range | IMU (`imu.topic`, `imu.noise`) |
| --- | --- | --- | --- | --- |
| Intel RealSense D435i | `realsense2_camera` 4.x: `ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ align_depth.enable:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2` (its defaults put the topics under `/camera/camera/` and leave the aligned depth and the IMU off) | `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info` | 0.3-6.0 m | `/camera/imu`; nominal BMI055 noise (`handheld_camera_imu.yaml`) |
| Luxonis OAK-D Lite | [oak_d_camera](https://github.com/jmguerreroh/oak_d_camera): `ros2 launch oak_d_camera camera.launch.py use_color:=true use_depth:=true use_imu:=true` (720p or lower) | `/oak/color/image`, `/oak/stereo/depth` (16UC1, aligned to colour), `/oak/color/camera_info` | 0.4-6.0 m | `/oak/imu` (raw BMI270); BMI270 datasheet: `accel_noise_density: 1.57e-3`, `gyro_noise_density: 1.22e-4` (random walks not in the datasheet: the nominal ones); `robot.base_frame: oak_base_frame` when hand-held (the IMU's mounting comes from the TF) |

Measure the IMU noise of your own unit with `strataslam imu-noise` ([imu-noise.md](imu-noise.md)); without `noise:` the
nominal values above are used.

Rules (validated before anything starts, all errors reported at once as `config error: <key.path>: <message>`):

- At least one of `camera` or `lidar` must be active. A present sensor section is enabled unless it says
  `enabled: false`.
- A camera needs `rgb_topic`, `depth_topic` and `info_topic`.
- `external_odometry` needs a `topic` (nav_msgs/Odometry) or a `frame` (the TF `frame -> base_frame`, what the
  wheel odometry of most ROS robots publishes), not both; `odometry_from: camera|lidar` needs that sensor.
- `barometer.reference_topic`: the sensor_msgs/FluidPressure of a fixed base station; the robot's height becomes the
  difference between the two, which cancels the weather (a front moving through changes the pressure by about 1 hPa,
  8 m, in an hour). `settle_time_s` is how long the low-pass filter of the barometric height takes to settle within
  1 % after a step (2 s by default).
- `external_odometry` is the robot's own odometry (wheels) as the motion prior of the camera tracking: it predicts
  each frame and carries the pose through a tracking loss, which then opens no new session. It is not an odometry
  source (`odometry_from: external` is an error) and with LiDAR odometry it is ignored, with a warning. It also
  vetoes camera motions it could not have made (ADR-0043).
- The IMU is used however it is mounted: its rotation and sign convention are learned from the motion when the TF or
  the driver are wrong (ADR-0044). It bounds the camera tracking's roll and pitch by gravity. `imu.floor_changes:
  false` keeps it to that (no elevator detection by vertical acceleration), for a ground robot's vibrating base.
- `platform: aerial` without an IMU is a strong warning.
- Every sensor section is a map: `imu: true` or `imu: 5` is an error (`expected a map`).
- `multifloor.mode` says how the levels are mapped:
  - `floors` (default): every level is a map of its own; `strataslam fuse` stacks them into the building.
  - `single`: one map in 3D, heights kept: a level with slopes or ramps, or an outdoor run. Ramps and climbs open no
    floor, and a tracking loss opens a new session of that level, never a new floor (its keyframes join the level
    when a loop closure reconnects them). `enabled: false`, from before `mode`, means this.
  - `planar`: one map on one plane, for a floor you know is flat (a corridor, a warehouse, an office): the base keeps
    the height where the run started and its roll and pitch come from the IMU's gravity (level without an IMU), so
    the odometry's drift in height never reaches the map. Ground robots only (`robot.platform: ground`). A real ramp
    is flattened, and what is seen from it lands at the wrong height: use `single` there (ADR-0045).
- Unknown keys are errors with a suggestion (`sensors.camra: unknown key; did you mean "camera"?`).

Which odometry is used (`auto`): camera only -> camera; LiDAR only -> LiDAR; both -> LiDAR, with camera loop
closure verified by LiDAR registration; the external odometry never (ADR-0043). With both sensors, `loop_closure_from: both`
also closes LiDAR loops: a ground robot that comes back along a corridor facing the other way is recognised by a
360-degree LiDAR, not by a camera.

`fusion.vocabulary` is the visual vocabulary of the camera loop closures: `default` is the one shipped with the
package (trained on TUM RGB-D offices, ADR-0054), or the path of your own (`strataslam vocab train`), relative to the
profile's folder so that a profile and its vocabulary travel together; `vocabulary:=` at launch overrides it. Without one, camera loop closure is off: the node warns at start-up and again when it saves the map.
A forward camera only recognises a place seen again in the same direction; a route that only comes back the other
way (along a corridor and back) gives it nothing to close.

`floor_height_prior` is your statement that floors are about that far apart. It is used weakly to place
floors whose height change nothing could measure, and it also corrects the height bias of a LiDAR climbing
a stairwell (ADR-0015). Set it to `null` if you do not know it.
