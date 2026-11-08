# Map data format (version 1)

Open and versioned; readable without StrataSLAM. `manifest.yaml` is written last, so its presence
marks a complete map. Readers reject any `format_version` other than 1.

```
<map_dir>/
├── manifest.yaml        strataslam_version (easyslam_version in maps from 0.2.x), format_version, profile, transitions, floors[{id,dir,keyframes}]
├── profile.yaml         verbatim copy of the profile used for capture
├── transitions.yaml     list of transition records
├── floor_000/
│   ├── metadata.yaml    floor_id, gravity, ground_normal, reference_pressure_pa, sensors,
│   │                    sessions, alignment_residual_rad, altitude_m, cloud_frame
│   ├── keyframes.csv    id,stamp_ns,session,x,y,z,qx,qy,qz,qw   (pose in the floor frame)
│   ├── edges.csv        type,from,to,measurement,info_dim,information
│   ├── clouds/<id>.pcd  per-keyframe cloud (binary PCD v0.7: x y z [rgb], float32), in the frame `cloud_frame` names
│   ├── visual/<id>.bin  keypoints + descriptors (only keyframes with a camera observation)
│   ├── lidar/<id>.bin   Scan Context + metadata (only when loop closure uses the LiDAR)
│   └── floor_cloud.pcd  voxelized floor cloud (optional)
├── floor_001/ ...
└── fused/               written by `strataslam fuse` (optional)
    ├── floor_transforms.yaml   T_map_floor of every floor (Pose4: yaw, x, y, z) and its covariance
    ├── fusion_report.yaml      constraints used/rejected, warnings, same-level candidates
    └── global_cloud.pcd        all floor clouds in the map frame (binary PCD; ADR-0039)
```

`visual/`, `lidar/`, `clouds/` and `floor_cloud.pcd` are optional: a reader must accept a floor without
them (a camera + LiDAR map whose loops are closed by the camera has no `lidar/`, a LiDAR-only map no
`visual/`).

Conventions: SI units, radians, timestamps in integer nanoseconds, quaternions as (x, y, z, w),
floor frames gravity-aligned (z up).

Per-keyframe clouds: with `cloud_frame: base` (every map written since the field exists) a cloud is in its
keyframe's base frame, and the keyframe's pose in `keyframes.csv` places it in the floor frame. Maps written before
have no `cloud_frame` field, which reads as `sensor`: their clouds are in the frame of the sensor that took them
(camera optical frame or LiDAR frame), whose mounting the map does not record. `floor_cloud.pcd` and the fused
`global_cloud.pcd` are in the floor and map frames in both. ADR-0062.

## Binary containers (`visual/*.bin`, `lidar/*.bin`)

Little-endian. Header: `"ESLM"` (4 bytes, from the former name easyslam) · u32 format version · u32 kind (1 visual, 2 lidar) ·
u64 payload size. Payload:

- Visual: u32 n · u32 descriptor_cols · n × (f32 x, y, response, depth) · n·descriptor_cols bytes.
- Lidar: u32 rings · u32 sectors · u32 num_points · f32 max_range · rings·sectors f32 scan
  context (row-major) · rings f32 ring key.

Integrity: readers check the magic, version, kind, declared sizes against the file size and the column count
of the CSV files, and fail with an error (never a crash or an allocation from an unchecked size). There is
no checksum: a byte flipped inside the data of a `.bin` or `.pcd` is read as data. A checksum would change
the format (version 2); until then, copy maps with a tool that verifies them (`rsync -c`, `sha256sum`).

## Transitions

`transitions.yaml` → `transitions:` list of `{type, from, to, kf_before, kf_after, delta_z|null,
covariance (16 values, row-major 4×4, Pose4 tangent order φ, ρx, ρy, ρz), continuous_tracking,
relative|null}`. `relative` is `[yaw, x, y, z]`: the base pose at `kf_after` relative to the base at
`kf_before` in the continuous odometry frame (z from the barometer for elevators).
`type` ∈ stairs, ramp, elevator, aerial, unknown.

## Edges

See ADR-0005 in `decisions.md`.
