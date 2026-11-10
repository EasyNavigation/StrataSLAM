# Fusing floors, export, evaluation

```bash
strataslam fuse <map_dir> [--formulation lie4|se3soft|euler] [--anchors anchors.yaml] [--floor-height-prior 3.0]
strataslam export <map_dir> [--format pcd|ply]       # default: pcd
strataslam export <map_dir> --format grid [--resolution 0.05] [--min-height 0.05] [--max-height 2.0]
strataslam eval <map_dir> --gt <ground_truth_dir> [--json]
```

`export --format grid` writes an occupancy grid per floor, `export/floor_<k>.pgm` and `.yaml` as nav2's map_server
loads them (trinary), in the frame of that floor's cloud. A cell is occupied where the keyframes' points (and, with a
planar scanner and a camera, their depth clouds) lie between `--min-height` and `--max-height` above the robot's base
(the obstacles a ground robot meets: the floor and a ceiling lamp are not), free where the rays from the sensor to
the points cross it, and unknown where nothing was seen. The band is counted in layers of 0.1 m and a ray clears, at
each step, the layer it is at: the scan passing under a table and the camera's rays passing over it to the wall do
not erase the table top the camera saw. In each layer a cell is as dark as the fraction of its observations that hit
it (a person who walked by leaves it free), and the cell is the darkest of its layers. The sensors' positions come
from the map (`cloud_origin`, `depth_origin`); a map written before them has its rays start at the base, horizontal. Any map of a ground robot gives one: a planar scanner's (its sweep
is the band), a 3D LiDAR's or a camera's. A map written
before `cloud_frame` (ADR-0062) keeps its clouds in a sensor frame whose pose it does not know, and is refused (ADR-0067).

`fuse` puts every floor in the frame of the lowest floor, optimizing one `Pose4 = SE(2)×R` per floor (yaw
about gravity + 3D translation). Constraints, most to least reliable: transitions with continuous tracking,
barometric height, verified cross-floor matches, structural anchors, the floor height prior. A wrong
constraint is rejected (graduated non-convexity). A floor that no measurement links to the others (an elevator with
neither IMU nor barometer, a floor opened after a tracking loss) is placed where the odometry left it, the weakest
constraint of all, and the report says so (`odometry_placement`): give anchors to place it better. It writes `<map_dir>/fused/`: `floor_transforms.yaml`,
`global_cloud.pcd` and `fusion_report.yaml` (accepted/rejected constraints, residuals, marginal sigma of
every floor, warnings such as "floor 2 has no usable rotation constraint").

## Anchors

Structural points that look the same from several floors (elevator door frame, shaft corners), measured in
each floor's frame (click them in each floor's cloud). Entry *i* of every floor is the same physical point.

```yaml
anchors:
  - name: elevator_shaft
    use_z: false        # a structure repeated on every floor is at a different height on each
    sigma: 0.03         # marking noise, m
    points:
      0: [[20.4, -3.3, 0.0], [22.8, -3.3, 0.0], [20.4, -0.2, 0.0], [22.8, -0.2, 0.0]]
      1: [[0.2, -3.1, 0.0], [2.6, -3.0, 0.0], [0.1, 0.0, 0.0], [2.5, 0.1, 0.0]]
```

Wide features (shaft corners 2-3 m apart) constrain the rotation far better than a door frame alone.

## Evaluation

`eval` expects `trajectory.tum` (building-frame ground truth) and optionally `floors.csv`
(`stamp_ns,floor`, the true floor at each instant, -1 in transit). It reports ATE (4-DoF aligned when the map
is fused) and, with `floors.csv`, the collapse rate (accepted loops between keyframes of different true floors
/ all accepted loops).

## Floors that are really one level

The odometry of a LiDAR drifts in z: in a long corridor it can read a climb of a metre or two that did not
happen, and one level then appears as two floors. `fuse` cannot tell this from a gentle ramp on its own (identical
floors stacked on one another look the same as one floor seen twice), so it **reports** the pairs that could be one
level and leaves the decision to you. In `fusion_report.yaml` (`same_level_candidates`) and in the warnings:

```
floors 2 and 3 could be the same level: compatible structure (score 0.51), heights 1.65 m apart; confirm with `same_level` if they are
```

A candidate needs compatible structure (the structure of the two floors overlaps) and a height difference below
`same_level_max_dz` (2 m by default; floors a whole storey apart are never listed). To confirm one:

```bash
strataslam fuse <map_dir> --same-level 2,3          # several pairs: --same-level "2,3;0,4"
```

The frames of the two floors are constrained to the same height and the height of the transition between them
is released (its x, y and yaw are kept), so the second floor lands at the height of the first.
