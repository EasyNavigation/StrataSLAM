# Install and build

Ubuntu 24.04, C++17, CMake >= 3.22, Python 3.12. ROS 2 Jazzy for the node, the messages and the command
line; the core library and the simulator also build without ROS.

The package lists live in one place, [scripts/install_deps.sh](../../scripts/install_deps.sh); CI runs both
recipes below in clean containers (`scripts/check_install.sh` does the same locally with Docker).

## Full system (ROS 2 Jazzy)

[Install ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) first (its apt
repository provides `ros-jazzy-gtsam`). Then clone the repository **inside** the `src/` folder of a
workspace and build it with colcon:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/strataslam_ws/src && cd ~/strataslam_ws/src
git clone <this repository> strataslam
strataslam/scripts/install_deps.sh ros
cd ~/strataslam_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
strataslam --help            # the command line is now on PATH
```

colcon builds five packages: `strataslam_core`, `strataslam_sim`, `strataslam_msgs`, `strataslam_ros` and
`strataslam_tools` (four without the simulator, see below). If it reports fewer, check that you cloned into
`src/` (the `superbuild/` folder carries a `COLCON_IGNORE` on purpose).

The simulator (`strataslam_sim`) is only needed for the demo of [mapping.md](mapping.md), the ROS 2 tests and
the synthetic scenarios. To map with your own robot you can leave it out:
`colcon build --symlink-install --packages-ignore strataslam_sim --cmake-args -DCMAKE_BUILD_TYPE=Release` builds
the other four packages, without `strataslam_sim_export_bag`.

The first build downloads small_gicp at a pinned commit (FetchContent): it needs network once. Building takes a few
minutes (about 2 minutes on 20 threads, 10 or more on a laptop).

Run the tests with `colcon test && colcon test-result --verbose` (the ROS 2 tests generate rosbags with the
simulator, so they take several minutes; `colcon test` itself returns 0 even when tests fail, `colcon test-result`
does not). Three Python tests check that Open3D reads the clouds StrataSLAM writes; they are skipped unless Open3D
is installed (`pip install open3d`).

## Core and simulator only (no ROS)

```bash
git clone <this repository> strataslam && cd strataslam
scripts/install_deps.sh core
cmake -S superbuild -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure      # unit tests and the scenarios S01-S14
```

`install_deps.sh core` installs Ubuntu's `libgtsam-dev` plus the libraries its CMake export needs (Boost,
METIS, SuiteSparse) and creates an empty `libCppUnitLite.a`: the Ubuntu package references that file
without shipping it, and CMake only checks that it exists (ADR-0007). If ROS 2 Jazzy is installed you can
use its GTSAM instead: `source scripts/dev_env.sh` before `cmake`.

## Python tools without colcon

The `strataslam` command line is a normal Python package. Outside a colcon workspace:

```bash
pip install -e "strataslam_tools[dev]"      # dev adds pytest, ruff, mypy and Open3D
```

`strataslam map` and `strataslam doctor` on rosbags need a sourced ROS 2 environment; `fuse`, `export`, `eval`
and `imu-noise` (with a CSV) do not.

## Notes

- small_gicp (MIT) is downloaded and built by CMake at a pinned commit (needs network the first time).
- XFeat (learned features through ONNX Runtime) is not built; ORB is used.
- Sanitizers: `-DSTRATASLAM_ENABLE_SANITIZERS=ON`. Coverage: `-DSTRATASLAM_ENABLE_COVERAGE=ON` then
  `scripts/check_coverage.py`.
- The tests of `strataslam_tools` skip what needs an optional dependency that is missing (Open3D for the three
  cloud-compatibility tests, Rerun for the live-view test, the core executables `strataslam_write_sample_clouds`
  and `strataslam_eval_tool`). `STRATASLAM_TEST_STRICT=1`, as in CI, makes each of those a failure instead.
- What the CI jobs need beyond `install_deps.sh` (`.github/workflows/ci.yml`): `gcovr` for coverage,
  `clang-tidy` and `libomp-dev` for the C++ lint, Python 3.12 with `pip install -e "strataslam_tools[dev]"` for
  ruff, mypy and pytest. `ctest` takes about 20 minutes in series (the scenarios); `-j` with as many jobs as cores
  shortens it. `colcon test` does not fail the shell on test failures: `colcon test-result --verbose` does.

Next: [write a profile](profiles.md).
