# Third-party dependencies

Every dependency is registered here with a pinned version, its license, its use and how it is
obtained. `scripts/license_audit.py` fails the build when a dependency found in a build file
(`find_package`, `FetchContent_Declare`, `package.xml`, `pyproject.toml`, `requirements*.txt`)
is not listed in the `Ids` column, when a listed license is not allowed, or when a dependency
marked `active` is not used anywhere. With `--ldd <library>` (CI runs it on the built
`libstrataslam_core.so`) it also checks every shared library the artifact loads against the `Ids`
(glob patterns allowed) and that each `transitive` row is really loaded.

ROS 2 platform packages (rclcpp, sensor_msgs, tf2, rosbag2, ...) are provided by the ROS 2 distribution (Apache-2.0 / BSD) and
are not tracked here.

Allowed licenses: MIT, BSD-2-Clause, BSD-3-Clause, BSD, Apache-2.0, MPL-2.0 (Eigen only), and the permissive
libpng-2.0, IJG and Zlib (ADR-0051). For libraries that an allowed dependency's Ubuntu package pulls in
(`transitive` rows), BSL-1.0 and X11 are also accepted (ADR-0036), and so is weak copyleft (LGPL-2.1, LGPL-3.0,
MPL-2.0) in `transitive` and `embedded` rows only: dynamically linked system libraries that StrataSLAM never
redistributes (ADR-0051). matplotlib (PSF-based Matplotlib license) is accepted as an optional plotting tool that
is never linked nor distributed (ADR-0036).
Forbidden for every row, transitive ones included: GPL, AGPL, non-commercial licenses (including network weights);
LGPL in anything this repository builds or declares itself.

Not tracked: the C/C++ toolchain and OS runtime (glibc, libstdc++, libgcc_s, libgomp, libgfortran;
the GCC ones under the GCC Runtime Library Exception), the ROS 2 platform packages, and the Python interpreter with
its standard library (PSF), `tkinter` included: the viewer's question when its window is closed uses it, through
Ubuntu's `python3-tk` (Tk is under the BSD-style Tcl/Tk license); without it the question is skipped.

`Status`: `active` = declared by a build file of this repository; `transitive` = loaded by the
built library through an `active` dependency (checked with `--ldd`); `embedded` = code compiled inside another
dependency (no library of its own on at least one install path, so `--ldd` does not require it to be loaded).

## Licence decisions

The 2026-11-04 audit (AUD-001, AUD-002, A2-04) found four libraries outside the allowed list of the specification;
ADR-0051 decides them:

1. **METIS (LGPL-2.1-or-later in GKlib)**, loaded by `libstrataslam_core.so` because libgtsam lists it as needed,
   on both install paths; StrataSLAM never calls it. Accepted as a transitive system library.
2. **Spectra (MPL-2.0)**, compiled into libgtsam. Accepted as embedded code.
3. **libpng (libpng-2.0) and libjpeg-turbo (IJG, BSD-3, Zlib)**: permissive; accepted. They replaced OpenCV's
   imgcodecs, which loaded GPL code (GDAL, poppler) into `tum_run` and `vocab_train`.

Whoever redistributes binaries that include METIS (a Docker image, a static build) must offer its source.

`libquadmath` (LGPL-2.1+) is no longer exempted as part of the GCC runtime: it has no such exception. None of the
artifacts loads it today.

<!-- audit:table -->
| Name | Ids | Version | License | Use | Source | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Eigen | Eigen3, eigen | 3.4 | MPL-2.0 | Linear algebra | apt | active |
| yaml-cpp | yaml-cpp, libyaml-cpp | 0.8 | MIT | Configuration and metadata files | apt | active |
| GTSAM | GTSAM, gtsam, libgtsam | 4.2 | BSD | Factor graphs, iSAM2, Lie-group traits of Pose4, IMU preintegration. Pulls in Boost, TBB and METIS (ADR-0007); compiled into it: Spectra, CCOLAMD, SuiteSparse_config and ceres's autodiff (rows below) | apt: `ros-jazzy-gtsam`, or Ubuntu's `libgtsam-dev` with the placeholder of `scripts/install_deps.sh core` | active |
| GoogleTest | GTest, gtest | 1.14 | BSD-3-Clause | C++ tests | apt | active |
| pytest | pytest, python3-pytest | 7.4 (tested 7.4.4) | MIT | Python tests | pip / apt | active |
| ruff | ruff | 0.17.3 (pinned) | MIT | Python lint (CI) | pip | active |
| mypy | mypy | 1.9.0 (pinned) | MIT | Python type checks (CI) | pip | active |
| numpy | numpy, python3-numpy | >= 1.26, < 2 with ROS 2 Jazzy (tested 1.26.4) | BSD-3-Clause | Python evaluation | pip / apt | active |
| PyYAML | pyyaml, python3-yaml | 6.0 (tested 6.0.1) | MIT | Profiles and map metadata in the CLI and the doctor | pip / apt | active |
| Open3D | open3d | >= 0.18 (tested 0.20.0) | MIT | Compatibility tests; viewer and export in tools (not in the core) | pip | active |
| small_gicp | small_gicp | v1.0.0 (commit fd29d8c) | MIT | Point cloud registration (GICP); contains a kd-tree derived from nanoflann and SO(3) code from Sophus (rows below) | FetchContent (pinned commit) | active |
| OpenCV | OpenCV, libopencv-dev, libopencv_* | 4.6 | Apache-2.0 | ORB, PnP RANSAC (not imgcodecs: on Ubuntu it loads GDAL and poppler, GPL; ADR-0047) | apt | active |
| Google Benchmark | benchmark | 1.8 | Apache-2.0 | Performance benchmarks | apt | active |
| pybind11 | pybind11 | 2.11 | BSD-3-Clause | Python bindings (profile validation from the launch file) | apt | active |
| Rerun | rerun-sdk, rerun | 0.22.1 with ROS 2 Jazzy (numpy 1.x), any later otherwise | MIT | Live view of a run (`viewer:=true`; optional extra `viewer`, 0.22.1; not linked) | pip | active |
| Apache Arrow (Python) | pyarrow | 16.0.0 with ROS 2 Jazzy (numpy 1.x; later releases need numpy 2), any otherwise | Apache-2.0 | Imported by Rerun; installed with it for the live view (optional extra `viewer`) | pip | active |
| matplotlib | matplotlib | >= 3.6 (tested 3.11) | PSF-2.0 | Plots of `strataslam imu-noise --plot` and the documentation scripts (optional extra `plot`) | pip | active |
| Boost | Boost, libboost_* | 1.83 | BSL-1.0 | filesystem, serialization, timer, required by GTSAM | apt (through GTSAM) | transitive |
| oneTBB | TBB, libtbb, libtbbmalloc | 2021.11 | Apache-2.0 | GTSAM's parallel elimination | apt (through GTSAM) | transitive |
| METIS | METIS, libmetis, libmetis-gtsam | 5.1 | Apache-2.0 AND LGPL-2.1-or-later (GKlib/getopt.c, GKlib/gk_mksort.h, from glibc) | GTSAM's nested-dissection ordering, loaded because libgtsam needs it although StrataSLAM never uses it (bundled with `ros-jazzy-gtsam`, `libmetis5` with Ubuntu's GTSAM). Accepted by ADR-0051 (see "Licence decisions") | apt (through GTSAM) | transitive |
| BLAS / LAPACK / ATLAS | libblas, liblapack, libatlas | 3.12 / 3.10 | BSD-3-Clause | Linear-algebra backend of OpenCV and GTSAM | apt (through OpenCV) | transitive |
| zlib | libz | 1.3 | Zlib | Compression, used by OpenCV | apt (through OpenCV) | transitive |
| X.Org client libraries | libX11, libxcb, libXau, libXdmcp | Ubuntu 24.04 | X11 | Pulled in by Ubuntu's OpenCV build (no GUI is used) | apt (through OpenCV) | transitive |
| libglvnd | libGLX, libGLdispatch | 1.7 | MIT | Pulled in by Ubuntu's OpenCV build (no OpenGL is used) | apt (through OpenCV) | transitive |
| libbsd / libmd | libbsd, libmd | 0.12 / 1.1 | BSD-3-Clause | Pulled in by the X.Org libraries | apt (through OpenCV) | transitive |
| types-PyYAML | types-PyYAML | >= 6.0 | Apache-2.0 | Type stubs for mypy (CI) | pip | active |
| libpng | PNG, libpng-dev, libpng16 | 1.6 | libpng-2.0 | PNG files in `strataslam_tum_run` and `strataslam_vocab_train` (`strataslam::image_io`, ADR-0047). Accepted by ADR-0051 (see "Licence decisions") | apt | active |
| libjpeg-turbo | JPEG, libjpeg-dev, libjpeg | 2.1 | IJG AND BSD-3-Clause AND Zlib | JPEG files in `strataslam_vocab_train` (`strataslam::image_io`, ADR-0047). Accepted by ADR-0051 (see "Licence decisions") | apt | active |
| Spectra | Spectra | 0.9 | MPL-2.0 | Eigenvalue solvers compiled into libgtsam (both GTSAM packages); never called by StrataSLAM. Accepted by ADR-0051 (see "Licence decisions") | apt (inside GTSAM) | embedded |
| SuiteSparse CCOLAMD / SuiteSparse_config | libccolamd, libsuitesparseconfig | 3.x / 7.x | BSD-3-Clause | GTSAM's COLAMD ordering: compiled into `ros-jazzy-gtsam`, separate libraries with Ubuntu's `libgtsam` | apt (through GTSAM) | embedded |
| ceres (autodiff headers) | ceres | GTSAM 4.2's copy | BSD-3-Clause | Automatic differentiation helpers compiled into libgtsam | apt (inside GTSAM) | embedded |
| nanoflann / FLANN (derived kd-tree) | nanoflann | small_gicp v1.0.0's copy | BSD-2-Clause | `small_gicp/ann/kdtree.hpp` is derived from it | FetchContent (inside small_gicp) | embedded |
| Sophus (SO(3) code) | Sophus | small_gicp v1.0.0's copy | MIT | `small_gicp/util/lie.hpp` takes its SO(3) exponential map from it | FetchContent (inside small_gicp) | embedded |
<!-- /audit:table -->

## Distributed data

| File | Derived from | Licence | Use |
| --- | --- | --- | --- |
| `strataslam_ros/vocabularies/tum_rgbd.bin` | TUM RGB-D fr1/xyz and fr2/xyz, 892 colour images (ORB cluster centres and word weights; no image) | CC BY 4.0, attribution in `strataslam_ros/vocabularies/README.md` | `fusion.vocabulary: default` (ADR-0054); rebuilt byte for byte by `scripts/train_default_vocabulary.sh` |

## Datasets (evaluation only, not distributed)

Downloaded by `scripts/download_datasets.sh` for the published results; nothing of them is in the repository or in
the nightly artifacts, except the vocabulary derived from TUM RGB-D above.

| Dataset | Licence | Use |
| --- | --- | --- |
| TUM RGB-D (fr1/xyz, fr2/xyz) | CC BY 4.0 | Visual odometry ATE (`strataslam_tum_run`) |
| KITTI raw 2011_09_30_drive_0027 | CC BY-NC-SA 3.0 (non-commercial) | LiDAR pipeline ATE (`strataslam_kitti_run`); its ground truth, derived from the OXTS data, is not redistributed (the nightly artifact excludes `kitti_0027*.gt.tum`) |

## Not built

- **XFeat** (Apache-2.0 weights) through **ONNX Runtime** (MIT): optional learned features named in the
  specification; this release uses ORB only, so neither is downloaded nor linked. They will get rows
  when they are built.

## Rejected

- **TEASER++** — not vendored: its max-clique/certification stack (pmc) is heavy for registrations that have
  4 degrees of freedom and a handful of correspondences. `align_pose4_gnc` implements the same family of
  estimator (graduated non-convexity with a truncated least-squares loss, Yang et al. RA-L 2020) specialised
  to Pose4, written from the paper. See ADR-0011.

- **DBoW3** — its LICENSE.txt is a modified BSD with a fourth clause requiring notification of the
  original author on redistribution. That is not one of the allowed licenses. StrataSLAM implements
  its own bag-of-words (see `docs/decisions.md`, ADR-0003).
- **ORB-SLAM `ORBvoc.txt`** — GPL-derived vocabulary; the vocabulary is trained with
  `strataslam vocab train` instead.
