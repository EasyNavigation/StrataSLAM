"""Clouds written by the C++ core must be readable by Open3D (external compatibility)."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import core_executable, require_open3d


@pytest.fixture(scope="module")
def o3d():  # type: ignore[no-untyped-def]
    # A fixture, not a module-level import: a module-level skip, under the launch_testing pytest plugin that
    # `colcon test` loads with ROS, skipped the whole session (0 tests collected; audit AUD-023).
    return require_open3d()


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = core_executable("STRATASLAM_SAMPLE_BIN", "strataslam_write_sample_clouds")
    out = tmp_path_factory.mktemp("clouds")
    subprocess.run([exe, str(out)], check=True)
    return out


@pytest.mark.parametrize("name", ["sample.pcd", "sample.ply"])
def test_open3d_reads_points_and_colors(o3d, sample_dir: Path, name: str) -> None:  # type: ignore[no-untyped-def]
    cloud = o3d.io.read_point_cloud(str(sample_dir / name))
    pts = np.asarray(cloud.points)
    cols = np.asarray(cloud.colors)
    assert pts.shape == (100, 3)
    assert np.allclose(pts[10], [5.0, -2.5, 1.25])
    assert cols.shape == (100, 3)
    assert np.allclose(cols[10], np.array([10, 20, 245]) / 255.0, atol=1e-6)
