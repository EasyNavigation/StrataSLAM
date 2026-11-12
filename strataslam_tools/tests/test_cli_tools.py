"""Map reading, export and the CLI parser."""

from pathlib import Path

import numpy as np
import pytest
from conftest import require_open3d

from strataslam_tools import cli, mapio


def make_map(root: Path) -> Path:
    (root / "floor_000").mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        "strataslam_version: 0.2.0\nformat_version: 1\nprofile: profile.yaml\ntransitions: transitions.yaml\n"
        "floors:\n  - {id: 0, dir: floor_000, keyframes: 2}\n"
    )
    (root / "profile.yaml").write_text("strataslam: {version: 1}\n")
    (root / "transitions.yaml").write_text("transitions: []\n")
    (root / "floor_000" / "metadata.yaml").write_text("floor_id: 0\n")
    (root / "floor_000" / "keyframes.csv").write_text(
        "id,stamp_ns,session,x,y,z,qx,qy,qz,qw\n0,1000000000,0,0,0,1.2,0,0,0,1\n1,2000000000,0,1,0,1.2,0,0,0,1\n"
    )
    (root / "floor_000" / "edges.csv").write_text(
        "type,from,to,measurement,info_dim,information\nodometry,0,1,1;0;0;0;0;0;1,1,2.5\n"
    )
    pts = np.random.default_rng(0).uniform(-2, 2, (50, 3)).astype(np.float32)
    mapio.write_pcd(root / "floor_000" / "floor_cloud.pcd", pts.astype(np.float64))
    return root


def test_validate_and_read(tmp_path: Path) -> None:
    root = make_map(tmp_path / "map")
    assert mapio.validate_map(root) == []
    kf = mapio.read_keyframes(root / "floor_000")
    assert kf.shape == (2, 10) and kf[1, 3] == 1.0
    edges = mapio.read_edges(root / "floor_000")
    assert edges[0]["type"] == "odometry" and edges[0]["info_dim"] == 1 and edges[0]["information"] == [2.5]


def test_unknown_format_version_and_missing_files(tmp_path: Path) -> None:
    root = make_map(tmp_path / "map")
    (root / "manifest.yaml").write_text("strataslam_version: x\nformat_version: 99\nfloors: []\n")
    assert "unsupported map format version 99" in mapio.validate_map(root)[0]
    assert "missing" in mapio.validate_map(tmp_path / "nothing")[0]


@pytest.mark.parametrize("colored", [False, True])
def test_pcd_ply_writers_roundtrip_and_open3d_reads_them(tmp_path: Path, colored: bool) -> None:
    o3d = require_open3d()
    rng = np.random.default_rng(2)
    pts = rng.uniform(-3, 3, (200, 3)).astype(np.float32).astype(np.float64)
    cols = rng.integers(0, 255, (200, 3), dtype=np.uint8) if colored else None
    for write, read, name in ((mapio.write_pcd, mapio.read_pcd, "a.pcd"), (mapio.write_ply, mapio.read_ply, "a.ply")):
        write(tmp_path / name, pts, cols)
        back, bc = read(tmp_path / name)
        assert np.array_equal(back, pts)
        assert (bc is None) == (cols is None)
        if cols is not None:
            assert np.array_equal(bc, cols)
        cloud = o3d.io.read_point_cloud(str(tmp_path / name))
        assert np.allclose(np.asarray(cloud.points), pts, atol=1e-6)


def test_export_converts_floor_clouds(tmp_path: Path) -> None:
    root = make_map(tmp_path / "map")
    assert cli.main(["export", str(root), "--format", "ply"]) == 0
    pts, _ = mapio.read_ply(root / "export" / "floor_000.ply")
    assert pts.shape == (50, 3)
    assert cli.main(["export", str(tmp_path / "missing"), "--format", "ply"]) == 1
    assert cli.main(["export", str(root)]) == 0  # default format
    pts_pcd, _ = mapio.read_pcd(root / "export" / "floor_000.pcd")
    assert pts_pcd.shape == pts.shape


def test_export_takes_the_fused_building_or_else_the_one_saved_with_the_map(tmp_path: Path) -> None:
    root = make_map(tmp_path / "map")
    placed = np.asarray([[1.0, 2.0, 3.0]] * 4, dtype=np.float32)
    mapio.write_pcd(root / "global_cloud.pcd", placed, None)  # as the odometry left the floors
    assert cli.main(["export", str(root), "--format", "ply"]) == 0
    pts, _ = mapio.read_ply(root / "export" / "global_cloud.ply")
    assert pts.shape == (4, 3)
    (root / "fused").mkdir()
    mapio.write_pcd(root / "fused" / "global_cloud.pcd", placed[:2], None)  # refined by `strataslam fuse`
    assert cli.main(["export", str(root), "--format", "ply"]) == 0
    pts, _ = mapio.read_ply(root / "export" / "global_cloud.ply")
    assert pts.shape == (2, 3)


def test_eval_on_a_tiny_map(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_map(tmp_path / "map")
    gt = tmp_path / "gt"
    gt.mkdir()
    rows = [[t, t, 0.0, 1.2, 0, 0, 0, 1] for t in np.linspace(0.5, 2.5, 21)]
    np.savetxt(gt / "trajectory.tum", np.asarray(rows))
    # Three keyframes are needed for an alignment; add one.
    kf = root / "floor_000" / "keyframes.csv"
    kf.write_text(kf.read_text() + "2,1500000000,0,0.5,0,1.2,0,0,0,1\n")
    assert cli.main(["eval", str(root), "--gt", str(gt), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"ate_rmse_m"' in out and '"keyframes": 3.0' in out


def test_parser_covers_every_command() -> None:
    p = cli.build_parser()
    for argv in (["doctor", "p.yaml", "--bag", "b", "--json"], ["map", "p.yaml"], ["fuse", "m", "--formulation", "euler", "--anchors", "a.yaml"],
                 ["export", "m", "--format", "pcd"], ["eval", "m", "--gt", "g"], ["vocab", "train", "imgs", "--out", "v.bin"]):
        assert p.parse_args(argv).command == argv[0]
    assert p.parse_args(["export", "m"]).format == "pcd"  # PCD unless asked otherwise
    with pytest.raises(SystemExit):
        p.parse_args(["export", "m", "--format", "obj"])


def test_core_tools_are_found_through_cmake_prefix_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # colcon lists the plain-CMake strataslam_core only in CMAKE_PREFIX_PATH (found by the user walkthrough).
    exe = tmp_path / "lib" / "strataslam_core" / "strataslam_fuse_test_tool"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("AMENT_PREFIX_PATH", "")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(tmp_path))
    assert cli._find_tool("strataslam_fuse_test_tool", "strataslam_core") == str(exe)
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "")
    with pytest.raises(SystemExit):
        cli._find_tool("strataslam_fuse_test_tool", "strataslam_core")


def test_a_missing_tool_says_how_to_make_it_found(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # A new terminal without the workspace sourced: the message names the command that fixes it.
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("AMENT_PREFIX_PATH", "")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "")
    with pytest.raises(SystemExit) as e:
        cli._find_tool("strataslam_fuse_test_tool", "strataslam_core")
    assert "install/setup.bash" in str(e.value) and "build/strataslam_core" in str(e.value)


def test_doctor_without_ros_says_to_source_it(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from strataslam_tools import doctor

    def no_ros(*_a: object, **_k: object) -> tuple[int, str]:
        raise ModuleNotFoundError("No module named 'rclpy'")

    monkeypatch.setattr(doctor, "run", no_ros)
    assert cli.main(["doctor", "p.yaml"]) == 3
    err = capsys.readouterr().err
    assert "source /opt/ros/jazzy/setup.bash" in err and "Traceback" not in err


def test_fuse_passes_same_level_and_stacking_to_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    # docs/user/fusion.md tells users to run `strataslam fuse <map> --same-level 2,3`.
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_find_tool", lambda name, package: "/bin/" + name)
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert cli.main(["fuse", "m", "--same-level", "2,3;0,4", "--stacking", "off"]) == 0
    assert calls[0][-4:] == ["--same-level", "2,3;0,4", "--stacking", "off"]


def test_corrupt_map_files_give_a_message_not_a_traceback(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    # Audit A6-06: a manifest that is not what the readers expect.
    from strataslam_tools import cli
    (tmp_path / "manifest.yaml").write_text("format_version: 1\nstrataslam_version: x\nfloors: [{id: zero, dir: floor_000}]\n")
    (tmp_path / "floor_000").mkdir()
    (tmp_path / "floor_000" / "keyframes.csv").write_text("id,stamp_ns\nnot,a,number\n")
    code = cli.main(["export", str(tmp_path)])
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err
