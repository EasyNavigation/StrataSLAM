"""Several launches in one ROS domain: each node runs in its own container, a duplicate refuses to run, a
namespace separates two instances, and a node that cannot start ends its launch.

Regression test for the 2026-11-04 audit: two launches started together loaded one node into the other's
container (fixed container name) and both `ros2 launch` hung; a missing vocabulary also left the launch hanging.
"""

import os
import subprocess
import time
from pathlib import Path

from ros_helpers import PROFILES, profile_with_map_dir

DOMAIN = str(10 + (os.getpid() + 37) % 80)


def env() -> dict:
    return dict(os.environ, ROS_DOMAIN_ID=DOMAIN, RCUTILS_COLORIZED_OUTPUT="0")


def launch(profile: Path, log: Path, *args: str) -> subprocess.Popen:
    return subprocess.Popen(["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={profile}", *args],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env())


def nodes() -> list[str]:
    out = subprocess.run(["ros2", "node", "list", "--no-daemon"], capture_output=True, text=True, env=env(), timeout=30, check=False)
    return out.stdout.split()


def topics() -> list[str]:
    return subprocess.run(["ros2", "topic", "list", "--no-daemon"], capture_output=True, text=True, env=env(), timeout=30,
                          check=False).stdout.split()


def eventually(check, timeout: float = 40.0) -> bool:  # type: ignore[no-untyped-def]
    """Discovery without the daemon is incomplete for a moment: ask again until it holds or time is up."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(1.0)
    return False


def wait_for_node(name: str, timeout: float = 40.0) -> bool:
    return eventually(lambda: name in nodes(), timeout)


def wait_exit(p: subprocess.Popen, log: Path, timeout: float = 60.0) -> str:
    """Waits for a launch to end by itself; fails with the end of its log if it does not."""
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{log.name}: the launch did not end within {timeout:.0f} s\n" + log.read_text()[-3000:]) from None
    return log.read_text()


def stop(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.send_signal(2)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()


def test_instances_in_one_domain(tmp_path: Path) -> None:
    profile = profile_with_map_dir(PROFILES / "ground_lidar.yaml", tmp_path / "map", tmp_path / "profiles" / "ground_lidar.yaml")
    first = launch(profile, tmp_path / "first.log")
    others: list[subprocess.Popen] = []
    try:
        assert wait_for_node("/strataslam_node"), (tmp_path / "first.log").read_text()[-2000:]

        # Same name, same domain: the newcomer refuses and its launch ends; the first keeps running.
        dup = launch(profile, tmp_path / "dup.log")
        others.append(dup)
        dup_log = wait_exit(dup, tmp_path / "dup.log")
        assert "another /strataslam_node is running" in dup_log, dup_log[-2000:]
        assert first.poll() is None, "the first instance stopped when a duplicate was launched"

        # A namespace separates them: both run, each with its own topics.
        ns = launch(profile, tmp_path / "ns.log", "namespace:=robot2")
        others.append(ns)
        assert wait_for_node("/robot2/strataslam_node"), (tmp_path / "ns.log").read_text()[-2000:]
        time.sleep(3.0)
        assert ns.poll() is None and first.poll() is None
        assert eventually(lambda: len({n for n in nodes() if "strataslam_container" in n}) == 2), nodes()
        assert eventually(lambda: {"/strataslam/odometry", "/robot2/strataslam/odometry"} <= set(topics())), topics()

        # A node that cannot start ends its launch instead of leaving the container waiting.
        bad = launch(profile, tmp_path / "bad.log", "vocabulary:=/no/such/vocab.bin", "namespace:=robot3")
        others.append(bad)
        assert "cannot start" in wait_exit(bad, tmp_path / "bad.log")
    finally:
        for p in others + [first]:
            stop(p)
