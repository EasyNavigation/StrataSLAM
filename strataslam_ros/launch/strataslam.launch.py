"""ros2 launch strataslam_ros strataslam.launch.py profile:=profile.yaml [mode:=map] [features_topic:=...] [vocabulary:=...]
                                                   [bag:=...] [viewer:=true] [namespace:=robot1]

The profile is validated with the core's own code (pybind11) before any node starts: every error is
printed as `config error: <path>: <message>` and the launch is aborted.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from strataslam_ros import validate_profile

VIEWER_ONLY = 3  # exit code of the viewer when the user closed only the viewer (strataslam_ros.viewer.VIEWER_ONLY)


def ends_on_slam_exit(viewer_running: bool) -> bool:
    """When the SLAM ends (a bag processed, a node that could not start), the launch ends with it, unless the viewer is
    still open: then the window stays, to look at the result, until it is closed or Ctrl+C is pressed."""
    return not viewer_running


def ends_on_viewer_exit(returncode: int | None) -> bool:
    """When the viewer ends, the launch ends (the window was closed after the SLAM, or "stop the SLAM" was chosen),
    unless the user closed only the viewer and the SLAM goes on."""
    return returncode != VIEWER_ONLY


def _launch_setup(context, *args, **kwargs):
    profile = os.path.expanduser(LaunchConfiguration("profile").perform(context))
    errors, warnings = validate_profile(profile)
    for w in warnings:
        print(w, file=sys.stderr)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        # Abort before starting any node.
        # The errors are printed once, above; the exception only says that the launch stops (audit AUD-093).
        raise RuntimeError(f"invalid profile {profile}: {len(errors)} error(s), listed above; nothing was started")
    mode = LaunchConfiguration("mode").perform(context)
    if mode != "map":
        raise RuntimeError(f"unknown mode '{mode}' (only 'map' is available)")
    params = {
        "profile": profile,
        "vocabulary": LaunchConfiguration("vocabulary").perform(context),
        "features_topic": LaunchConfiguration("features_topic").perform(context),
        "bag": os.path.expanduser(LaunchConfiguration("bag").perform(context)),
    }
    namespace = LaunchConfiguration("namespace").perform(context).strip("/")
    actions = []
    if LaunchConfiguration("viewer").perform(context).lower() in ("true", "1", "yes"):
        try:
            import rerun  # noqa: F401  (checked here, so a missing Rerun stops the launch with its fix)
        except ImportError:
            raise RuntimeError("viewer:=true needs Rerun: pip install rerun-sdk==0.22.1 "
                               "(the last release that works with the numpy 1.x of ROS 2 Jazzy)") from None
        # A bag starts once the viewer listens (so it sees every pose), and the floor cloud is sent every second.
        params.update({"wait_for_viewer_s": 30.0, "cloud_period_s": 1.0})
        viewer = Node(package="strataslam_ros", executable="strataslam_viewer", namespace=namespace, output="screen")
        actions.append(viewer)
    else:
        viewer = None
    container = ComposableNodeContainer(
        # The node is loaded into the container through a service named after the container: a name of
        # its own per launch, so that two launches never load their nodes into each other's container.
        name=f"strataslam_container_{os.getpid()}",
        namespace=namespace,
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(package="strataslam_ros", plugin="strataslam_ros::StrataslamNode", name="strataslam_node",
                           namespace=namespace, parameters=[params])
        ],
        output="screen",
        # With a bag, Ctrl+C saves the map of what was processed: the final optimization of a large map can
        # take longer than the 5 s that launch waits before escalating to SIGTERM and SIGKILL.
        sigterm_timeout="60",
        sigkill_timeout="60",
    )
    # The launch ends with its container, whatever ended it (a bag processed, a node that could not start or
    # refused to run as a duplicate). Without this, a container that exits before its answer to the node load
    # reaches the launch (a race in the ROS 2 service layer) leaves `ros2 launch` waiting for that answer.
    viewer_running = {"value": viewer is not None}

    def on_container_exit(event, context):  # type: ignore[no-untyped-def]
        if ends_on_slam_exit(viewer_running["value"]):
            return [EmitEvent(event=Shutdown(reason="strataslam container exited"))]
        return [LogInfo(msg="the SLAM has ended: close the viewer window, or press Ctrl+C, to end")]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=container, on_exit=on_container_exit))]
    if viewer is not None:
        def on_viewer_exit(event, context):  # type: ignore[no-untyped-def]
            viewer_running["value"] = False
            if ends_on_viewer_exit(event.returncode):
                return [EmitEvent(event=Shutdown(reason="viewer closed"))]
            return [LogInfo(msg="the viewer was closed; the SLAM goes on (Ctrl+C stops it and saves the map)")]

        handlers.append(RegisterEventHandler(OnProcessExit(target_action=viewer, on_exit=on_viewer_exit)))
    return actions + [container] + handlers


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Path to the robot profile YAML"),
            DeclareLaunchArgument("mode", default_value="map"),
            DeclareLaunchArgument("vocabulary", default_value=""),
            DeclareLaunchArgument("bag", default_value="", description="Process this rosbag2 offline (every message, in order, no drops), save the map and exit"),
            DeclareLaunchArgument("features_topic", default_value=""),
            DeclareLaunchArgument("viewer", default_value="false", description="Live view in Rerun: 3D, top view and level line"),
            DeclareLaunchArgument("namespace", default_value="",
                                  description="Namespace of the node, its topics and services (to run several in one ROS domain)"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
