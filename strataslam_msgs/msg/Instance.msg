# Announced (transient local) on ~/instance by every strataslam_node, so that two nodes with the same fully
# qualified name in one ROS domain find each other: the newer one (later `started`, or the same time and a
# larger `id`) refuses to run.
builtin_interfaces/Time started
uint64 id
