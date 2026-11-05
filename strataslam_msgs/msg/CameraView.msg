# What the camera tracking sees, published on /strataslam/camera_view for the live view: the image and
# the depth reduced to at most 320 px wide, and each keypoint with what the tracking did with it.
std_msgs/Header header
sensor_msgs/Image image    # rgb8 or mono8; empty with feature-frame input (no image reaches the node)
sensor_msgs/Image depth    # 32FC1 metres, 0 where the depth is invalid or outside depth_min..depth_max
float32[] u                # keypoint pixel coordinates in `image` / `depth`
float32[] v
uint8[] use                # per keypoint, one of:
uint8 UNMATCHED = 0        # no landmark matched it
uint8 REJECTED = 1         # matched, but a PnP outlier or the frame was lost
uint8 INLIER = 2           # used to estimate the pose
uint8 NOT_TRACKED = 3      # the camera is not the odometry source: keypoints kept for loop closures
